import math
import numpy as np
from pathlib import Path

import logging

from ikob2.core.numerics import DTYPE, IKOB_INFINITE


logger = logging.getLogger(__name__)

# Fraction of skim rows that may be discarded as sentinels before the
# loader refuses to continue. A genuine skim never has a majority of
# its pairs unreachable; if this trips, the sentinel threshold is wrong
# for the file, not the other way around.
MAX_SENTINEL_SKIP_FRAC = 0.5

# Fraction above which we warn (a few unreachable islands are normal;
# more than this deserves a look).
WARN_SENTINEL_SKIP_FRAC = 0.01

# Distance skims are expected in METERS. If the 99th percentile of a
# distance skim is below this, the file is almost certainly in
# kilometers and the generalized-cost distance term would be silently
# wrong by a factor 1000.
MIN_PLAUSIBLE_DIST_P99_METERS = 500.0

# ─────────────────────────────────────────────
# Correct IKOB → LMS region mapping
# ─────────────────────────────────────────────

IKOB_TO_REGION: dict[str, str] = {}

for i in range(1, 5):
    IKOB_TO_REGION[f"IKOB{str(i).zfill(2)}"] = "Noord"

for i in range(5, 11):
    IKOB_TO_REGION[f"IKOB{str(i).zfill(2)}"] = "Oost"

for i in list(range(11, 20)) + [26]:
    IKOB_TO_REGION[f"IKOB{str(i).zfill(2)}"] = "West"

for i in range(20, 26):
    IKOB_TO_REGION[f"IKOB{str(i).zfill(2)}"] = "Zuid"


# ─────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────

def _load_omnummer(path: Path) -> np.ndarray:
    """
    Returns LMS IDs in IKOB order (int64 array; row i = IKOB zone i+1).

    Hardened against the Excel-export variants observed in practice
    (IKOB13): UTF-8 BOM, semicolon delimiter, header line — detected
    per file instead of assumed. Refuses anything it cannot prove is
    an N x >=2 integer table with a contiguous 1..N first column.

    LMS ID 0 is accepted as an explicit "no mapping" placeholder and
    returned as-is (with a warning); the caller must exclude such
    zones from the LMS->index mapping. Negative or non-integer IDs
    are an error.
    """
    # encoding="utf-8-sig" strips the BOM that Excel's "CSV UTF-8"
    # format prepends; harmless for plain ASCII/UTF-8 files.
    with open(path, encoding="utf-8-sig") as fh:
        first = fh.readline()
    if not first.strip():
        raise ValueError(f"{path.name}: file is empty")

    # Delimiter sniff. None = whitespace (genfromtxt's default split).
    delimiter = next((c for c in (";", ",", "\t") if c in first), None)

    # Header detection: a data line starts with an integer zone number.
    fields = first.split(delimiter) if delimiter else first.split()
    skip = 0 if fields[0].strip().lstrip("+-").isdigit() else 1

    logger.info(
        "%s: detected delimiter %r, %d header line(s)",
        path.name, delimiter if delimiter else "whitespace", skip,
    )

    data = np.genfromtxt(
        path,
        delimiter=delimiter,
        skip_header=skip,
        dtype=np.float64,
        encoding="utf-8-sig",
    )
    # A single-row file squeezes to 1-D; restore the row axis so the
    # shape check below means what it says.
    if data.ndim == 1:
        data = data.reshape(1, -1)

    if data.ndim != 2 or data.shape[1] < 2:
        raise ValueError(
            f"{path.name}: expected an N x >=2 Omnummer table, got shape "
            f"{data.shape} (delimiter={delimiter!r}, skip_header={skip}). "
            f"First line: {first.strip()!r}"
        )

    # genfromtxt encodes unparseable fields as NaN SILENTLY; that
    # silence is how a malformed file becomes a crash (or a wrong
    # zone mapping) far from its cause. Refuse it here, with line
    # numbers.
    nan_rows = np.flatnonzero(np.isnan(data).any(axis=1))
    if len(nan_rows):
        raise ValueError(
            f"{path.name}: non-numeric entries at file line(s) "
            f"{(nan_rows[:5] + 1 + skip).tolist()} "
            f"(delimiter={delimiter!r}) — inspect the file format"
        )

    if np.mod(data[:, :2], 1).any():
        raise ValueError(
            f"{path.name}: zone columns contain non-integer values — "
            f"this is not an Omnummer table"
        )

    data = data.astype(np.int64)

    # Row-order invariant: downstream code indexes matrices by row
    # position, so row i MUST be IKOB zone i+1. A file with a deleted
    # or reordered row would otherwise shift every zone silently.
    expected = np.arange(1, len(data) + 1, dtype=np.int64)
    if not np.array_equal(data[:, 0], expected):
        bad = np.flatnonzero(data[:, 0] != expected)
        raise ValueError(
            f"{path.name}: first column is not contiguous 1..{len(data)}; "
            f"first mismatch at file line {int(bad[0]) + 1 + skip} "
            f"(found {int(data[bad[0], 0])}, expected {int(expected[bad[0]])})"
        )

    ids = data[:, 1]

    if (ids < 0).any():
        bad_zones = (np.flatnonzero(ids < 0) + 1).tolist()
        raise ValueError(
            f"{path.name}: negative LMS IDs for IKOB zone(s) {bad_zones[:10]}"
        )

    zero_zones = (np.flatnonzero(ids == 0) + 1).tolist()
    if zero_zones:
        logger.warning(
            "%s: %d IKOB zone(s) with LMS ID 0 (no mapping): %s%s — these "
            "zones will be unreachable in every skim-derived matrix. If "
            "they are deliberately unused, confirm with the data owner "
            "and record it; if not, the Omnummer table is incomplete.",
            path.name, len(zero_zones), zero_zones[:10],
            " ..." if len(zero_zones) > 10 else "",
        )

    return ids


def _sniff_skim_format(path: Path, sample_bytes: int = 65536) -> tuple[str, bool, int]:
    """
    Detect (delimiter, decimal_comma, n_header_lines) for a long skim.

    Relies on the guarantee that the first two columns are integer zone
    numbers: the first ',' or ';' on a data line is therefore always a
    delimiter, never a decimal separator.

        ';' first  ->  semicolon-delimited, decimal commas (1;2;9,75)
        ',' first  ->  comma-delimited, decimal points     (1,2,9.75)

    Header presence is detected the same way: a data line starts with
    an integer zone number; a header line does not.
    """
    with open(path) as f:
        sample = f.read(sample_bytes)

    lines = sample.splitlines()
    if len(lines) > 1:
        lines = lines[:-1]   # last line may be truncated by the byte cut
    if not lines:
        raise ValueError(f"{path.name}: file is empty")

    delim = next((ch for ch in lines[0] if ch in ",;"), None)
    if delim is None:
        raise ValueError(
            f"{path.name}: no ',' or ';' on the first line — "
            f"not a recognizable long skim"
        )

    # Header detection: a data line starts with an integer zone number.
    try:
        int(lines[0].split(delim)[0])
        skip = 0
    except ValueError:
        skip = 1

    # Guard against the ambiguous case: a comma-delimited file that ALSO
    # used decimal commas would sniff as comma-delimited and misparse
    # ('1,2,9,75' -> origin 1, dest 2, value 9). A 3-column row has
    # exactly 2 delimiters; verify that invariant on the sample.
    bad = [i for i, ln in enumerate(lines[skip:], start=skip + 1)
           if ln.strip() and ln.count(delim) != 2]
    if bad:
        raise ValueError(
            f"{path.name}: line(s) {bad[:5]} do not have exactly 3 "
            f"'{delim}'-separated fields — mixed or ambiguous format"
        )

    return delim, delim == ";", skip


def _load_long_skim(path: Path) -> np.ndarray:
    """
    Load long LMS skim (origin/dest/value) as float32, auto-detecting
    delimiter, decimal convention, and header presence per file.

    Known variants:
      - semicolon-delimited, decimal comma, with header  (1;2;9,75)
      - comma-delimited, decimal point, headerless       (1,2,9.75)

    Returns array shape (N, 3): [origin, destination, value]
    """
    delim, decimal_comma, skip = _sniff_skim_format(path)
    logger.info(
        "%s: detected delimiter '%s', decimal %s, %d header line(s)",
        path.name, delim, "comma" if decimal_comma else "point", skip,
    )

    with open(path) as f:
        for _ in range(skip):
            next(f)
        stream = (
            (line.replace(",", ".") for line in f) if decimal_comma else f
        )
        data = np.loadtxt(
            stream,
            delimiter=delim,
            usecols=(0, 1, 2),
            dtype=np.float32,
        )

    nan_frac = np.isnan(data[:, 2]).mean()
    if nan_frac > 0:
        raise ValueError(
            f"{path.name}: {100 * nan_frac:.1f}% of values NaN "
            f"after parsing — inspect the file format"
        )

    return data


def _fill_matrix(
    long_data: np.ndarray,
    lms_to_idx: dict[int, np.ndarray],
    n: int,
    fill_value: float,
    unreachable_value: float | None,
    skim_name: str,
) -> np.ndarray:
    """
    Expand a long LMS skim into a dense IKOB-order matrix.

    Cells never touched by the skim keep `fill_value`.

    `unreachable_value` is the sentinel used by THIS skim, in THIS
    skim's units (e.g. 9999.0 minutes for a time skim). Pass None if
    the skim has no sentinel; no values are then filtered. NaN values
    are always skipped (cell keeps fill_value), so the result contains
    no NaN.

    Raises if more than MAX_SENTINEL_SKIP_FRAC of rows are discarded
    as sentinels — that always means the threshold does not match the
    file's units.
    """
    threshold = math.inf if unreachable_value is None else unreachable_value

    matrix = np.full((n, n), fill_value, dtype=DTYPE)

    n_rows = len(long_data)
    n_sentinel = 0
    n_unknown_id = 0

    for origin_f, dest_f, val in long_data:
        # NaN-safe: NaN fails this comparison and is skipped too,
        # but _load_long_skim already guarantees no NaN.
        if not (val < threshold):
            n_sentinel += 1
            continue

        rows = lms_to_idx.get(int(origin_f))
        if rows is None:
            n_unknown_id += 1
            continue

        cols = lms_to_idx.get(int(dest_f))
        if cols is None:
            n_unknown_id += 1
            continue

        matrix[np.ix_(rows, cols)] = val

    sentinel_frac = n_sentinel / n_rows if n_rows else 0.0

    logger.info(
        "%s: %d rows, %d sentinel-skipped (%.2f%%), %d unknown-ID-skipped",
        skim_name, n_rows, n_sentinel, 100 * sentinel_frac, n_unknown_id,
    )

    if sentinel_frac > MAX_SENTINEL_SKIP_FRAC:
        raise ValueError(
            f"{skim_name}: {100 * sentinel_frac:.1f}% of rows discarded as "
            f"sentinels (>= {threshold}). The sentinel threshold does not "
            f"match this skim's units — refusing to build a mostly-empty "
            f"matrix."
        )

    if sentinel_frac > WARN_SENTINEL_SKIP_FRAC:
        logger.warning(
            "%s: %.2f%% of rows discarded as sentinels (>= %s) — "
            "verify this is plausible for the region.",
            skim_name, 100 * sentinel_frac, threshold,
        )

    return matrix


def _check_coverage(
    long_data: np.ndarray,
    lms_to_idx: dict[int, np.ndarray],
    skim_name: str,
    ikob_id: str,
    scenario: str,
) -> None:
    """Warn about LMS IDs that never appear as origin in a skim."""
    covered = set(int(o) for o in long_data[:, 0])
    missing = [k for k in lms_to_idx if k not in covered]
    if missing:
        logger.warning(
            "%s %s: %d LMS zone IDs absent from %s "
            "(their rows keep the fill value): %s%s",
            ikob_id, scenario, len(missing), skim_name,
            missing[:10], " ..." if len(missing) > 10 else "",
        )


def _check_dist_units(long_data: np.ndarray, skim_name: str) -> None:
    """
    Warn if a distance skim looks like kilometers instead of meters.
    A km-valued file parses without error but would make the
    generalized-cost distance term wrong by a factor 1000.
    """
    values = long_data[:, 2]
    values = values[values > 0]
    if len(values) == 0:
        return
    p99 = float(np.percentile(values, 99))
    if p99 < MIN_PLAUSIBLE_DIST_P99_METERS:
        logger.warning(
            "%s: 99th percentile of distance values is %.1f — implausibly "
            "small for meters at LMS scale. Is this file in KILOMETERS?",
            skim_name, p99,
        )


# ─────────────────────────────────────────────
# Raw skim matrices (parameter-free, cacheable)
# ─────────────────────────────────────────────

def load_ikob_raw_matrices(
    ikob_id: str,
    scenario: str,
    skim_root: Path,
    omnummertabellen_root: Path,
    time_unreachable_value: float = 9999.0,
    dist_unreachable_value: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build the dense, parameter-free skim matrices for one IKOB.

    Sentinels are per-skim, in each skim's own units:
      - time skim: minutes, sentinel 9999.0 by default
      - dist skim: meters, no sentinel by default (a meters-valued
        skim with a 9999 threshold would silently discard every pair
        beyond ~10 km; pass a value only if the distance files are
        confirmed to use one)

    IKOB zones whose Omnummer entry is 0 (explicitly unmapped) are
    excluded from the LMS mapping: their rows/columns keep the fill
    values (time = IKOB_INFINITE, dist = 0.0), i.e. they are
    unreachable. _load_omnummer warns about them.

    Returns:
        lms_ids      : (n,)   int64, LMS IDs in IKOB order (0 = unmapped)
        time_matrix  : (n, n) float32, minutes; IKOB_INFINITE = unreachable
        dist_matrix  : (n, n) float32, meters; 0.0 = missing pair
    """

    try:
        region = IKOB_TO_REGION[ikob_id]
    except KeyError:
        raise ValueError(f"Unknown IKOB ID: {ikob_id}")

    omnummer_path = omnummertabellen_root / f"{ikob_id}_Omnummer.csv"
    lms_ids = _load_omnummer(omnummer_path)
    n = len(lms_ids)

    # LMS ID → array of IKOB indices (one-to-many). LMS ID 0 means
    # "explicitly unmapped" and must NOT enter the mapping: if a skim
    # ever contained a 0-ID row (e.g. as its own sentinel), it would
    # otherwise write real-looking values into a zone that has none.
    _tmp: dict[int, list[int]] = {}
    for i, lms_id in enumerate(lms_ids):
        if lms_id <= 0:
            continue  # warned in _load_omnummer
        _tmp.setdefault(int(lms_id), []).append(i)
    lms_to_idx = {k: np.asarray(v, dtype=np.intp) for k, v in _tmp.items()}

    skim_dir = skim_root / region / scenario

    time_name = "reistijd_auto_ochtendspits.csv"
    dist_name = "afstand_auto_freeflow.csv"

    time_data = _load_long_skim(skim_dir / time_name)
    dist_data = _load_long_skim(skim_dir / dist_name)

    _check_dist_units(dist_data, dist_name)

    time_matrix = _fill_matrix(
        time_data, lms_to_idx, n,
        fill_value=IKOB_INFINITE,
        unreachable_value=time_unreachable_value,
        skim_name=time_name,
    )
    dist_matrix = _fill_matrix(
        dist_data, lms_to_idx, n,
        fill_value=0.0,
        unreachable_value=dist_unreachable_value,
        skim_name=dist_name,
    )

    # Pairs unreachable in time get no meaningful distance either:
    # propagate the time skim's unreachability so dist_matrix never
    # carries a value for a pair the model treats as unreachable.
    unreachable = time_matrix >= IKOB_INFINITE
    dist_matrix[unreachable] = 0.0

    _check_coverage(time_data, lms_to_idx, time_name, ikob_id, scenario)
    _check_coverage(dist_data, lms_to_idx, dist_name, ikob_id, scenario)

    return lms_ids, time_matrix, dist_matrix


# ─────────────────────────────────────────────
# Beta combination (cheap, run every iteration)
# ─────────────────────────────────────────────

def combine_generalized_cost(
    time_matrix: np.ndarray,
    dist_matrix: np.ndarray,
    beta_time: float = 1.0,
    beta_distance: float = 0.0,
    unreachable_value: float = 9999.0,
) -> np.ndarray:
    """
    Generalized cost from raw skim matrices. ~10 ms for n=1357,
    so betas can vary freely between iterations without touching
    the CSVs.

    beta_distance is per KILOMETER (dist_matrix holds meters; the
    0.001 below converts).
    """
    valid = time_matrix < unreachable_value

    cost = np.full(time_matrix.shape, IKOB_INFINITE, dtype=DTYPE)
    cost[valid] = (
        beta_time * time_matrix[valid]
        + beta_distance * (dist_matrix[valid] * 0.001)
    )
    return cost


# ─────────────────────────────────────────────
# Backwards-compatible entry point
# ─────────────────────────────────────────────

def load_ikob_cost_matrix(
    ikob_id: str,
    scenario: str,
    skim_root: Path,
    omnummertabellen_root: Path,
    beta_time: float = 1.0,
    beta_distance: float = 0.0,
    time_unreachable_value: float = 9999.0,
    dist_unreachable_value: float | None = None,
) -> np.ndarray:
    """
    Original one-shot API: parse CSVs and combine in one call.
    Prefer the container (ikob2.data.container) for repeated runs.
    """
    _, time_matrix, dist_matrix = load_ikob_raw_matrices(
        ikob_id=ikob_id,
        scenario=scenario,
        skim_root=skim_root,
        omnummertabellen_root=omnummertabellen_root,
        time_unreachable_value=time_unreachable_value,
        dist_unreachable_value=dist_unreachable_value,
    )
    return combine_generalized_cost(
        time_matrix, dist_matrix,
        beta_time=beta_time,
        beta_distance=beta_distance,
        unreachable_value=time_unreachable_value,
    )