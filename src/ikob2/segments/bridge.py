"""
Bridge from household-type x income segments to the accessibility engine.

Three inputs meet here:

  * the segment POPULATIONS per zone (ikob2.segments pipeline output),
  * the segment FILTERS: a time margin shared by all segments of a mode
    (e.g. Weibull) and, for priced modes, a per-segment cost margin from
    the reference-budget envelope table (uniform on [low, high] EUR per
    trip, optional atom at zero), composed through a copula,
  * the engine ZONE ORDER (the skim ordering the cost matrices use).

`build_segments` turns the envelope table into engine Segments,
`populations_for_zones` aligns segment persons to the engine's zone
order, and `aggregate_by` reports segment results grouped by household
type or income class. The engine side is SegmentedRunner.run (Shen
competition) or SegmentedRunner.run_hansen (the paper's measure).

Envelope table (CSV or DataFrame), one row per segment:

    household_type, income_class, low, high[, atom]

`low`/`high` are the bounds of the per-trip cost threshold in EUR;
`atom` is the share of the segment for whom no priced trip is
acceptable (a censored cell is atom = 1). Free modes (walking, private
cycling) pass no envelope and get a time-only filter.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from ikob2.domain.filter_config import (
    INDEPENDENCE,
    ClassFilter,
    CopulaSpec,
    CurveSpec,
)
from ikob2.domain.segments import Segment
from ikob2.segments.config import HOUSEHOLD_TYPES, INCOME_CLASSES

logger = logging.getLogger(__name__)

ENVELOPE_COLUMNS = ("household_type", "income_class", "low", "high")


def segment_name(household_type: str, income_class: str) -> str:
    """Same naming as the pipeline's segment columns ('single_D3')."""
    return f"{household_type}_{income_class}"


# ── Envelope table ───────────────────────────────────────────────────

def validate_envelope(
    envelope: pd.DataFrame,
    household_types: Sequence[str] = HOUSEHOLD_TYPES,
    income_classes: Sequence[str] = INCOME_CLASSES,
) -> pd.DataFrame:
    """Checked copy of the envelope table, with an `atom` column."""
    missing = [c for c in ENVELOPE_COLUMNS if c not in envelope.columns]
    if missing:
        raise ValueError(
            f"Envelope table lacks column(s) {missing}; expected "
            f"{list(ENVELOPE_COLUMNS)} (+ optional 'atom').")
    df = envelope.copy()
    if "atom" not in df.columns:
        df["atom"] = 0.0
    df["household_type"] = df["household_type"].astype(str).str.strip()
    df["income_class"] = df["income_class"].astype(str).str.strip()
    for col in ("low", "high", "atom"):
        df[col] = pd.to_numeric(df[col], errors="coerce")

    unknown_t = sorted(set(df["household_type"]) - set(household_types))
    unknown_c = sorted(set(df["income_class"]) - set(income_classes))
    if unknown_t or unknown_c:
        raise ValueError(
            f"Envelope has unknown household type(s) {unknown_t} / income "
            f"class(es) {unknown_c}; known: {list(household_types)} / "
            f"{list(income_classes)}.")
    dup = df[df.duplicated(["household_type", "income_class"], keep=False)]
    if not dup.empty:
        raise ValueError(
            "Envelope has duplicate segment rows: "
            f"{sorted(set(zip(dup.household_type, dup.income_class)))[:5]}")
    bad = df[~np.isfinite(df[["low", "high", "atom"]]).all(axis=1)]
    if not bad.empty:
        raise ValueError(
            "Envelope has missing/non-finite low/high/atom for: "
            f"{list(zip(bad.household_type, bad.income_class))[:5]}")
    bad = df[(df.low < 0) | (df.high < df.low)]
    if not bad.empty:
        raise ValueError(
            "Envelope needs 0 <= low <= high; violated for: "
            f"{list(zip(bad.household_type, bad.income_class))[:5]}")
    bad = df[(df.atom < 0) | (df.atom > 1)]
    if not bad.empty:
        raise ValueError(
            "Envelope atom must be in [0, 1]; violated for: "
            f"{list(zip(bad.household_type, bad.income_class))[:5]}")
    return df.reset_index(drop=True)


def load_envelope(path: str | Path, **kwargs) -> pd.DataFrame:
    return validate_envelope(pd.read_csv(path), **kwargs)


def rescale_budgets(envelope: pd.DataFrame,
                    legs_per_tour: float | Mapping[str, float]) -> pd.DataFrame:
    """Convert per-TOUR budgets to per-one-way-TRIP budgets.

    The reference budgets come from ODiN tours, which are trip chains
    (not necessarily round trips) of one or more legs, while the fare
    matrix prices one one-way trip. A tour budget spread over
    `legs_per_tour` priced legs gives the per-trip budget
    low / legs, high / legs (the km equivalents likewise). 1.0 leaves
    the table as it is: it is then read as a one-way trip budget.

    legs_per_tour : one number for all segments, or a mapping household
        type -> number (every type present in the envelope is needed).
    The atom is a share of households and does not change.
    """
    df = envelope.copy()
    if isinstance(legs_per_tour, Mapping):
        missing = sorted(set(df["household_type"]) - set(legs_per_tour))
        if missing:
            raise KeyError(f"legs_per_tour lacks household type(s) "
                           f"{missing}.")
        legs = df["household_type"].map(legs_per_tour).astype(float)
    else:
        legs = pd.Series(float(legs_per_tour), index=df.index)
    if not np.all(np.isfinite(legs)) or (legs <= 0).any():
        raise ValueError("legs_per_tour must be positive and finite.")
    for col in ("low", "high", "km_low", "km_high"):
        if col in df.columns:
            df[col] = df[col] / legs
    df.attrs["legs_per_tour"] = (dict(legs_per_tour)
                                 if isinstance(legs_per_tour, Mapping)
                                 else float(legs_per_tour))
    return df


def load_reference_budgets(path: str | Path, *,
                           censored: str = "atom",
                           legs_per_tour: float | Mapping[str, float] = 1.0
                           ) -> pd.DataFrame:
    """The reference-budget table (per-trip cost budgets by household
    type and income decile; data/envelope/reference_budgets.csv) as a
    validated envelope.

    Cells the reference budget cannot compute (the first decile: the
    protected basket exhausts the income) have no bounds. `censored`
    decides what they become:

      "atom"  : every priced trip is unacceptable to the segment
                (atom = 1, low = high = 0); free modes still clear;
      "drop"  : the rows are removed, so `envelope_segment_names` (and
                `only=`) leave those segments out of a run;
      "error" : raise.

    legs_per_tour : the basis of the table's budgets. The published
    values are per ODiN tour; the default 1.0 reads them as per ONE-WAY
    TRIP (no conversion). Pass the average number of priced legs per
    tour (a number, or a mapping household type -> number) to divide
    them down to per-trip budgets; see `rescale_budgets`.

    Extra columns (km_low, km_high, the distance equivalents of the
    budgets) are kept.
    """
    if censored not in ("atom", "drop", "error"):
        raise ValueError(f"censored must be 'atom', 'drop' or 'error', "
                         f"got {censored!r}.")
    df = pd.read_csv(path)
    for col in ("low", "high"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    missing = df["low"].isna() | df["high"].isna()
    if missing.any():
        if censored == "error":
            raise ValueError(f"{int(missing.sum())} censored row(s) without "
                             f"bounds in {path}.")
        if censored == "drop":
            df = df[~missing].copy()
        else:
            df["atom"] = 0.0
            df.loc[missing, ["low", "high", "atom"]] = [0.0, 0.0, 1.0]
    return rescale_budgets(validate_envelope(df), legs_per_tour)


def envelope_segment_names(envelope: pd.DataFrame) -> list[str]:
    """Segment names covered by an envelope, for build_segments(only=)."""
    return [segment_name(r.household_type, r.income_class)
            for r in envelope.itertuples()]


# ── Engine segments ──────────────────────────────────────────────────

def build_segments(
    time_curve: CurveSpec,
    *,
    envelope: pd.DataFrame | None = None,
    money_cost_id: str | None = None,
    copula: CopulaSpec = INDEPENDENCE,
    scaling: float = 1.0,
    time_cost_id: str = "time",
    pool_by: str | Callable[[str, str], str] = "none",
    household_types: Sequence[str] = HOUSEHOLD_TYPES,
    income_classes: Sequence[str] = INCOME_CLASSES,
    only: Iterable[str] | None = None,
    cost_curve: Callable | None = None,
) -> list[Segment]:
    """One engine Segment per household-type x income-class cell.

    time_curve : the mode's time margin (shared by all segments).
    envelope : cost-margin table for priced modes; None gives time-only
        filters (free modes), which all share a single composed matrix.
        With an envelope, every requested segment needs a row.
    money_cost_id : key of the money matrix in the runner's
        cost_matrices; required iff an envelope is given.
    pool_by : "none" (one competition/supply pool "default"),
        "income_class" (pool = income class, e.g. for income-matched
        jobs), or a callable (household_type, income_class) -> pool.
    only : restrict to these segment names.
    cost_curve : optional function envelope row -> CurveSpec replacing the
        uniform cost margin (the exponential specifications M1, M1').
    """
    if envelope is not None and money_cost_id is None:
        raise ValueError("An envelope needs money_cost_id (the key of "
                         "the money matrix in cost_matrices).")
    if envelope is None and money_cost_id is not None:
        raise ValueError("money_cost_id given without an envelope; the "
                         "time-only filter has no cost margin.")

    if callable(pool_by):
        pool_of = pool_by
    elif pool_by == "none":
        def pool_of(t, c):
            return "default"
    elif pool_by == "income_class":
        def pool_of(t, c):
            return c
    else:
        raise ValueError(f"Unknown pool_by {pool_by!r}; use 'none', "
                         f"'income_class' or a callable.")

    rows = None
    if envelope is not None:
        env = validate_envelope(envelope, household_types, income_classes)
        rows = {(r.household_type, r.income_class): r
                for r in env.itertuples(index=False)}

    wanted = None if only is None else set(only)
    cells = [(t, c) for c in income_classes for t in household_types]
    if wanted is not None:
        unknown = wanted - {segment_name(t, c) for t, c in cells}
        if unknown:
            raise KeyError(f"Unknown segment name(s) in `only`: "
                           f"{sorted(unknown)}")
        cells = [(t, c) for t, c in cells if segment_name(t, c) in wanted]

    if rows is not None:
        missing = [segment_name(t, c) for t, c in cells if (t, c) not in rows]
        if missing:
            raise KeyError(
                f"Envelope has no row for {len(missing)} segment(s), e.g. "
                f"{missing[:5]}. Add the rows, or restrict with `only=`.")

    segments = []
    for t, c in cells:
        cost = None
        if rows is not None:
            r = rows[(t, c)]
            cost = (cost_curve(r) if cost_curve is not None else
                    CurveSpec("uniform", (float(r.low), float(r.high)),
                              atom=float(r.atom)))
        cf = ClassFilter(time=time_curve, cost=cost, copula=copula,
                         scaling=scaling)
        segments.append(Segment(
            name=segment_name(t, c), income=c, car_access=None,
            preference=None, class_filter=cf,
            time_cost_id=time_cost_id,
            money_cost_id=money_cost_id if cost is not None else None,
            pool=pool_of(t, c), household_type=t))
    return segments


# ── Populations ──────────────────────────────────────────────────────

def populations_for_zones(
    table: pd.DataFrame,
    zone_codes: Sequence[str],
    segments: Sequence[Segment],
) -> dict[str, np.ndarray]:
    """Segment persons per engine zone: name -> (n_zones,) float32.

    `table` is a pipeline output layer (household_based or
    population_scaled) with a `buurtcode` column and one column per
    segment. Zones without a row get zero population (logged); NaN
    counts are zero. Engine zones must be unique.
    """
    codes = [str(c).strip() for c in zone_codes]
    if len(set(codes)) != len(codes):
        raise ValueError("zone_codes contains duplicates.")
    if "buurtcode" not in table.columns:
        raise ValueError("Population table lacks a 'buurtcode' column.")
    tab = table.set_index("buurtcode")
    if not tab.index.is_unique:
        raise ValueError("Population table has duplicate buurtcodes.")

    lacking = [s.name for s in segments if s.name not in tab.columns]
    if lacking:
        raise KeyError(f"Population table lacks segment column(s): "
                       f"{lacking[:5]}")
    aligned = tab.reindex(codes)
    n_missing = int(aligned.iloc[:, 0].isna().sum()) if len(codes) else 0
    if n_missing:
        logger.warning("%d of %d engine zones have no segment population "
                       "row; treated as empty.", n_missing, len(codes))
    return {
        s.name: np.nan_to_num(aligned[s.name].to_numpy(float),
                              nan=0.0).astype(np.float32)
        for s in segments
    }


# ── Reporting ────────────────────────────────────────────────────────

def aggregate_by(
    per_segment: Mapping[str, np.ndarray],
    populations: Mapping[str, np.ndarray],
    segments: Sequence[Segment],
    key: str = "income_class",
) -> dict[str, np.ndarray]:
    """Population-weighted mean accessibility per group of segments.

    key : "income_class" (Segment.income), "household_type", or "all".
    Zones where a group has no population get NaN.
    """
    if key not in ("income_class", "household_type", "all"):
        raise ValueError(f"Unknown key {key!r}.")

    def group(s: Segment) -> str:
        if key == "income_class":
            return str(s.income)
        if key == "household_type":
            return str(s.household_type)
        return "all"

    num: dict[str, np.ndarray] = {}
    den: dict[str, np.ndarray] = {}
    for s in segments:
        g = group(s)
        pop = np.asarray(populations[s.name], dtype=np.float64)
        a = np.asarray(per_segment[s.name], dtype=np.float64)
        num[g] = num.get(g, 0.0) + a * pop
        den[g] = den.get(g, 0.0) + pop
    return {g: np.where(den[g] > 0, num[g] / np.where(den[g] > 0, den[g], 1.0),
                        np.nan)
            for g in num}
