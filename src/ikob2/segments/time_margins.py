"""
Time margins: Weibull acceptable-travel-time distributions per mode and
job type.

The time margin S_T(t) of the non-exponential specifications is a
Weibull, fitted by interval-censored maximum likelihood to the stated
maximum acceptable commuting times of transport professionals (paper,
Section 3.2), separately for each mode (bike, public transport, car) and
for whether the job admits working from home (WFH). The fits live in
`data/margins/S_T_work.csv`:

    wfh, mode, eta (scale, minutes), k (shape), median, class (IFR/DFR)

k > 1 in every cell (increasing hazard): the soft-threshold corner of
the threshold family. `median` is stored in the file and is checked
against eta * ln(2)^(1/k) on loading, so a mangled row is caught.

The WFH split describes the JOB (destination), not the traveller: which
curve applies to a destination depends on whether its jobs admit
working from home. Until jobs are split that way, a run picks one of the
two curves for all jobs.
"""

from __future__ import annotations

import math
from pathlib import Path

import pandas as pd

from ikob2.domain.filter_config import CurveSpec

MODE_NAMES = {"bike": "bike", "public transport": "pt", "pt": "pt",
              "car": "car", "walk": "walk"}
WFH_NAMES = {"no home working": "no_wfh", "home working possible":
             "wfh_possible", "no_wfh": "no_wfh", "wfh_possible":
             "wfh_possible"}


def load_time_margins(path: str | Path) -> dict[tuple[str, str], CurveSpec]:
    """{(mode, wfh): CurveSpec('weibull', (k, eta))} from the fit table.

    Modes are 'bike', 'pt', 'car' (the names of skims.router.MODES) and
    wfh is 'no_wfh' or 'wfh_possible'. Checks: positive shape and scale,
    the stored median against eta * ln(2)^(1/k) (relative tolerance
    1e-6), the stored class against k (IFR iff k > 1), no duplicates.
    """
    df = pd.read_csv(path)
    need = ["wfh", "mode", "eta", "k", "median", "class"]
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise KeyError(f"{path}: missing column(s) {missing}.")
    out: dict[tuple[str, str], CurveSpec] = {}
    for r in df.itertuples(index=False):
        wfh = WFH_NAMES.get(str(r.wfh).strip().lower())
        mode = MODE_NAMES.get(str(r.mode).strip().lower())
        if wfh is None or mode is None:
            raise ValueError(f"Unknown wfh/mode in row: {r.wfh!r}, "
                             f"{r.mode!r}.")
        eta, k = float(r.eta), float(r.k)
        if not (math.isfinite(eta) and eta > 0 and math.isfinite(k)
                and k > 0):
            raise ValueError(f"({mode}, {wfh}): eta and k must be positive "
                             f"and finite, got eta={eta}, k={k}.")
        median = eta * math.log(2.0) ** (1.0 / k)
        if not math.isclose(median, float(r.median), rel_tol=1e-6):
            raise ValueError(f"({mode}, {wfh}): stored median {r.median} "
                             f"differs from eta*ln(2)^(1/k) = {median}.")
        klass = "IFR" if k > 1 else "CFR" if k == 1 else "DFR"
        if str(r._5).strip().upper() != klass:
            raise ValueError(f"({mode}, {wfh}): class {r._5!r} does not "
                             f"match shape k={k} ({klass}).")
        if (mode, wfh) in out:
            raise ValueError(f"Duplicate row for ({mode}, {wfh}).")
        out[(mode, wfh)] = CurveSpec("weibull", (k, eta))
    return out


def time_margin(margins: dict, mode: str, wfh: str = "no_wfh") -> CurveSpec:
    """The Weibull time margin for a mode and job type."""
    try:
        return margins[(mode, wfh)]
    except KeyError:
        raise KeyError(f"No time margin for mode '{mode}', job type "
                       f"'{wfh}'; available: {sorted(margins)}.") from None
