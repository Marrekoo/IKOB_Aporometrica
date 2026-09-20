"""The reachability gap G (paper, Section 4.4).

    G[i, s] = sum_j D[j, s] [ S_T(t_ij) - f_s(t_ij, c_ij,s) ]

the opportunities that clear the time standard but fail the money standard, in
opportunity units. It is the difference between the time-only accessibility (a
run with the cost gate switched off, `--no-cost-gate`) and the gated
accessibility of the same mode, segment and origin, both with the same
income-matched jobs. Under independence it equals
sum_j D[j, s] S_T(t_ij) [1 - S_M(c_ij,s)]. The atom of the cost margin (the share
for whom no priced trip is acceptable) is reported next to it: it is the part
of the gap that no fare cut can close.

Inputs are accessibility tables (`AccessibilityResult.table`).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

KEY = ["buurtcode", "segment"]


def reachability_gap(gated: pd.DataFrame, time_only: pd.DataFrame, mode: str,
                     value: str = "accessibility") -> pd.DataFrame:
    """G per origin and segment for `mode`: time_only, gated, gap, the gap as
    a share of the time-only accessibility, and the atom."""
    g = gated[gated["mode"] == mode].set_index(KEY)
    t = time_only[time_only["mode"] == mode].set_index(KEY)
    if not g.index.equals(t.index):
        t = t.reindex(g.index)
    out = g[["household_type", "income_class", "population", "atom"]].copy()
    out["time_only"] = t[value]
    out["gated"] = g[value]
    out["gap"] = out["time_only"] - out["gated"]
    out["gap_share"] = out["gap"] / out["time_only"].where(out["time_only"] > 0)
    return out.reset_index()


def summarise(gap: pd.DataFrame, by: str = "income_class") -> pd.DataFrame:
    """Population-weighted mean gap per group (opportunities per person), its
    share of the time-only accessibility and the mean atom."""
    w = gap["population"]
    d = gap.assign(_w=w, _g=gap["gap"] * w, _t=gap["time_only"] * w,
                   _a=gap["atom"] * w)
    s = d.groupby(by)[["_w", "_g", "_t", "_a"]].sum()
    return pd.DataFrame({
        "population": s["_w"],
        "gap": s["_g"] / s["_w"].where(s["_w"] > 0),
        "time_only": s["_t"] / s["_w"].where(s["_w"] > 0),
        "gap_share": s["_g"] / s["_t"].where(s["_t"] > 0),
        "atom": s["_a"] / s["_w"].where(s["_w"] > 0)})
