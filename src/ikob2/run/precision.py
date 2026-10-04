"""
How many digits a result deserves: the spread of a statistic over runs whose
inputs were drawn within the rounding of their sources (run.perturb).

For a statistic with reference value x (the unperturbed run) and standard
deviation u over the draws, the number of significant digits is

    n = floor(log10 |x|) - floor(log10 u) + 1,

the last digit at the decimal position of the leading digit of u, the rule
of the Guide to the Expression of Uncertainty in Measurement (JCGM 100:2008,
7.2.6) with u quoted to one digit. It covers the rounding of the data only;
model structure (the specification) and parameters add their own spread.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

KEY = ["buurtcode", "segment"]


def significant_digits(x: float, u: float) -> float:
    """Significant digits of `x` with standard uncertainty `u` (NaN when u
    is zero or x is zero: no digit is uncertain, or there is nothing to
    count)."""
    if not (np.isfinite(x) and np.isfinite(u)) or u <= 0 or x == 0:
        return np.nan
    return float(max(1, math.floor(math.log10(abs(x))) - math.floor(math.log10(u)) + 1))


def round_significant(x: float, digits: float) -> float:
    """`x` rounded to `digits` significant digits (x itself when digits is
    NaN)."""
    if not np.isfinite(digits) or x == 0 or not np.isfinite(x):
        return x
    return float(round(x, int(digits) - 1 - math.floor(math.log10(abs(x)))))


def decile_means(table: pd.DataFrame, mode: str, value: str = "accessibility") -> pd.Series:
    """Population-weighted mean of `value` per income class for `mode`."""
    t = table[table["mode"] == mode]
    w = t["population"]
    return (t[value] * w).groupby(t["income_class"]).sum() / w.groupby(t["income_class"]).sum()


def gains(base: pd.DataFrame, run: pd.DataFrame, mode: str) -> pd.DataFrame:
    """Per (origin, segment): gain of `run` over `base`, with the population
    and income class of the base."""
    a = base[base["mode"] == mode].set_index(KEY)
    b = run[run["mode"] == mode].set_index(KEY)["accessibility"].reindex(a.index)
    return a[["population", "income_class"]].assign(gain=b - a["accessibility"])


def statistics(runs: dict, mode: str, costs: dict | None = None) -> pd.DataFrame:
    """The statistics of one set of runs (one draw, or the reference):
    `runs` maps a scenario to its accessibility table and must contain s0;
    `costs` maps a scenario to its yearly public cost. Rows: statistic,
    scenario, income_class (or 'all'), value."""
    rows = []
    base = runs["s0"]
    for cls, v in decile_means(base, mode).items():
        rows.append(("level", "s0", cls, float(v)))
    for scen, run in runs.items():
        if scen == "s0":
            continue
        g = gains(base, run, mode)
        w = g["population"]
        per = (g["gain"] * w).groupby(g["income_class"]).sum() / w.groupby(g["income_class"]).sum()
        for cls, v in per.items():
            rows.append(("gain_per_person", scen, cls, float(v)))
        total = float((g["gain"] * w).sum())
        rows.append(("gain_job_persons", scen, "all", total))
        c = (costs or {}).get(scen)
        if c:
            rows.append(("gain_per_eur", scen, "all", total / c))
    return pd.DataFrame(rows, columns=["statistic", "scenario", "income_class", "value"])


def summarise(reference: pd.DataFrame, draws: pd.DataFrame) -> pd.DataFrame:
    """Per statistic: the reference value, the mean and standard deviation
    over the draws (column `draw`), the relative standard deviation, the
    significant digits of the reference and the reference rounded to them."""
    key = ["statistic", "scenario", "income_class"]
    spread = draws.groupby(key)["value"].agg(mean="mean", sd="std", draws="count").reset_index()
    out = reference.rename(columns={"value": "reference"}).merge(spread, on=key, how="left")
    out["rel_sd"] = out["sd"] / out["reference"].abs().where(out["reference"] != 0)
    out["digits"] = [significant_digits(x, u) for x, u in zip(out["reference"], out["sd"])]
    out["rounded"] = [round_significant(x, n) for x, n in zip(out["reference"], out["digits"])]
    return out
