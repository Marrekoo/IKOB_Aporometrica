"""
The income axis: the standardised income of each
decile, EUR/month, from CBS percentiles.

  * D2..D9: the within-decile median (percentiles 15, 25, ..., 85),
    interpolated linearly between log income and the normal score;
  * D1: the 10th percentile, an upper bound: the decile is censored (no
    envelope below the social-assistance anchor);
  * D10: the 95th percentile of a Pareto tail through p80 and p90
    (`top_decile_rule` "pareto_p95"), or p90 itself ("boundary_p90").
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm

from ikob2.envelope.sources import Sources


def income_axis(src: Sources, prm) -> pd.DataFrame:
    """Columns quantile (Q1..Q10), y_std (EUR/month), env_eligible."""
    e = prm.envelope
    p = src["cbs_income_percentiles"]
    p = p[(p["measure"] == e.income_measure) & (p["year"] == e.income_year)]
    p = p.sort_values("p")
    if len(p) != 9:
        raise ValueError(f"Need p10..p90 of {e.income_measure} {e.income_year}.")
    pg = p["p"].to_numpy(float)
    vg = p["k_eur_year"].to_numpy(float) * 1000.0 / e.months_per_year
    if not np.all(np.diff(vg) > 0):
        raise ValueError("CBS percentiles are not increasing.")
    at = dict(zip(np.round(pg * 100).astype(int), vg))
    mids = np.arange(15, 90, 10) / 100.0
    interior = np.exp(np.interp(norm.ppf(mids), norm.ppf(pg), np.log(vg)))
    alpha = np.log(0.20 / 0.10) / np.log(at[90] / at[80])
    top = {"pareto_p95": at[90] * (0.10 / 0.05) ** (1.0 / alpha),
           "boundary_p90": at[90]}[e.top_decile_rule]
    return pd.DataFrame({
        "quantile": [f"Q{k}" for k in range(1, 11)],
        "y_std": [at[10], *interior, top],
        "env_eligible": [False] + [True] * 9})
