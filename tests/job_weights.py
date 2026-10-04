"""Job weights in closed form, for hand computations in the tests.

Every job of a sector at one wage: sectors ranked by wage and laid along the
income-rank axis [0, 1] in proportion to their jobs, decile Dk covering
[(k-1)/10, k/10], so that

    W[k, s] = |decile k ∩ sector s| / |sector s|      (columns sum to 1).

This is the limit of `segments.occupations.occupation_job_weights` with one
occupation per sector and no wage spread within cells
(test_occupations.test_one_occupation_per_sector_and_no_spread_is_the_rank_partition),
and simple enough to compute by hand.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ikob2.segments.config import INCOME_CLASSES


def rank_weights(wage: pd.Series, jobs: pd.Series,
                 income_classes=INCOME_CLASSES) -> pd.DataFrame:
    """Income class x sector weights; 'onbekend' sees all jobs."""
    jobs = jobs.reindex(wage.index).astype(float)
    order = wage.astype(float).sort_values(kind="stable").index
    edges = np.concatenate([[0.0], np.cumsum(jobs[order]) / jobs.sum()])
    ranked = [c for c in income_classes if c != "onbekend"]
    n = len(ranked)
    W = pd.DataFrame(0.0, index=list(income_classes), columns=wage.index)
    for k, cls in enumerate(ranked):
        lo, hi = k / n, (k + 1) / n
        for i, s in enumerate(order):
            length = edges[i + 1] - edges[i]
            if length > 0:
                W.loc[cls, s] = max(0.0, min(hi, edges[i + 1]) - max(lo, edges[i])) / length
    if "onbekend" in income_classes:
        W.loc["onbekend"] = 1.0
    return W


def split_by_share(sector_jobs: pd.DataFrame, share: pd.Series):
    """(jobs without home working, jobs with home working), buurt x sector."""
    wfh = sector_jobs.mul(share.reindex(sector_jobs.columns), axis=1)
    return sector_jobs - wfh, wfh


def by_type(wage: pd.Series, jobs: pd.Series, share: pd.Series,
            types=("no_wfh", "wfh_possible")) -> dict[str, pd.DataFrame]:
    """Weights per job type: `rank_weights` times the home-working share of
    each sector (the `job_weights` argument of run_accessibility)."""
    W = rank_weights(wage, jobs)
    s = share.reindex(W.columns)
    return {types[0]: W * (1.0 - s), types[1]: W * s}
