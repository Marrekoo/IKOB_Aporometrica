"""
Job supply per income-matched pool.

The paper measures opportunities D_{j,s}: jobs matched to a segment's
income. Jobs come per buurt and LISA sector (see jobs_impute); each
sector is placed on the income-rank axis by its mean wage, and the
income-decile pools PARTITION the jobs (`sector_income_weights`,
`sector_pools`). This module also reads the legacy IKOB job table
(four income groups per buurt), whose buurt totals are the row marginal
of the sector imputation. See docs/data_lineage.md.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from ikob2.segments.config import INCOME_CLASSES

logger = logging.getLogger(__name__)

JOB_GROUPS = ("laag", "middellaag", "middelhoog", "hoog")   # low -> high


# ── Legacy table ─────────────────────────────────────────────────────

def parse_legacy_jobs(raw: pd.DataFrame, year: str = "2018") -> pd.DataFrame:
    """Jobs per buurt and income group from the legacy pivot sheet.

    `raw` is the 'buurten-arbeidsplaatsen' sheet of Alle_Zones_2030_2040
    read with header=2: first column the buurt code, then columns named
    'Som van arb_<year>_<group>'. Non-buurt rows (totals) are dropped.
    Returns a frame indexed by buurtcode with one column per job group.
    """
    df = raw.copy()
    df = df.rename(columns={df.columns[0]: "buurtcode"})
    df["buurtcode"] = df["buurtcode"].astype(str).str.strip()
    df = df[df["buurtcode"].str.match(r"^BU\d+")]

    cols = {}
    for g in JOB_GROUPS:
        matches = [c for c in df.columns
                   if str(c).strip().endswith(f"arb_{year}_{g}")]
        if len(matches) != 1:
            raise KeyError(
                f"Expected exactly one column ending 'arb_{year}_{g}', "
                f"found {matches}; available: {list(df.columns)}")
        cols[matches[0]] = g
    out = df[["buurtcode", *cols]].rename(columns=cols)
    for g in JOB_GROUPS:
        out[g] = pd.to_numeric(out[g], errors="coerce")
    if out["buurtcode"].duplicated().any():
        raise ValueError("Legacy job table has duplicate buurt codes.")
    bad = out[list(JOB_GROUPS)].lt(0).any(axis=1)
    if bad.any():
        raise ValueError(f"{int(bad.sum())} buurt(en) with negative jobs.")
    return out.set_index("buurtcode")


def load_legacy_jobs(path: str | Path, year: str = "2018",
                     sheet: str = "buurten-arbeidsplaatsen") -> pd.DataFrame:
    """Read the legacy Excel file (needs `openpyxl`)."""
    raw = pd.read_excel(path, sheet_name=sheet, header=2)
    return parse_legacy_jobs(raw, year)


# ── Sector jobs -> income pools (wage-ranked partition) ──────────────

def sector_income_weights(
    sector_wage: pd.Series,
    sector_jobs: pd.Series,
    income_classes: Sequence[str] = INCOME_CLASSES,
) -> pd.DataFrame:
    """Share of each sector's jobs that falls in each income class.

    Sectors are ranked by mean wage (lowest first) and laid along the
    income-rank axis [0, 1] in proportion to their national jobs; income
    decile Dk covers [(k-1)/10, k/10]. Sector s's jobs are spread over
    the deciles it overlaps, in proportion to the overlap:

        W[k, s] = |decile k ∩ sector s| / |sector s|      (columns sum to 1)

    so the decile pools PARTITION the jobs: pool_k = sum_s W[k, s] * J_s
    and sum_k pool_k = J. This is the D_{j,s} of the paper (jobs matched
    to an income level). 'onbekend' has no rank and sees all jobs
    (all-ones row). Within-sector wage dispersion is ignored: every job
    of a sector sits at that sector's mean-wage rank.

    """
    wage = sector_wage.astype(float)
    jobs = sector_jobs.reindex(wage.index).astype(float)
    if jobs.isna().any() or (jobs < 0).any() or jobs.sum() <= 0:
        raise ValueError("Sector jobs must be non-negative, complete and "
                         "sum to a positive number.")
    if wage.isna().any():
        raise ValueError("Sector wages are incomplete.")
    order = wage.sort_values(kind="stable").index
    edges = np.concatenate([[0.0], np.cumsum(jobs[order]) / jobs.sum()])

    ranked = [c for c in income_classes if c != "onbekend"]
    n = len(ranked)
    W = pd.DataFrame(0.0, index=list(income_classes), columns=wage.index)
    for k, cls in enumerate(ranked):
        lo, hi = k / n, (k + 1) / n
        for i, s in enumerate(order):
            length = edges[i + 1] - edges[i]
            if length <= 0:
                continue
            overlap = max(0.0, min(hi, edges[i + 1]) - max(lo, edges[i]))
            W.loc[cls, s] = overlap / length
    if "onbekend" in income_classes:
        W.loc["onbekend"] = 1.0
    return W


def sector_pools(
    sector_jobs: pd.DataFrame,
    zone_codes: Sequence[str],
    weights: pd.DataFrame,
) -> dict[str, np.ndarray]:
    """Opportunity vector per income class from imputed sector jobs
    (buurt x sector), aligned to the engine zones: pool = J @ W.T."""
    codes = [str(c).strip() for c in zone_codes]
    if len(set(codes)) != len(codes):
        raise ValueError("zone_codes contains duplicates.")
    missing = [s for s in weights.columns if s not in sector_jobs.columns]
    if missing:
        raise KeyError(f"Sector jobs lack column(s) {missing}.")
    aligned = sector_jobs[list(weights.columns)].reindex(codes)
    n_missing = int(aligned.isna().all(axis=1).sum())
    if n_missing:
        logger.warning("%d of %d engine zones have no sector jobs; "
                       "treated as having none.", n_missing, len(codes))
    J = np.nan_to_num(aligned.to_numpy(float), nan=0.0)
    return {cls: (J @ weights.loc[cls].to_numpy(float)).astype(np.float32)
            for cls in weights.index}
