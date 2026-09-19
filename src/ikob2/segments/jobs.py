"""
Job supply per income-matched pool (interim, legacy-derived).

The paper measures opportunities D_{j,s}: jobs matched to a segment's
income. The proper source is LISA (jobs by industry, SBI), which is not
available. Until it is, the legacy IKOB job table is used: jobs per
buurt in FOUR income groups (laag, middellaag, middelhoog, hoog),
derived upstream from NRM zone totals and a 2016 LISA education-level
distribution (see docs/data_lineage.md).

The segments have eleven income classes, so the four job groups are
matched to them by QUANTILE MATCHING: the job groups partition the
income-rank axis [0, 1] in proportion to their national job shares
(laag lowest), and income decile D_k covers the rank interval
[(k-1)/10, k/10]. The share of decile k's interval falling in job group
g is the weight W[k, g], and the pool of decile k is the mixture

    pool_k = sum_g W[k, g] * jobs_g        (rows of W sum to 1)

so a decile sees the jobs of the group(s) at its own income rank, not a
share of them. 'onbekend' has no rank and sees all jobs. This is an
assumption, replaced when LISA-by-SBI data becomes available.
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


# ── Quantile matching ────────────────────────────────────────────────

def quantile_weights(
    group_totals: Sequence[float],
    income_classes: Sequence[str] = INCOME_CLASSES,
) -> pd.DataFrame:
    """Weights W (income class x job group); rows sum to 1 except
    'onbekend', which is all ones (it sees every job group in full).

    group_totals : national jobs per group, ordered low -> high. Only
        their proportions matter.
    """
    totals = np.asarray(group_totals, dtype=float)
    if totals.shape != (len(JOB_GROUPS),):
        raise ValueError(f"Need {len(JOB_GROUPS)} group totals, got "
                         f"{totals.shape}.")
    if not np.all(np.isfinite(totals)) or np.any(totals < 0) \
            or totals.sum() <= 0:
        raise ValueError("Group totals must be finite, non-negative and "
                         "sum to a positive number.")
    edges = np.concatenate([[0.0], np.cumsum(totals) / totals.sum()])

    ranked = [c for c in income_classes if c != "onbekend"]
    n = len(ranked)
    rows = {}
    for k, cls in enumerate(ranked):
        lo, hi = k / n, (k + 1) / n
        overlap = np.clip(np.minimum(hi, edges[1:])
                          - np.maximum(lo, edges[:-1]), 0.0, None)
        rows[cls] = overlap / (hi - lo)
    if "onbekend" in income_classes:
        rows["onbekend"] = np.ones(len(JOB_GROUPS))
    return pd.DataFrame(rows, index=list(JOB_GROUPS)).T.loc[
        list(income_classes)]


def job_pools(
    jobs: pd.DataFrame,
    zone_codes: Sequence[str],
    weights: pd.DataFrame | None = None,
) -> dict[str, np.ndarray]:
    """Opportunity vector per income class, aligned to the engine zones.

    jobs : frame indexed by buurtcode with the four job-group columns
        (parse_legacy_jobs output). Zones without a row have no jobs
        (logged). weights : quantile_weights(); defaults to weights
        from the national totals in `jobs` itself.
    Returns {income_class: (n_zones,) float32} for use as the runner's
    `opportunities`, with SegmentedRunner pools named by income class
    (build_segments(pool_by="income_class")).
    """
    codes = [str(c).strip() for c in zone_codes]
    if len(set(codes)) != len(codes):
        raise ValueError("zone_codes contains duplicates.")
    missing = [g for g in JOB_GROUPS if g not in jobs.columns]
    if missing:
        raise KeyError(f"Job table lacks group column(s) {missing}.")
    if weights is None:
        weights = quantile_weights(jobs[list(JOB_GROUPS)].sum().to_numpy())

    aligned = jobs[list(JOB_GROUPS)].reindex(codes)
    n_missing = int(aligned.isna().all(axis=1).sum())
    if n_missing:
        logger.warning("%d of %d engine zones have no job row; treated "
                       "as having no jobs.", n_missing, len(codes))
    J = np.nan_to_num(aligned.to_numpy(float), nan=0.0)         # (n, 4)
    W = weights[list(JOB_GROUPS)].to_numpy(float)               # (C, 4)
    return {cls: (J @ W[k]).astype(np.float32)
            for k, cls in enumerate(weights.index)}
