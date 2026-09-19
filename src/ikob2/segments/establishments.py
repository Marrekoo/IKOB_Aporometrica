"""
Establishments per buurt and SBI group (CBS Kerncijfers wijken en
buurten), as information for the sector-job imputation.

KWB counts establishments in eight SBI groups (A, B-F, G+I, H+J, K-L,
M-N, O-Q, R-U). Counts are rounded, and the group cells of some buurten
are suppressed (NaN) while the total is known. The counts agree with
LISA's establishment counts (KWB/LISA 0.97-1.09 for the groups that map
cleanly), so they are usable as buurt-level evidence of WHERE each
sector's establishments are; they say nothing about establishment size.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

GROUPS = ("A", "B-F", "G+I", "H+J", "K-L", "M-N", "O-Q", "R-U")


def read_establishments(path: str | Path) -> pd.DataFrame:
    """Snapshot written by `cli.segments fetch`, indexed by buurtcode
    (columns: total and the eight groups)."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Establishment snapshot not found: {path}. Run "
            f"`python -m ikob2.cli.segments fetch` (needs network).")
    df = pd.read_csv(path)
    df["buurtcode"] = df["buurtcode"].astype(str).str.strip()
    missing = [c for c in ("total", *GROUPS) if c not in df.columns]
    if missing:
        raise KeyError(f"{path}: missing column(s) {missing}. Tables from "
                       f"before 2017 lack the O-Q group.")
    return df.set_index("buurtcode")[["total", *GROUPS]]


def complete_group_counts(
    est: pd.DataFrame,
    gemeente: pd.Series,
    fallback_total: pd.Series | None = None,
) -> pd.DataFrame:
    """Group counts for every buurt in `gemeente`'s index.

    * Suppressed group cells (NaN with a known total) are filled by
      spreading the buurt's total over the groups in proportion to the
      group composition of the municipality's fully observed buurten
      (national composition if the municipality has none).
    * Buurten without a row, or without a total, get a total from
      `fallback_total` (a buurt job count, converted with the
      municipality's establishments per job) or, failing that, zero.
      Zero-total buurten keep all-zero counts.

    Returns a frame indexed like `gemeente` with the eight groups.
    """
    idx = gemeente.index
    e = est.reindex(idx)[["total", *GROUPS]].astype(float)
    counts = e[list(GROUPS)]

    observed = counts.notna().all(axis=1) & (e["total"] > 0)
    comp_all = counts[observed].sum()
    comp_all = comp_all / comp_all.sum() if comp_all.sum() > 0 else \
        pd.Series(1.0 / len(GROUPS), index=list(GROUPS))
    comp_gm = counts[observed].groupby(gemeente[observed]).sum()
    comp_gm = comp_gm.div(comp_gm.sum(axis=1).replace(0, np.nan), axis=0)

    # establishments per fallback job, per municipality (for buurten
    # with no establishment row at all)
    total = e["total"].copy()
    if fallback_total is not None:
        jobs = fallback_total.reindex(idx).astype(float)
        ok = total.notna() & (jobs > 0)
        est_per_job = (total[ok].groupby(gemeente[ok]).sum()
                       / jobs[ok].groupby(gemeente[ok]).sum())
        national = float(total[ok].sum() / jobs[ok].sum()) if ok.any() else np.nan
        ratio = gemeente.map(est_per_job).fillna(national)
        need = total.isna() & (jobs > 0)
        total[need] = (jobs[need] * ratio[need]).round()
    total = total.fillna(0.0)

    filled = counts.copy()
    partial = counts.isna().any(axis=1)
    for k in GROUPS:
        gm_share = gemeente.map(comp_gm[k]) if k in comp_gm else \
            pd.Series(np.nan, index=idx)
        share = gm_share.fillna(comp_all[k])
        # Only the suppressed cells are filled: the observed groups keep
        # their counts and the remainder of the total is split over the
        # missing ones.
        fill = (total * share).where(counts[k].isna())
        filled[k] = counts[k].where(counts[k].notna(), fill)
    if partial.any():
        logger.info("%d buurten had suppressed/missing establishment "
                    "groups; filled from the municipal composition.",
                    int(partial.sum()))
    return filled.fillna(0.0)
