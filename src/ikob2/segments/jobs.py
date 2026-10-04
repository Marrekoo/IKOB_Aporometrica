"""
Job supply per income-matched pool.

The paper measures opportunities D_{j,s}: jobs matched to a segment's
income. Jobs come per buurt and LISA sector (see jobs_impute); the share
of each sector's jobs in each income class (segments.occupations) turns
them into income-decile pools (`sector_pools`). This module also reads
the IKOB job table, whose buurt totals are the row marginal of the
sector imputation. See docs/data_lineage.md.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd


logger = logging.getLogger(__name__)

JOB_GROUPS = ("laag", "middellaag", "middelhoog", "hoog")   # low -> high


# ── IKOB job table ───────────────────────────────────────────────────

def parse_ikob_jobs(raw: pd.DataFrame, year: str = "2018") -> pd.DataFrame:
    """Jobs per buurt and income group from the IKOB job table (pivot sheet).

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
        raise ValueError("IKOB job table has duplicate buurt codes.")
    bad = out[list(JOB_GROUPS)].lt(0).any(axis=1)
    if bad.any():
        raise ValueError(f"{int(bad.sum())} buurt(en) with negative jobs.")
    return out.set_index("buurtcode")


def load_ikob_jobs(path: str | Path, year: str = "2018",
                     sheet: str = "buurten-arbeidsplaatsen") -> pd.DataFrame:
    """Read the IKOB job table from Excel (needs `openpyxl`)."""
    raw = pd.read_excel(path, sheet_name=sheet, header=2)
    return parse_ikob_jobs(raw, year)


# ── Sector jobs -> income pools (wage-ranked partition) ──────────────

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
