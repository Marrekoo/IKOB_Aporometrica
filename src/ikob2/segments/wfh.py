"""
Home-working (WFH) capability of jobs, per LISA sector: a rough derivation.

The time margin depends on whether a JOB admits working from home (the
two Weibull fits of `time_margins`). No source gives that per sector, so
it is derived from two CBS tables:

  * the share of employed people who at least sometimes work from home,
    by education level (85718NED, 2024): a person-level incidence;
  * the education mix of employee jobs per SBI2008 section (82072NED,
    published for 2010 only).

    share_wfh(section) = sum over education e of  P(e | section) x
                         incidence(e)

averaged over the sections of each LISA sector, weighted by jobs
(81431NED). This is a ROUGH proxy: it takes education as the only
driver of home working (occupation and task content are ignored), uses
a 2010 education mix, and reads "at least sometimes works from home" as
"the job admits home working". The result is a share per sector, used
to split jobs into a WFH-capable and a not-WFH-capable part.
"""

from __future__ import annotations

import pandas as pd

from ikob2.segments.lisa import SECTOR_TO_SBI, SECTORS

EDUCATION_KEYS = {"low": "2018700", "middle": "2018740", "high": "2018790"}
SECTOR_EDUCATION_KEYS = {"low": "18700", "middle": "18740", "high": "18790"}
WFH_ANY = "A027929"       # 'meestal of soms thuiswerken'
WFH_ALL = "T001205"       # total


def wfh_incidence_by_education(raw: pd.DataFrame) -> pd.Series:
    """Share of employed people (any position) who mostly or sometimes
    work from home, by education level: index low / middle / high."""
    df = raw.copy()
    for c in ("Persoonskenmerken", "Thuiswerken"):
        df[c] = df[c].astype(str).str.strip()
    df["n"] = pd.to_numeric(df["WerkzameBeroepsbevolking_1"], errors="coerce")
    out = {}
    for level, key in EDUCATION_KEYS.items():
        sel = df[df["Persoonskenmerken"] == key].set_index("Thuiswerken")["n"]
        if WFH_ANY not in sel.index or WFH_ALL not in sel.index:
            raise KeyError(f"Home-working table lacks education {key} "
                           f"totals.")
        out[level] = float(sel[WFH_ANY] / sel[WFH_ALL])
    inc = pd.Series(out, name="wfh_incidence")
    if not ((inc > 0) & (inc < 1)).all():
        raise ValueError(f"Implausible incidences: {inc.to_dict()}")
    return inc


def sector_education_mix(raw: pd.DataFrame) -> pd.DataFrame:
    """Education mix (low / middle / high shares) of employee jobs per
    SBI2008 section key (82072NED)."""
    df = raw.copy()
    df["edu"] = df["Onderwijsniveau"].astype(str).str.strip()
    df["sbi"] = df["BedrijfstakkenSBI2008"].astype(str).str.strip()
    df["jobs"] = pd.to_numeric(df["BanenVanWerknemers_1"], errors="coerce")
    inv = {v: k for k, v in SECTOR_EDUCATION_KEYS.items()}
    df = df[df["edu"].isin(inv)]
    wide = df.pivot_table(index="sbi", columns="edu", values="jobs",
                          aggfunc="sum").rename(columns=inv)
    wide = wide[list(SECTOR_EDUCATION_KEYS)]
    return wide.div(wide.sum(axis=1), axis=0)


def sbi_wfh_share(incidence: pd.Series, mix: pd.DataFrame) -> pd.Series:
    """Education-weighted home-working incidence per SBI section key."""
    return (mix[list(incidence.index)] * incidence).sum(axis=1)


def lisa_sector_wfh_share(sbi_share: pd.Series,
                          sbi_jobs: pd.Series) -> pd.Series:
    """Home-working share per LISA sector: the job-weighted mean over
    its SBI sections (`lisa.SECTOR_TO_SBI`). sbi_jobs is jobs per section
    key, e.g. the 81431NED job counts."""
    out = {}
    for sector, keys in SECTOR_TO_SBI.items():
        missing = [k for k in keys if k not in sbi_share.index
                   or k not in sbi_jobs.index]
        if missing:
            raise KeyError(f"Missing SBI section(s) {missing} for {sector}.")
        w = sbi_jobs[list(keys)].astype(float)
        out[sector] = float((sbi_share[list(keys)] * w).sum() / w.sum())
    return pd.Series(out, name="wfh_share").reindex(list(SECTORS))


def split_jobs_by_wfh(sector_jobs: pd.DataFrame,
                      wfh_share: pd.Series) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(jobs not admitting WFH, jobs admitting WFH), both buurt x sector,
    adding up to `sector_jobs`."""
    share = wfh_share.reindex(sector_jobs.columns)
    if share.isna().any() or ((share < 0) | (share > 1)).any():
        raise ValueError("wfh_share must cover every sector, in [0, 1].")
    wfh = sector_jobs.mul(share, axis=1)
    return sector_jobs - wfh, wfh
