"""
Jobs matched to income deciles through sector x occupation cells, and the
home-working share of jobs from their occupations.

Each LISA sector is split into ISCO-08 major groups by the national
employment of the matching NACE sections (Eurostat LFS, lfsa_eisn2). A
cell's hourly wage is lognormal: its mean is the cell's mean hourly
earnings (Eurostat Structure of Earnings Survey, earn_ses22_47) and its
log standard deviation `sigma` is one common value, the spread of wages
within the occupations (CBS 85517NED quartiles of the BRC 2014
occupation groups, carried to the ISCO major groups by the CBS
ISCO 2008 - BRC 2014 correspondence). The job-weighted mixture of the
cells is the national wage distribution of jobs; its deciles D1-D10
partition every cell, so each sector reaches several deciles:

    W[k, c] = Phi((ln b_k - mu_c)/sigma) - Phi((ln b_{k-1} - mu_c)/sigma)
    mu_c    = ln(mean wage of c) - sigma^2/2
    b_k     the k/10 quantile of sum_c p_c LN(mu_c, sigma),  p_c the job share

A job admits home working with the probability of its occupation: the
technical teleworkability of Sostero et al. (2020, the 'physical
interaction' indicator: the share of an ISCO 3-digit group that can
potentially work remotely), averaged to ISCO 2-digit groups and weighted
to the major groups by Dutch employment (lfsa_egai2d). The decile pools
by job type are

    pool[w, k] = J @ W_w[k].T,  W_w[k, s] = sum_{c in s} f_c W[k, c] t_w(c)

with f_c the cell's share of its sector's jobs, t_wfh_possible(c) the
teleworkability of its occupation and t_no_wfh = 1 - t_wfh_possible.
Over both job types and all deciles the weights of a sector add up to 1:
the pools partition the jobs. 'onbekend' (unknown income) sees all jobs
of each type.

The tables live in data/occupations (README there: sources, licences,
retrieval); `fetch_tables` downloads them again.
"""

from __future__ import annotations

import itertools
import json
import logging
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from scipy import optimize, stats

from ikob2.segments.config import INCOME_CLASSES

logger = logging.getLogger(__name__)

ISCO_MAJORS = tuple(f"OC{i}" for i in range(1, 10))   # armed forces (OC0) excluded
EUROSTAT_ROOT = "https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data"
CROSSWALK_URL = ("https://www.cbs.nl/-/media/imported/onze-diensten/methoden/"
                 "classificaties/documents/2015/09/schakelschema-isco2008-"
                 "brc2014.xls")
TELEWORK_URL = ("https://zenodo.org/api/records/7716456/files/"
                "Telework%20ISCO%20indices.csv/content")

FILES = {
    "employment": "lfs_nace_isco_nl.csv",
    "wages": "ses_hourly_nace_isco_nl.csv",
    "employment_isco2": "lfs_isco2_nl.csv",
    "quartiles": "brc_wage_quartiles.csv",
    "crosswalk": "isco_brc_crosswalk.csv",
    "telework": "teleworkability_isco3.csv",
}


# ── loading ──────────────────────────────────────────────────────────

@dataclass(frozen=True)
class OccupationTables:
    employment: pd.DataFrame        # nace_r2, isco08, employed_thousands
    wages: pd.DataFrame             # nace_r2, isco08, mean_hourly_eur
    employment_isco2: pd.DataFrame  # isco08 (OCnn), employed_thousands
    quartiles: pd.DataFrame         # brc, employees_thousands, p25, p50, p75
    crosswalk: pd.DataFrame         # isco08_unit, brc_group
    telework: pd.DataFrame          # isco08 (3 digits), physical_interaction


def load_tables(folder: str | Path) -> OccupationTables:
    folder = Path(folder)
    missing = [f for f in FILES.values() if not (folder / f).exists()]
    if missing:
        raise FileNotFoundError(
            f"Occupation tables missing in {folder}: {missing}. Run "
            f"`python -m ikob2.cli.layout create` (copies data/occupations).")
    read = {k: pd.read_csv(folder / f, dtype={"isco08": str, "brc": str,
                                              "brc_group": str,
                                              "isco08_unit": str})
            for k, f in FILES.items()}
    return OccupationTables(**read)


# ── cells ────────────────────────────────────────────────────────────

def sector_cells(tables: OccupationTables,
                 sector_to_nace: Mapping[str, Sequence[str]]) -> pd.DataFrame:
    """sector, isco08, employment (thousands), mean_wage (EUR/hour): one row
    per LISA sector and ISCO major group with employment. The wage is the
    employment-weighted mean over the sector's NACE sections that publish
    one; a cell without any takes the employment-weighted mean wage of its
    occupation over all sections."""
    emp = tables.employment[tables.employment.isco08.isin(ISCO_MAJORS)]
    emp = emp.pivot(index="nace_r2", columns="isco08", values="employed_thousands")
    wag = tables.wages[tables.wages.isco08.isin(ISCO_MAJORS)]
    wag = wag.pivot(index="nace_r2", columns="isco08", values="mean_hourly_eur")
    both = emp.reindex_like(wag).where(wag.notna())
    occ_mean = (both * wag).sum() / both.sum()
    rows = []
    for sector, sections in sector_to_nace.items():
        unknown = [x for x in sections if x not in emp.index]
        if unknown:
            raise KeyError(f"No employment for NACE section(s) {unknown} "
                           f"of sector {sector}.")
        for occ in [o for o in ISCO_MAJORS if o in emp.columns]:
            e = emp.loc[list(sections), occ].fillna(0.0)
            if e.sum() <= 0:
                continue
            w = (wag.reindex(index=list(sections))[occ] if occ in wag.columns
                 else pd.Series(np.nan, index=e.index))
            ok = w.notna() & (e > 0)
            wage = (float((e[ok] * w[ok]).sum() / e[ok].sum()) if ok.any()
                    else float(occ_mean.get(occ, np.nan)))
            rows.append((sector, occ, float(e.sum()), wage))
    cells = pd.DataFrame(rows, columns=["sector", "isco08", "employment",
                                        "mean_wage"])
    if cells.mean_wage.isna().any() or (cells.mean_wage <= 0).any():
        raise ValueError("A cell has no positive wage.")
    return cells


def within_cell_sd(quartiles: pd.DataFrame, crosswalk: pd.DataFrame) -> float:
    """Common log standard deviation of hourly wages within a cell: per ISCO
    major group, the spread of the lognormal mixture of its BRC occupation
    groups (each lognormal from its median and interquartile range; a
    group's employees are split evenly over the ISCO unit groups it is
    made of), averaged over the major groups by employees.

    A BRC group without published quartiles takes those of the nearest
    published level above it (3 digits, then 2)."""
    q = quartiles.set_index("brc")
    z75 = stats.norm.ppf(0.75)

    def quart(code):
        for c in (code, code[:3], code[:2]):
            if c in q.index and q.loc[c, ["p25", "p50", "p75"]].notna().all():
                return q.loc[c, ["p25", "p50", "p75"]].astype(float).to_numpy()
        raise KeyError(f"No wage quartiles for BRC group {code} or above.")

    rows = []
    for code, grp in crosswalk.groupby("brc_group"):
        n = q["employees_thousands"].get(code, np.nan)
        if not np.isfinite(n) or n <= 0:
            continue
        p25, p50, p75 = quart(code)
        mu, sd = np.log(p50), np.log(p75 / p25) / (2.0 * z75)
        majors = "OC" + grp.isco08_unit.str[0]
        for major, k in majors.value_counts().items():
            rows.append((major, n * k / len(grp), mu, sd))
    g = pd.DataFrame(rows, columns=["major", "n", "mu", "sd"])
    g = g[g.major.isin(ISCO_MAJORS)]
    per_major, weight = [], []
    for _, d in g.groupby("major"):
        m = np.average(d.mu, weights=d.n)
        per_major.append(np.sqrt(np.average(d.sd ** 2 + (d.mu - m) ** 2,
                                            weights=d.n)))
        weight.append(d.n.sum())
    return float(np.average(per_major, weights=weight))


def teleworkability_by_major(telework: pd.DataFrame,
                             employment_isco2: pd.DataFrame) -> pd.Series:
    """Share of the jobs of each ISCO major group that admit home working:
    the 3-digit teleworkability averaged to 2-digit groups, then weighted
    to major groups by employment."""
    t = telework.assign(d2="OC" + telework.isco08.str[:2])
    t2 = t.groupby("d2")["physical_interaction"].mean()
    e2 = employment_isco2.set_index("isco08")["employed_thousands"].reindex(t2.index)
    ok = e2.notna() & (e2 > 0)
    out = {}
    for major in ISCO_MAJORS:
        sel = ok & (t2.index.str[2] == major[2])
        if not sel.any():
            raise KeyError(f"No teleworkability for {major}.")
        out[major] = float(np.average(t2[sel], weights=e2[sel]))
    return pd.Series(out, name="teleworkability")


# ── decile weights ───────────────────────────────────────────────────

@dataclass(frozen=True)
class JobMatching:
    """Weights income class x sector per job type (`by_type`), and what
    produced them (for run.json)."""
    by_type: dict[str, pd.DataFrame]
    meta: dict


def occupation_job_weights(cells: pd.DataFrame, sigma: float,
                           teleworkability: pd.Series,
                           sector_jobs_total: pd.Series,
                           wfh_types: Sequence[str],
                           income_classes: Sequence[str] = INCOME_CLASSES
                           ) -> JobMatching:
    """Decile weights per job type from the sector x occupation cells (module
    docstring). `sector_jobs_total` (national jobs per LISA sector) scales
    each sector's occupation shares; `wfh_types` is (no home working, home
    working possible)."""
    if not sigma > 0:
        raise ValueError("The within-cell spread must be positive.")
    no_wfh, wfh = wfh_types
    sectors = list(sector_jobs_total.index)
    c = cells[cells.sector.isin(sectors)].copy()
    lacking = sorted(set(sectors) - set(c.sector))
    if lacking:
        raise KeyError(f"No occupation cells for sector(s) {lacking}.")
    c["share"] = c.employment / c.groupby("sector").employment.transform("sum")
    c["jobs"] = c.share * sector_jobs_total.reindex(c.sector).to_numpy(float)
    if c.jobs.sum() <= 0:
        raise ValueError("Sector jobs must sum to a positive number.")
    mu = np.log(c.mean_wage.to_numpy(float)) - sigma ** 2 / 2.0
    p = c.jobs.to_numpy(float) / c.jobs.sum()
    ranked = [k for k in income_classes if k != "onbekend"]
    n = len(ranked)

    def cdf(x):
        return float(np.sum(p * stats.norm.cdf((np.log(x) - mu) / sigma)))

    lo = float(np.exp(mu.min() - 10 * sigma))
    hi = float(np.exp(mu.max() + 10 * sigma))
    edges = [optimize.brentq(lambda x, q=k / n: cdf(x) - q, lo, hi, xtol=1e-10)
             for k in range(1, n)]
    F = np.vstack([np.zeros(len(c))]
                  + [stats.norm.cdf((np.log(b) - mu) / sigma) for b in edges]
                  + [np.ones(len(c))])
    Wc = np.diff(F, axis=0)                          # (deciles, cells)
    tw = c.isco08.map(teleworkability).to_numpy(float)
    if np.isnan(tw).any():
        raise KeyError("Teleworkability lacks an ISCO major group.")
    by_type = {}
    for wtype, share in ((no_wfh, 1.0 - tw), (wfh, tw)):
        cell_w = pd.DataFrame(Wc * (c.share.to_numpy() * share), index=ranked,
                              columns=c.index)
        W = cell_w.T.groupby(c.sector.to_numpy()).sum().T.reindex(
            columns=sectors, fill_value=0.0)
        if "onbekend" in income_classes:
            W.loc["onbekend"] = W.sum()
        by_type[wtype] = W.reindex(list(income_classes))
    meta = {"matching": "occupation", "within_cell_sd": sigma,
            "decile_edges_eur_per_hour": [round(b, 4) for b in edges],
            "teleworkability": {k: round(float(v), 4)
                                for k, v in teleworkability.items()}}
    return JobMatching(by_type, meta)


# ── fetching (network) ───────────────────────────────────────────────

def jsonstat_frame(payload: dict) -> pd.DataFrame:
    """A Eurostat JSON-stat 2.0 response as a long frame: one column per
    dimension (codes) and `value` (NaN where not published)."""
    ids = payload["id"]
    cats = [sorted(payload["dimension"][k]["category"]["index"].items(),
                   key=lambda kv: kv[1]) for k in ids]
    values = payload["value"]
    rows = [{**{k: code for k, (code, _) in zip(ids, combo)},
             "value": values.get(str(i), np.nan) if isinstance(values, dict)
             else values[i]}
            for i, combo in enumerate(itertools.product(*cats))]
    return pd.DataFrame(rows)


def eurostat_url(dataset: str, **filters) -> str:
    """The Eurostat dissemination API query of a dataset with filters."""
    params = {**filters, "format": "JSON", "lang": "EN"}
    return f"{EUROSTAT_ROOT}/{dataset}?{urllib.parse.urlencode(params)}"


def _get(url: str) -> bytes:
    logger.info("GET %s", url)
    with urllib.request.urlopen(url, timeout=120) as resp:
        return resp.read()


def fetch_tables(out: str | Path, year: str = "2022") -> dict:
    """Download every table of `load_tables` from its public API or file
    into `out`, and write `sources.json`: per file the query URL, the time
    of retrieval and the SHA-256 of the written CSV. Returns that manifest.
    Reading the CBS correspondence (an Excel 97 file) needs `xlrd`."""
    import datetime as dt
    import hashlib
    import io

    from ikob2.segments import statline

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    frames, urls = {}, {}

    urls["employment"] = eurostat_url("lfsa_eisn2", geo="NL", sex="T",
                                      age="Y_GE15", unit="THS_PER", time=year)
    emp = jsonstat_frame(json.loads(_get(urls["employment"])))
    emp = emp[(emp.nace_r2.str.len() == 1) & emp.isco08.str.match(r"^OC\d$")]
    frames["employment"] = emp.rename(columns={"value": "employed_thousands"})[
        ["nace_r2", "isco08", "employed_thousands"]]

    urls["wages"] = eurostat_url(f"earn_ses{year[2:]}_47", geo="NL", sex="T",
                                 indic_se="ERN", sizeclas="TOTAL", unit="EUR")
    ses = jsonstat_frame(json.loads(_get(urls["wages"])))
    ses = ses[(ses.nace_r2.str.len() == 1) & ses.isco08.str.match(r"^OC\d$")]
    frames["wages"] = ses.rename(columns={"value": "mean_hourly_eur"})[
        ["nace_r2", "isco08", "mean_hourly_eur"]]

    urls["employment_isco2"] = eurostat_url("lfsa_egai2d", geo="NL", sex="T",
                                            age="Y_GE15", unit="THS_PER",
                                            time=year)
    e2 = jsonstat_frame(json.loads(_get(urls["employment_isco2"])))
    frames["employment_isco2"] = e2[e2.isco08.str.match(r"^OC\d\d$")].rename(
        columns={"value": "employed_thousands"})[["isco08", "employed_thousands"]]

    select = ["Beroep", "Perioden", "Werknemer_1", "k_25ePercentiel_2",
              "k_50ePercentielMediaan_3", "k_75ePercentiel_4"]
    flt = f"Perioden eq '{year}JJ00'"
    urls["quartiles"] = statline.odata_url("85517NED", select, flt)
    q = statline._odata_get("85517NED", select, flt)
    q["title"] = q.Beroep.map(statline._odata_get_dimension("85517NED", "Beroep"))
    q["brc"] = q.title.str.split(" ").str[0]
    frames["quartiles"] = q.rename(columns={
        "Werknemer_1": "employees_thousands", "k_25ePercentiel_2": "p25",
        "k_50ePercentielMediaan_3": "p50", "k_75ePercentiel_4": "p75"})[
        ["brc", "title", "employees_thousands", "p25", "p50", "p75"]]

    urls["crosswalk"] = CROSSWALK_URL
    xw = pd.read_excel(io.BytesIO(_get(CROSSWALK_URL)),
                       sheet_name="BRC 2014 variabelen", dtype=str,
                       engine="xlrd")
    frames["crosswalk"] = xw.rename(columns={
        "ISCO2008unitgroup": "isco08_unit",
        "ISCO2008unitgrouplabel": "isco08_unit_label",
        "BRC2014beroepsgroep": "brc_group",
        "BRC2014beroepsgroep_label": "brc_group_label",
        "BRC2014beroepsklasse": "brc_class"})[
        ["isco08_unit", "isco08_unit_label", "brc_group", "brc_group_label",
         "brc_class"]]

    urls["telework"] = TELEWORK_URL
    tw = pd.read_csv(io.BytesIO(_get(TELEWORK_URL)), dtype={"ISCO08": str})
    frames["telework"] = tw.rename(columns={
        "ISCO08": "isco08", "Occupation Title": "title",
        "Physical interaction": "physical_interaction",
        "Social interaction": "social_interaction"})

    when = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    manifest = {}
    for key, df in frames.items():
        path = out / FILES[key]
        df.to_csv(path, index=False)
        manifest[FILES[key]] = {
            "url": urls[key], "retrieved": when,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    (out / "sources.json").write_text(json.dumps(manifest, indent=1) + "\n")
    return manifest
