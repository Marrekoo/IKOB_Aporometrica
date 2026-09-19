"""
Impute LISA sector jobs onto buurten (GSPREE-style).

Known: jobs per municipality x 15 LISA sectors (LISA), and a job total
per buurt (legacy NRM-derived table). Unknown: how each municipality's
sector jobs are spread over its buurten. The imputation

  1. fits a log-linear model of the municipal sector composition on
     jobs-weighted buurt covariates (education mix of jobs, urbanisation,
     house value) - 2016 LISA composition against the 2016 buurt
     education shares, both at municipal level;
  2. predicts a seed composition for every buurt from ITS covariates;
  3. rakes (IPF) each municipality's buurt x sector table to the buurt
     job totals (rows) and the municipality's LISA sector totals
     (columns), for the requested year.

The municipal marginals are reproduced exactly. What is NOT identified
from municipal data is the association within a municipality: the model
transfers the between-municipality relation between covariates and
sector mix to buurten (an ecological inference), and buurt-level truth
is not available to check it. Read the result as a model, not as data.

Covariates enter in natural units (shares, class 1-5, ln house value),
not z-scores, so a coefficient means the same at municipal and buurt
level.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from ikob2.segments.ipf import ipf_batch
from ikob2.segments.lisa import SECTORS

logger = logging.getLogger(__name__)

COVARIATES = ("edu_praktisch", "edu_hoger", "stedelijkheid", "ln_woz")


@dataclass(frozen=True)
class SectorModel:
    alpha: np.ndarray          # (n_sectors,)
    beta: np.ndarray           # (n_sectors, n_covariates)
    covariates: tuple[str, ...]
    sectors: tuple[str, ...]
    # Support of the municipal covariates the model was fitted on.
    # Buurt values (which vary far more than municipal means) are
    # clipped to it: the log-linear model would otherwise extrapolate
    # exponentially outside the range it has seen.
    lower: np.ndarray | None = None
    upper: np.ndarray | None = None

    def shares(self, x: np.ndarray) -> np.ndarray:
        """Predicted sector composition (rows sum to 1) for covariate
        rows x (n, n_covariates), clipped to the fitted support."""
        x = np.asarray(x, dtype=float)
        if self.lower is not None:
            x = np.clip(x, self.lower, self.upper)
        eta = self.alpha[None, :] + x @ self.beta.T
        eta -= eta.max(axis=1, keepdims=True)
        e = np.exp(eta)
        return e / e.sum(axis=1, keepdims=True)


@dataclass(frozen=True)
class SectorJobs:
    jobs: pd.DataFrame           # buurtcode x sector, imputed jobs
    model: SectorModel
    report: dict


# ── Covariates ───────────────────────────────────────────────────────

def parse_education_shares(edu: pd.DataFrame) -> pd.DataFrame:
    """Buurt education mix of jobs (2016 LISA, Amsterdam file), indexed
    by BU_CODE: shares Praktisch / Middelbaar / Hoger and the job count
    behind them."""
    need = ["BU_CODE", "Praktisch", "Middelbaar", "Hoger",
            "Aantal_Middelbaar", "Aantal_Onbekend", "Aantal_Praktisch",
            "Aantal_Theoretisch"]
    missing = [c for c in need if c not in edu.columns]
    if missing:
        raise KeyError(f"Education table lacks column(s) {missing}.")
    df = edu[need].copy()
    df["BU_CODE"] = df["BU_CODE"].astype(str).str.strip()
    for c in need[1:]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["edu_jobs"] = df[need[4:]].sum(axis=1)
    df = df.rename(columns={"Praktisch": "edu_praktisch",
                            "Middelbaar": "edu_middelbaar",
                            "Hoger": "edu_hoger"})
    return df.drop_duplicates("BU_CODE").set_index("BU_CODE")[
        ["edu_praktisch", "edu_middelbaar", "edu_hoger", "edu_jobs"]]


def buurt_covariates(buurten: pd.DataFrame,
                     education: pd.DataFrame) -> pd.DataFrame:
    """Covariate matrix per buurt (index = buurtcode).

    buurten needs buurtcode, stedelijkheid (class 1-5) and gem_woz
    (mean house value, thousands EUR); missing values stay NaN.
    """
    df = buurten.set_index("buurtcode")[["stedelijkheid", "gem_woz"]].copy()
    df["ln_woz"] = np.log(df["gem_woz"].where(df["gem_woz"] > 0))
    edu = education.reindex(df.index)
    df["edu_praktisch"] = edu["edu_praktisch"]
    df["edu_hoger"] = edu["edu_hoger"]
    return df[list(COVARIATES)]


def municipal_covariates(cov: pd.DataFrame, weights: pd.Series,
                         gemeente: pd.Series) -> pd.DataFrame:
    """Jobs-weighted mean covariates per municipality. Each covariate is
    averaged over the buurten where it is known (weights renormalised)."""
    g = gemeente.reindex(cov.index)
    w = weights.reindex(cov.index).fillna(0.0)
    out = {}
    for c in cov.columns:
        ok = cov[c].notna() & g.notna() & (w > 0)
        num = (cov[c].where(ok, 0.0) * w.where(ok, 0.0)).groupby(g).sum()
        den = w.where(ok, 0.0).groupby(g).sum()
        out[c] = num / den.where(den > 0)
    return pd.DataFrame(out)


# ── Model ────────────────────────────────────────────────────────────

def fit_sector_model(municipal_jobs: pd.DataFrame,
                     municipal_x: pd.DataFrame) -> SectorModel:
    """Poisson log-linear model per sector with a log-total offset:
    E[jobs_gs] = T_g * exp(alpha_s + x_g . beta_s), fitted on the
    municipalities with complete covariates."""
    import statsmodels.api as sm

    idx = municipal_jobs.index.intersection(municipal_x.index)
    x = municipal_x.loc[idx].dropna()
    jobs = municipal_jobs.loc[x.index]
    if len(x) < 20:
        raise ValueError(f"Only {len(x)} municipalities with complete "
                         f"covariates; need at least 20 to fit the model.")
    total = jobs.sum(axis=1).to_numpy(float)
    keep = total > 0
    x, jobs, total = x[keep], jobs[keep], total[keep]
    X = sm.add_constant(x.to_numpy(float), has_constant="add")
    offset = np.log(total)

    alpha = np.zeros(len(SECTORS))
    beta = np.zeros((len(SECTORS), x.shape[1]))
    for k, s in enumerate(SECTORS):
        fit = sm.GLM(jobs[s].to_numpy(float), X, family=sm.families.Poisson(),
                     offset=offset).fit(maxiter=100)
        if not fit.converged:
            logger.warning("Sector model for %s did not converge.", s)
        alpha[k], beta[k] = fit.params[0], fit.params[1:]
    return SectorModel(alpha, beta, tuple(x.columns), tuple(SECTORS),
                       lower=x.min().to_numpy(float),
                       upper=x.max().to_numpy(float))


# ── Imputation ───────────────────────────────────────────────────────

def impute_sector_jobs(
    buurt_jobs: pd.Series,
    buurt_gemeente: pd.Series,
    lisa: pd.DataFrame,
    model: SectorModel,
    cov: pd.DataFrame,
    *,
    tol: float = 1e-8,
    max_iter: int = 500,
) -> SectorJobs:
    """Rake each municipality's buurt x sector table to its marginals.

    buurt_jobs : job total per buurt (indexed by buurtcode); only
        proportions within a municipality matter, the level comes from
        LISA.
    buurt_gemeente : LISA municipality name per buurt (NaN = not
        covered; those buurten get no jobs).
    lisa : municipality x sector jobs for the target year.
    cov : covariates per buurt (buurt_covariates output); missing
        values are replaced by the municipality's mean, then the
        national mean.
    """
    codes = buurt_gemeente.dropna().index
    codes = codes.intersection(buurt_jobs.index.union(cov.index))
    gem = buurt_gemeente.reindex(codes)
    rows = buurt_jobs.reindex(codes).fillna(0.0).clip(lower=0.0)

    x = cov.reindex(codes)[list(model.covariates)]
    fill = x.groupby(gem).transform("mean")
    n_fill = int(x.isna().any(axis=1).sum())
    x = x.fillna(fill).fillna(x.mean())
    if n_fill:
        logger.info("%d buurten had incomplete covariates; filled with "
                    "the municipal/national mean.", n_fill)
    xv = x.to_numpy(float)
    if model.lower is not None:
        n_clip = int(((xv < model.lower) | (xv > model.upper)).any(axis=1).sum())
        if n_clip:
            logger.info("%d buurten had covariates outside the fitted "
                        "municipal range; clipped to it.", n_clip)
    seed_share = model.shares(xv)
    seed_share = pd.DataFrame(seed_share, index=codes,
                              columns=list(model.sectors))

    out = pd.DataFrame(0.0, index=codes, columns=list(model.sectors))
    stats = {"municipalities": 0, "not_converged": 0,
             "no_buurt_jobs_equal_split": 0, "lisa_without_buurten": [],
             "max_marginal_error": 0.0}

    for name, sector_totals in lisa.iterrows():
        col_t = sector_totals.to_numpy(float)
        if col_t.sum() <= 0:
            continue
        members = gem.index[gem == name]
        if len(members) == 0:
            stats["lisa_without_buurten"].append(name)
            continue
        r = rows.loc[members].to_numpy(float)
        if r.sum() <= 0:
            r = np.ones(len(members))
            stats["no_buurt_jobs_equal_split"] += 1
        row_t = r / r.sum() * col_t.sum()
        seed = seed_share.loc[members].to_numpy(float)
        res = ipf_batch(seed[None], row_t[None], col_t[None],
                        tol=tol, max_iter=max_iter)
        out.loc[members] = res.table[0]
        stats["municipalities"] += 1
        stats["not_converged"] += int(not res.converged[0])
        err = max(np.abs(res.table[0].sum(axis=1) - row_t).max(),
                  np.abs(res.table[0].sum(axis=0) - col_t).max())
        stats["max_marginal_error"] = max(stats["max_marginal_error"],
                                          float(err))

    stats["total_jobs_imputed"] = float(out.to_numpy().sum())
    stats["total_jobs_lisa"] = float(lisa.to_numpy().sum())
    stats["buurten_with_jobs"] = int((out.sum(axis=1) > 0).sum())
    if stats["lisa_without_buurten"]:
        logger.warning("LISA municipalities without buurten (their jobs "
                       "are not imputed): %s", stats["lisa_without_buurten"])
    logger.info("Sector-job imputation: %s", {
        k: v for k, v in stats.items() if k != "lisa_without_buurten"})
    return SectorJobs(out, model, stats)
