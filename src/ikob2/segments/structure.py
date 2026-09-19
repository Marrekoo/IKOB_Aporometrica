"""Structure model: the household-type x income association.

Municipal counts (86161NED, per 1000 households) are modelled as
    mu = exp(alpha_cell + beta_sted_cell * z_sted + beta_woz_cell * z_woz)
per (household type, income class) cell, fitted as a Poisson-family GLM
(quasi-Poisson in R; identical point estimates). With too few
municipalities/covariates it degrades to the saturated cell model,
whose maximum-likelihood fit is simply the cell mean (SPREE-like).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from ikob2.segments.config import SegmentConfig

logger = logging.getLogger(__name__)

_MIN_ROWS = 10
_MIN_GEMEENTEN = 10


@dataclass(frozen=True)
class StructureModel:
    cells: tuple[tuple[str, str], ...]     # (household_type, income_class)
    alpha: np.ndarray                      # (n_cells,)
    beta_sted: np.ndarray | None           # (n_cells,) or None
    beta_woz: np.ndarray | None
    with_covariates: bool

    def predict(self, sted_std, woz_std, cfg: SegmentConfig) -> np.ndarray:
        """Seed tables (B, n_types, n_classes), floored at 1e-8."""
        sted = np.asarray(sted_std, dtype=float)
        woz = np.asarray(woz_std, dtype=float)
        eta = np.broadcast_to(self.alpha, (len(sted), len(self.alpha))).copy()
        if self.with_covariates:
            eta = (eta + np.outer(sted, self.beta_sted)
                   + np.outer(woz, self.beta_woz))
        with np.errstate(over="ignore", invalid="ignore"):
            mu = np.exp(eta)
        mu = np.where(np.isfinite(mu) & (mu > 0), mu, 1e-8)

        t_idx = {t: i for i, t in enumerate(cfg.household_types)}
        c_idx = {c: i for i, c in enumerate(cfg.income_classes)}
        out = np.full((len(sted), len(t_idx), len(c_idx)), 1e-8)
        for k, (t, c) in enumerate(self.cells):
            out[:, t_idx[t], c_idx[c]] = mu[:, k]
        return out


def std_vec(x: pd.Series) -> pd.Series:
    """z-score with sample sd; constant/undefined -> zeros (as R std_vec)."""
    x = pd.to_numeric(x, errors="coerce").astype(float)
    m, s = x.mean(), x.std(ddof=1)
    if not np.isfinite(s) or s == 0:
        return pd.Series(np.zeros(len(x)), index=x.index)
    return (x - m) / s


def gemeente_covariates(kwb: pd.DataFrame) -> pd.DataFrame:
    g = kwb.groupby("gemeentecode", as_index=False).agg(
        stedelijkheid=("stedelijkheid", "mean"), gem_woz=("gem_woz", "mean"))
    g["stedelijkheid_std"] = std_vec(g["stedelijkheid"]).to_numpy()
    g["gem_woz_std"] = std_vec(g["gem_woz"]).to_numpy()
    return g


def fit_structure_model(reference_seed: pd.DataFrame,
                        covariates: pd.DataFrame,
                        cfg: SegmentConfig) -> StructureModel:
    df = reference_seed.merge(covariates[["gemeentecode", "stedelijkheid_std",
                                          "gem_woz_std"]],
                              how="left", on="gemeentecode")
    cells = tuple((t, c) for t in cfg.household_types
                  for c in cfg.income_classes)
    cell_id = {cell: k for k, cell in enumerate(cells)}

    have_cov = (df["stedelijkheid_std"].notna()
                & df["gem_woz_std"].notna()).sum() >= _MIN_ROWS
    if have_cov:
        dm = df.dropna(subset=["n_hh", "stedelijkheid_std", "gem_woz_std"])
        if dm["gemeentecode"].nunique() >= _MIN_GEMEENTEN:
            logger.info("Fitting structure model with covariate "
                        "interactions (%d rows).", len(dm))
            return _fit_glm(dm, cells, cell_id)

    logger.info("Fitting structure model without covariates (SPREE-like).")
    dm = df.dropna(subset=["n_hh"]).copy()
    dm["y"] = np.maximum(dm["n_hh"] / 1000, 1e-6)
    means = dm.groupby(["household_type", "income_class"])["y"].mean()
    alpha = np.array([np.log(means[cell]) for cell in cells])
    return StructureModel(cells, alpha, None, None, False)


def _fit_glm(dm: pd.DataFrame, cells, cell_id) -> StructureModel:
    import statsmodels.api as sm

    k = np.array([cell_id[(t, c)] for t, c in
                  zip(dm["household_type"], dm["income_class"])])
    n, m = len(dm), len(cells)
    sted = dm["stedelijkheid_std"].to_numpy(float)
    woz = dm["gem_woz_std"].to_numpy(float)
    X = np.zeros((n, 3 * m))
    X[np.arange(n), k] = 1.0
    X[np.arange(n), m + k] = sted
    X[np.arange(n), 2 * m + k] = woz
    y = np.maximum(dm["n_hh"].to_numpy(float) / 1000, 1e-6)

    fit = sm.GLM(y, X, family=sm.families.Poisson()).fit(maxiter=200)
    if not fit.converged:
        logger.warning("Structure-model GLM did not converge.")
    b = np.asarray(fit.params)
    return StructureModel(cells, b[:m], b[m:2 * m], b[2 * m:], True)
