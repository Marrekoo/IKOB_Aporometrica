"""
Perturbation of the inputs within the rounding of their published sources.

The job and population inputs rest on rounded official statistics: LISA
publishes jobs per municipality and sector rounded to tens, and the CBS
neighbourhood statistics (KWB) round counts and percentages. A perturbed run
replaces each rounded figure by a value drawn uniformly within its rounding
interval and propagates it, so that the spread of a result over a set of
draws shows how many of its digits the data support (docs/scenarios.md,
precision).

  * jobs: the total of each municipality x LISA sector (the published,
    rounded figure) moves by U(-unit/2, unit/2); the buurten of that
    municipality and sector change in proportion. A cell published as zero
    stays zero.
  * population: per buurt the total moves by U(-unit/2, unit/2) inhabitants,
    and each income-class and household-type share by U(-share/2, share/2);
    the shares are renormalised and the segments rescaled by the ratios of
    their class and type shares (one raking step) and to the new total.

The draws are reproducible from the seed. Every scenario run with the same
seed gets the same perturbed inputs, so differences between scenarios keep
the shared part of the data uncertainty, as in the unperturbed comparison.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ikob2.segments.config import HOUSEHOLD_TYPES, INCOME_CLASSES


def _rng(seed: int, stream: int) -> np.random.Generator:
    return np.random.default_rng([int(seed), int(stream)])


def perturb_jobs(sector_jobs: pd.DataFrame, seed: int, unit: float) -> pd.DataFrame:
    """Jobs per buurt (index) and sector (columns) with each municipality x
    sector total moved within +-unit/2. The municipality is digits 2-5 of the
    CBS buurtcode (BU<gggg><ww><bb>)."""
    if unit <= 0:
        raise ValueError("The rounding unit must be positive.")
    jobs = sector_jobs.astype(float)
    gem = pd.Series(jobs.index.astype(str).str[2:6], index=jobs.index)
    totals = jobs.groupby(gem).sum()
    rng = _rng(seed, 1)
    shift = pd.DataFrame(rng.uniform(-unit / 2, unit / 2, totals.shape),
                         index=totals.index, columns=totals.columns)
    new = (totals + shift).clip(lower=0.0).where(totals > 0, 0.0)
    factor = (new / totals.where(totals > 0)).fillna(0.0)
    return jobs * factor.reindex(gem.to_numpy()).to_numpy()


def perturb_population(pop: pd.DataFrame, seed: int, count_unit: float,
                       share_unit: float, total_col: str = "inwoners"
                       ) -> pd.DataFrame:
    """Segment populations (columns '<type>_<class>', one row per buurt) with
    the buurt total moved within +-count_unit/2 and the class and type
    shares within +-share_unit/2 (as fractions), then renormalised."""
    if count_unit <= 0 or share_unit <= 0:
        raise ValueError("The rounding units must be positive.")
    out = pop.copy()
    cols = [f"{t}_{c}" for c in INCOME_CLASSES for t in HOUSEHOLD_TYPES
            if f"{t}_{c}" in pop.columns]
    seg = pop[cols].astype(float).fillna(0.0).to_numpy()
    n = len(pop)
    total = seg.sum(axis=1)
    rng = _rng(seed, 2)

    def shares(groups):
        idx = [[cols.index(c) for c in g] for g in groups]
        s = np.stack([seg[:, i].sum(axis=1) for i in idx], axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            return idx, np.where(total[:, None] > 0, s / total[:, None], 0.0)

    by_class = [[c for c in cols if c.rsplit("_", 1)[1] == k] for k in INCOME_CLASSES]
    by_type = [[c for c in cols if c.rsplit("_", 1)[0] == t] for t in HOUSEHOLD_TYPES]
    by_class = [g for g in by_class if g]
    by_type = [g for g in by_type if g]
    new = seg.copy()
    for groups in (by_class, by_type):
        idx, s = shares(groups)
        s2 = np.clip(s + rng.uniform(-share_unit / 2, share_unit / 2, s.shape), 0.0, None)
        s2 = np.where(s > 0, s2, 0.0)                 # an empty group stays empty
        norm = s2.sum(axis=1, keepdims=True)
        s2 = np.where(norm > 0, s2 / np.where(norm > 0, norm, 1.0), s)
        with np.errstate(invalid="ignore", divide="ignore"):
            ratio = np.where(s > 0, s2 / np.where(s > 0, s, 1.0), 1.0)
        for j, i in enumerate(idx):
            new[:, i] *= ratio[:, [j]]
    t2 = np.clip(total + rng.uniform(-count_unit / 2, count_unit / 2, n), 0.0, None)
    t2 = np.where(total > 0, t2, 0.0)
    cur = new.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        new *= np.where(cur > 0, t2 / np.where(cur > 0, cur, 1.0), 0.0)[:, None]
    out[cols] = new
    if total_col in out.columns:
        out[total_col] = pd.to_numeric(out[total_col], errors="coerce") * np.where(
            total > 0, t2 / np.where(total > 0, total, 1.0), 1.0)
    return out
