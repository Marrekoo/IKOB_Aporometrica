"""Where to put extra hubs (scenario S2): low accessibility, low bicycle
ownership.

Candidates are buurt centroids. A buurt scores by the weighted mean of two
percentile ranks, lowest first: its baseline accessibility (population-weighted
mean over its segments) and the share of its residents with a private bicycle.
Buurten are taken in that order and skipped when they lie within
`min_spacing_m` of an existing or already chosen hub, until the requested
number of new hubs is reached, or, without a number, until no candidate
fits. The hub is placed at the buurt centroid: a buurt-level siting, not a
street-level one. Restricting the candidates to some buurten (`within`)
sites the hubs in a target area only.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from ikob2.params import DEFAULTS


def score_candidates(candidates: pd.DataFrame,
                     access_weight: float = DEFAULTS.siting.access_weight
                     ) -> pd.Series:
    """Score in [0, 1]; lower = more in need (low access, low ownership)."""
    if not 0.0 <= access_weight <= 1.0:
        raise ValueError("access_weight must be in [0, 1].")
    a = candidates["access"].rank(pct=True, method="average")
    b = candidates["bike_share"].rank(pct=True, method="average")
    return access_weight * a + (1.0 - access_weight) * b


def propose_hubs(candidates: pd.DataFrame, existing_xy: np.ndarray,
                 n_new: int | None, *,
                 min_spacing_m: float = DEFAULTS.siting.min_spacing_m,
                 access_weight: float = DEFAULTS.siting.access_weight,
                 within=None) -> pd.DataFrame:
    """The `n_new` extra hubs (None: as many as fit). candidates: code, x, y
    (RD New metres), access, bike_share; `within` restricts them to these
    codes (the ranks are taken among them). Returns the chosen buurten in
    order of choice with their score and the columns of the candidates."""
    c = candidates.dropna(subset=["access", "bike_share", "x", "y"]).copy()
    if within is not None:
        within = {str(w) for w in within}
        unknown = sorted(within - set(candidates["code"].astype(str)))
        if unknown:
            raise ValueError(f"{len(unknown)} buurt(en) to site in are not "
                             f"candidates, e.g. {unknown[:5]}.")
        c = c[c["code"].astype(str).isin(within)]
    c["score"] = score_candidates(c, access_weight)
    c = c.sort_values(["score", "code"]).reset_index(drop=True)
    taken = [np.asarray(p, dtype=float) for p in np.asarray(existing_xy,
                                                            dtype=float)
             .reshape(-1, 2)]
    chosen = []
    for row in c.itertuples(index=False):
        if n_new is not None and len(chosen) >= n_new:
            break
        pt = np.array([row.x, row.y])
        if taken and cKDTree(np.array(taken)).query(pt)[0] < min_spacing_m:
            continue
        taken.append(pt)
        chosen.append(row._asdict())
    if n_new is not None and len(chosen) < n_new:
        raise ValueError(f"Only {len(chosen)} of {n_new} hubs fit with a "
                         f"spacing of {min_spacing_m:g} m; lower the spacing "
                         f"or the density factor.")
    return pd.DataFrame(chosen)
