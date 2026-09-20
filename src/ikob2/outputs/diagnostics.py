"""Aggregated decay curve of the money gate per neighbourhood (paper, 4.5).

Within a segment the cost threshold is uniform on [low, high] with an atom at
zero: an increasing-hazard (IFR) survival function. A neighbourhood mixes
segments, and a mixture of survival functions has a lower hazard than its
components (Proschan): the aggregate curve can drift toward a decreasing
hazard (DFR), the power-law look of gravity decay. This module computes the
population-weighted mixture

    S_bar(c) = sum_s pi_s S_s(c),   S_s(c) = (1 - atom_s) clip((high_s - c) /
                                                (high_s - low_s), 0, 1)  (c > 0)

for every origin, its hazard, and the total-time-on-test (TTT) transform

    phi(u) = int_0^{F^-1(u)} S_bar / int_0^inf S_bar,    F = 1 - S_bar,

whose position against the diagonal shows the hazard class: above it concave
(IFR), on it exponential, below it convex (DFR). The summary reports per origin
the atom, the mean threshold and its CV, the share of the curve with an
increasing hazard, the TTT area (integral of phi(u) - u) of the aggregate and
of the segments alone, and their difference (`ttt_shift`, negative: the
aggregation moved the curve toward DFR).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def segment_survival(low: float, high: float, atom: float,
                     grid: np.ndarray) -> np.ndarray:
    """S(c) of one segment on a grid starting at 0 (S(0) = 1: free travel
    always clears)."""
    if high > low:
        s = np.clip((high - grid) / (high - low), 0.0, 1.0)
    else:
        s = (grid <= low).astype(float)
    s = (1.0 - atom) * s
    s[grid <= 0] = 1.0
    return s


def ttt(survival: np.ndarray, grid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(u, phi(u)) of a survival function given on a grid (trapezoids)."""
    area = np.concatenate([[0.0], np.cumsum(
        0.5 * (survival[1:] + survival[:-1]) * np.diff(grid))])
    mean = area[-1]
    if mean <= 0:
        return np.array([0.0, 1.0]), np.array([0.0, 1.0])
    u = 1.0 - survival
    return u, area / mean


def _area_over_diagonal(u: np.ndarray, phi: np.ndarray) -> float:
    """Integral of phi(u) - u over [0, 1] (0 for an exponential, 1/6 for a
    uniform without atom); phi is taken as 0 before the atom (u < F(0))."""
    order = np.argsort(u, kind="stable")
    u, phi = u[order], phi[order]
    if u[0] > 0:
        u, phi = np.concatenate([[0.0], u]), np.concatenate([[0.0], phi])
    if u[-1] < 1:
        u, phi = np.concatenate([u, [1.0]]), np.concatenate([phi, [1.0]])
    return float(np.trapezoid(phi - u, u))


def hazard(survival: np.ndarray, grid: np.ndarray, floor: float) -> np.ndarray:
    """-d log S / dc where S exceeds `floor` (NaN elsewhere)."""
    h = np.full_like(survival, np.nan)
    ok = survival > floor
    logs = np.where(ok, np.log(np.where(ok, survival, 1.0)), np.nan)
    h[1:-1] = -(logs[2:] - logs[:-2]) / (grid[2:] - grid[:-2])
    return h


def money_gate(envelope: pd.DataFrame, populations: pd.DataFrame, *,
               c_max: float | None = None, step: float = 1.0,
               floor: float = 0.05, ttt_band: float = 0.05
               ) -> dict[str, pd.DataFrame]:
    """Aggregated money-gate curves per origin.

    envelope    : household_type, income_class, low, high, atom per segment;
    populations : origins x segments (persons), columns '<type>_<class>';
    c_max, step : grid in EUR per trip (default: the highest bound, step 1);
    floor       : hazard and its monotonicity are read where the aggregate
                  survival is above this share of S_bar(0+) (the bulk);
    ttt_band    : the shape class of the whole curve: the TTT area above
                  +band is increasing hazard (IFR), below -band decreasing
                  (DFR), else near-exponential.
    Returns `curves` (buurtcode, c, survival, hazard), `ttt` (buurtcode, u,
    phi at 101 points) and `summary` (one row per origin)."""
    env = envelope.copy()
    env["segment"] = env["household_type"] + "_" + env["income_class"]
    env = env[env["segment"].isin(populations.columns)]
    c_max = float(c_max if c_max is not None else env["high"].max())
    grid = np.arange(0.0, c_max + step, step)
    seg = np.vstack([segment_survival(r.low, r.high, r.atom, grid)
                     for r in env.itertuples()])              # (S, G)
    own = [_area_over_diagonal(*ttt(s, grid)) if s[1:].max() > 0 else np.nan
           for s in seg]
    own = np.array(own)
    pop = populations[list(env["segment"])].fillna(0.0)
    curves, ttts, rows = [], [], []
    for code, w in pop.iterrows():
        w = w.to_numpy(dtype=float)
        if w.sum() <= 0:
            continue
        pi = w / w.sum()
        s_bar = pi @ seg
        u, phi = ttt(s_bar, grid)
        atom = 1.0 - s_bar[1]
        mean = float(np.trapezoid(s_bar, grid))
        second = 2.0 * float(np.trapezoid(grid * s_bar, grid))
        cv = np.sqrt(max(second - mean ** 2, 0.0)) / mean if mean > 0 else np.nan
        h = hazard(s_bar, grid, floor * max(s_bar[1], 1e-12))
        dh = np.diff(h[~np.isnan(h)])
        inc = float((dh > 0).mean()) if len(dh) else np.nan
        weights = pi[~np.isnan(own)]
        within = float((weights * own[~np.isnan(own)]).sum() / weights.sum()) \
            if weights.sum() > 0 else np.nan
        agg = _area_over_diagonal(u, phi)
        rows.append({
            "buurtcode": code, "population": float(w.sum()),
            "atom": float(atom), "mean_threshold": mean, "cv": cv,
            "share_hazard_increasing": inc,
            "hazard_class": ("increasing" if inc >= 0.8 else
                             "decreasing" if inc <= 0.2 else "mixed"),
            "ttt_area": agg, "ttt_area_within": within,
            "ttt_class": ("IFR" if agg > ttt_band else
                          "DFR" if agg < -ttt_band else "near-exponential"),
            "ttt_shift": agg - within})
        curves.append(pd.DataFrame({"buurtcode": code, "c": grid,
                                    "survival": s_bar, "hazard": h}))
        q = np.linspace(0.0, 1.0, 101)
        ttts.append(pd.DataFrame({"buurtcode": code, "u": q,
                                  "phi": np.interp(q, u, phi)}))
    return {"curves": pd.concat(curves, ignore_index=True),
            "ttt": pd.concat(ttts, ignore_index=True),
            "summary": pd.DataFrame(rows)}
