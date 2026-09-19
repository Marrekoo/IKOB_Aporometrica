"""Batched 2-D iterative proportional fitting."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class IpfResult:
    table: np.ndarray       # (B, R, C)
    iters: np.ndarray       # (B,)
    converged: np.ndarray   # (B,) bool
    max_diff: np.ndarray    # (B,)


def ipf_batch(seed, row_targets, col_targets, *, tol: float = 1e-8,
              max_iter: int = 200) -> IpfResult:
    """
    Rake B seed tables (B, R, C) to row targets (B, R) and column
    targets (B, C), independently per table.

    Same arithmetic as the R ipf2d(): non-positive/NaN seed cells are
    floored at 1e-8, column targets are rescaled to the row-target
    total, a row step is followed by a column step, and convergence is
    max(|row sums - targets|, |col sums - targets|) < tol. Each table
    stops at its own convergence iteration.

    Callers must pass non-negative targets with positive totals.
    """
    seed = np.asarray(seed, dtype=np.float64)
    row_t = np.asarray(row_targets, dtype=np.float64)
    col_t = np.asarray(col_targets, dtype=np.float64)
    B, R, C = seed.shape
    if row_t.shape != (B, R) or col_t.shape != (B, C):
        raise ValueError(
            f"target shapes {row_t.shape}/{col_t.shape} do not match "
            f"seed {seed.shape}")
    if np.any(row_t.sum(1) <= 0) or np.any(col_t.sum(1) <= 0):
        raise ValueError("IPF targets must have positive totals.")

    col_t = col_t * (row_t.sum(1) / col_t.sum(1))[:, None]
    tab = np.where(np.isfinite(seed) & (seed > 0), seed, 1e-8)

    active = np.ones(B, dtype=bool)
    converged = np.zeros(B, dtype=bool)
    iters = np.zeros(B, dtype=np.int64)
    max_diff = np.full(B, np.inf)

    for it in range(1, max_iter + 1):
        idx = np.flatnonzero(active)
        if idx.size == 0:
            break
        sub = tab[idx]
        rs = sub.sum(axis=2)
        rs[rs == 0] = 1e-12
        sub *= (row_t[idx] / rs)[:, :, None]
        cs = sub.sum(axis=1)
        cs[cs == 0] = 1e-12
        sub *= (col_t[idx] / cs)[:, None, :]
        tab[idx] = sub

        diff = np.maximum(
            np.abs(sub.sum(axis=2) - row_t[idx]).max(axis=1),
            np.abs(sub.sum(axis=1) - col_t[idx]).max(axis=1),
        )
        iters[idx] = it
        max_diff[idx] = diff
        done = diff < tol
        converged[idx[done]] = True
        active[idx[done]] = False

    return IpfResult(tab, iters, converged, max_diff)
