"""
Decay curves and clean sparsification pipeline.

No artificial cutoff.
Only epsilon-based sparsification.
Supports multi-parameter curves (logistic).

epsilon is a REQUIRED keyword argument: sparsification strength is a
modelling choice with a single owner (the CLI), never a per-module
default. Pass None to disable sparsification entirely.

All exponentials are clipped to a safe range before evaluation:
float32 exp() overflows above ~88.7, so we clip at 80. Weights at
the clip boundary (~1e-35) are far below any practical epsilon and
are removed by sparsification, so clipping never changes results.
"""

import numpy as np

from ikob2.core.numerics import DTYPE, ensure_dense, maybe_to_sparse


# Largest exponent we ever feed to np.exp (float32-safe).
_EXP_CLIP = 80.0


# ── Curve definitions ───────────────────────

def exponential(cost: np.ndarray, beta: float) -> np.ndarray:
    x = np.clip(-beta * cost, -_EXP_CLIP, _EXP_CLIP)
    return np.exp(x).astype(DTYPE, copy=False)


def power(cost: np.ndarray, beta: float) -> np.ndarray:
    safe_cost = np.maximum(cost, 1e-6)
    return (safe_cost ** (-beta)).astype(DTYPE, copy=False)


def logistic(cost: np.ndarray, alpha: float, omega: float,
             scaling: float = 1.0) -> np.ndarray:
    """
    Numerically stable logistic decay.

    f(c) = scaling / (1 + exp(alpha * (c - omega)))

    Split into two branches so the exponent is always <= 0 in
    magnitude-critical cases, and clipped so extreme costs
    (e.g. unreachable sentinels) cannot overflow float32.
    """

    x = alpha * (cost - omega)

    out = np.empty_like(cost, dtype=DTYPE)

    positive = x >= 0
    negative = ~positive

    # x >= 0: exp(x) can overflow for large x → clip.
    # At the clip boundary the weight is ~1e-35, i.e. zero
    # after epsilon-sparsification.
    x_pos = np.clip(x[positive], None, _EXP_CLIP)
    out[positive] = scaling / (1.0 + np.exp(x_pos))

    # x < 0: exp(-x) can overflow for very negative x → clip.
    x_neg = np.clip(-x[negative], None, _EXP_CLIP)
    exp_neg = np.exp(x_neg)
    out[negative] = scaling * exp_neg / (1.0 + exp_neg)

    return out.astype(DTYPE, copy=False)


_CURVES = {
    "exponential": exponential,
    "power": power,
    "logistic": logistic,
}


def get_decay_function(decay_type: str):
    if decay_type not in _CURVES:
        raise ValueError(f"Unknown decay type: {decay_type}")
    return _CURVES[decay_type]


# ── Clean decay pipeline ───────────────────

def apply_decay(
    cost,
    decay_type: str,
    params,
    *,
    epsilon: float | None,
):
    """
    epsilon : float or None (REQUIRED, keyword-only)
        Weights below epsilon are zeroed before sparsification.
        None disables sparsification. There is deliberately no
        default: the caller must state the modelling choice.
    """
    if isinstance(params, (int, float)):
        params = (float(params),)

    decay_fn = get_decay_function(decay_type)
    cost = ensure_dense(cost).astype(DTYPE, copy=False)

    weights = decay_fn(cost, *params)

    if epsilon is not None:
        weights[weights < epsilon] = 0.0

    return maybe_to_sparse(weights)