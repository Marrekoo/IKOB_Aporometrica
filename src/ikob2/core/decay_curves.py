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

from ikob2.core import families
from ikob2.core.numerics import DTYPE, ensure_dense, maybe_to_sparse


# Largest exponent we ever feed to np.exp (float32-safe).
_EXP_CLIP = 80.0


# ── Curve definitions ───────────────────────

def exponential(cost: np.ndarray, beta: float) -> np.ndarray:
    x = np.clip(-beta * cost, -_EXP_CLIP, _EXP_CLIP)
    return np.exp(x).astype(DTYPE, copy=False)


def power(cost: np.ndarray, beta: float) -> np.ndarray:
    """LEGACY power decay c^-beta. Not a survival function: it exceeds 1
    below c = 1 (and is capped only by the 1e-6 floor), so it cannot be a
    probability marginal for compose_filters. Use 'pareto' (power law
    beyond a threshold) or 'lomax' (shifted power law) instead."""
    safe_cost = np.maximum(cost, 1e-6)
    return (safe_cost ** (-beta)).astype(DTYPE, copy=False)


def weibull(cost: np.ndarray, shape: float, scale: float) -> np.ndarray:
    """
    Weibull survival function of a threshold distribution.

    S(t) = exp(-(t / scale) ** shape),   shape k > 0, scale eta > 0.

    k > 1 is increasing-hazard (soft-threshold decay, the shape found
    for the time margin), k = 1 is exponential with rate 1/scale, and
    k < 1 is decreasing-hazard. S(0) = 1 exactly, so this is a pure
    survival function (no atom at zero).

    The exponent is clipped like the other curves: (t/eta)^k overflows
    float32 for large t and k, and weights at the clip boundary are
    ~1e-35, removed by epsilon-sparsification anyway.
    """
    if not (shape > 0 and np.isfinite(shape)):
        raise ValueError(f"weibull shape must be positive and finite, got {shape}")
    if not (scale > 0 and np.isfinite(scale)):
        raise ValueError(f"weibull scale must be positive and finite, got {scale}")
    ratio = np.maximum(cost, 0.0).astype(np.float64, copy=False) / scale
    with np.errstate(over="ignore"):
        hazard = np.minimum(ratio ** shape, _EXP_CLIP)
    return np.exp(-hazard).astype(DTYPE, copy=False)


def uniform(cost: np.ndarray, low: float, high: float) -> np.ndarray:
    """
    Survival function of a threshold uniform on [low, high].

    S(c) = Pr(X >= c) = 1                    for c <= low
                      = (high - c)/(high - low)  for low < c < high
                      = 0                    for c >= high

    This is the paper's cost margin within one segment: the reference-
    budget envelope gives an interval [low, high] of plausible per-trip
    budgets, taken uniform, so S_M is piecewise linear. It expresses
    identification uncertainty over budget assumptions, not observed
    dispersion across households. high == low is the degenerate
    (isochrone-like) step: 1 up to and including low, 0 above.

    Left-continuous like every survival function here, so a trip costing
    exactly `low` still clears the gate, and free travel (c = 0) always
    gives S = 1 when low >= 0. No atom at zero is modelled.
    """
    if not (np.isfinite(low) and np.isfinite(high)):
        raise ValueError(f"uniform bounds must be finite, got ({low}, {high})")
    if low < 0 or high < low:
        raise ValueError(
            f"uniform requires 0 <= low <= high, got low={low}, high={high}"
        )
    c = cost.astype(np.float64, copy=False)
    if high == low:
        out = (c <= low).astype(np.float64)
    else:
        out = np.clip((high - c) / (high - low), 0.0, 1.0)
    return out.astype(DTYPE, copy=False)


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


def with_atom(survival: np.ndarray, cost: np.ndarray, atom: float) -> np.ndarray:
    """
    Add an atom at zero to a survival function.

    `atom` (pi) is the share of the population for whom NO positive
    value of the margin is acceptable: S(0+) = 1 - pi. Applied as

        S_pi(x) = S(x)            for x <= 0   (zero cost/time always clears)
                = (1 - pi) * S(x) for x >  0

    so a priced trip is acceptable to at most 1 - pi of the segment,
    while walking and private cycling (c = 0) are untouched. For the
    cost margin this is the segment whose protected basket exhausts its
    income; see the paper's M2 (f(0+, 0+) = 1 - pi_M < 1).

    Multiplying by (1 - pi) keeps S non-increasing, in [0, 1], and
    tending to zero, so the result is still a survival function.
    """
    if not (0.0 <= atom <= 1.0):
        raise ValueError(f"atom must be in [0, 1], got {atom}")
    if atom == 0.0:
        return survival
    factor = DTYPE(1.0 - atom)
    return np.where(cost > 0, survival * factor, survival).astype(
        DTYPE, copy=False)


def _family_curve(name: str):
    """Survival curve of a core.families family, as a float32 marginal."""
    fam = families.get_family(name)

    def curve(cost: np.ndarray, *params: float) -> np.ndarray:
        families.validate_params(name, params)
        log_s = fam.log_survival(np.maximum(cost, 0.0).astype(np.float64,
                                                              copy=False),
                                 *[float(p) for p in params])
        return np.exp(log_s).astype(DTYPE, copy=False)

    curve.__name__ = name
    curve.__doc__ = f"Survival function of the '{name}' family " \
                    f"(parameters {fam.params}); see core.families."
    return curve


_CURVES = {
    "exponential": exponential,
    "power": power,
    "logistic": logistic,
    "weibull": weibull,
    "uniform": uniform,
    # further survival families (core.families)
    "lomax": _family_curve("lomax"),
    "pareto": _family_curve("pareto"),
    "tanner": _family_curve("tanner"),
    "gamma": _family_curve("gamma"),
    "lognormal": _family_curve("lognormal"),
    "loglogistic": _family_curve("loglogistic"),
    "step": _family_curve("step"),
    "quadratic_ramp": _family_curve("quadratic_ramp"),
    # knot families: parameters are the flattened knots (z0, f0, z1, f1, ...)
    "piecewise_linear": _family_curve("piecewise_linear"),
    "piecewise_quadratic": _family_curve("piecewise_quadratic"),
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