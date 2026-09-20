"""
core/compose.py — joint tolerance filters via survival copulas.

A trip weight is the joint survival probability that a group member's
time tolerance exceeds the trip time AND their money tolerance exceeds
the trip cost:

    w = P(tau > T, mu > C) = C_hat( F_t(T), F_c(C) )

where F_t, F_c are marginal survival filters (e.g. the logistic in
decay_curves, with scaling == 1) and C_hat is a survival copula.

Conventions, matching decay_curves.py:
  * epsilon is a REQUIRED keyword argument with a single owner (the
    CLI). It is applied ONCE, to the composed matrix. This is exact:
    C(u,v) <= min(u,v), so any entry with a sub-epsilon marginal is
    sub-epsilon after composition too.
  * Marginals fed to the copula MUST be unscaled probabilities in
    [0, 1]. The legacy `scaling` factor is a mode-availability
    Bernoulli filter and is applied multiplicatively AFTER the
    copula, via the `scaling` argument.
  * cost_weights=None means F_c == 1; by the copula boundary
    condition C(u, 1) = u this collapses exactly to the pure time
    filter, so time-only runs need no separate code path.
  * Frank uses expm1/log1p and clips |theta| at _THETA_MAX (beyond
    float32 resolution of the tail); the independence limit theta->0
    is handled by an explicit switch to the product, mirroring the
    branch-and-clip style of decay_curves.logistic.

Copula parameter semantics (Frank):
  theta -> -inf : countermonotone (substitution regime, TVOM-like)
  theta  =   0  : independence (product of filters) — DEFAULT
  theta -> +inf : comonotone (binding constraint decides)
"""

import numpy as np

from ikob2.core.numerics import DTYPE, ensure_dense, maybe_to_sparse

# theta below this is numerically indistinguishable from independence.
_THETA_INDEPENDENCE_TOL = 1e-4
# |theta| above this is indistinguishable from the Frechet bound in
# float32; users wanting the exact bound should name it explicitly.
_THETA_MAX = 35.0


# ── Copula families (elementwise on ndarrays in [0,1]) ──────────────

def independence(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    return u * v


def comonotone(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    return np.minimum(u, v)


def countermonotone(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    return np.maximum(u + v - 1.0, 0.0)


def frank(u: np.ndarray, v: np.ndarray, theta: float) -> np.ndarray:
    """
    Frank copula, numerically stable form.

        C(u,v) = -(1/theta) * log1p( expm1(-theta*u) * expm1(-theta*v)
                                     / expm1(-theta) )

    Radially symmetric, so the survival copula is Frank with the same
    theta — we can apply it to survival marginals directly.
    """
    if abs(theta) < _THETA_INDEPENDENCE_TOL:
        return u * v
    theta = float(np.clip(theta, -_THETA_MAX, _THETA_MAX))

    # Work in float64 internally: the expm1 products cancel harshly
    # near the corners in float32. Cast back once at the end.
    u64 = u.astype(np.float64, copy=False)
    v64 = v.astype(np.float64, copy=False)

    num = np.expm1(-theta * u64) * np.expm1(-theta * v64)
    den = np.expm1(-theta)
    out = (-1.0 / theta) * np.log1p(num / den)

    # Guard float rounding at the corners; the true value is in [0,1].
    np.clip(out, 0.0, 1.0, out=out)
    return out.astype(DTYPE, copy=False)


def gumbel_hougaard(u: np.ndarray, v: np.ndarray, theta: float) -> np.ndarray:
    """
    Gumbel-Hougaard copula, theta >= 1, evaluated on SURVIVAL values.

        C(u,v) = exp( -[ (-ln u)^theta + (-ln v)^theta ]^(1/theta) )

    With u = S_T(t), v = S_M(c) this is exactly the joint survival
    f_theta(t, c) = exp{-[Lambda_T^theta + Lambda_M^theta]^(1/theta)}
    of the two-gate model (cumulative hazards Lambda = -ln S): theta = 1
    is independence (u * v), theta -> inf is min(u, v), and Kendall's
    tau = 1 - 1/theta. Positive dependence only; theta < 1 is rejected.

    Unlike Frank this copula is NOT radially symmetric, so it is only
    correct applied to survival marginals directly, as done here.

    Stable form: with a = max(L_u, L_v), b = min(L_u, L_v),
        [L_u^theta + L_v^theta]^(1/theta) = a * (1 + (b/a)^theta)^(1/theta)
    which never overflows however large theta is (b/a <= 1), and
    handles S = 0 (L = inf -> result 0) and S = 1 (L = 0) exactly.
    """
    if not np.isfinite(theta) or theta < 1.0:
        raise ValueError(f"gumbel theta must be finite and >= 1, got {theta}")
    if theta == 1.0:
        return u * v

    with np.errstate(divide="ignore", invalid="ignore", over="ignore",
                   under="ignore"):
        lu = -np.log(u.astype(np.float64, copy=False))
        lv = -np.log(v.astype(np.float64, copy=False))
        a = np.maximum(lu, lv)
        b = np.minimum(lu, lv)
        ratio = np.where(a > 0.0, b / np.where(a > 0.0, a, 1.0), 0.0)
        # a = inf: b/a may be nan; the survival is 0 there anyway.
        total = np.where(np.isinf(a), np.inf,
                         a * (1.0 + ratio ** theta) ** (1.0 / theta))
        out = np.exp(-total)

    np.clip(out, 0.0, 1.0, out=out)
    return out.astype(DTYPE, copy=False)


_PARAMETRIC_COPULAS = {
    "frank": frank,
    "gumbel": gumbel_hougaard,
}

_COPULAS = {
    "independence": lambda u, v: independence(u, v),
    "comonotone": lambda u, v: comonotone(u, v),
    "countermonotone": lambda u, v: countermonotone(u, v),
}


def get_copula(family: str):
    if family in _PARAMETRIC_COPULAS:
        raise ValueError(f"Copula '{family}' is parametric; pass theta "
                         f"via compose_filters.")
    if family not in _COPULAS:
        raise ValueError(f"Unknown copula family: {family}")
    return _COPULAS[family]


# ── Invariant: Frechet–Hoeffding sandwich ────────────────────────────

def frechet_violation(u: np.ndarray, v: np.ndarray,
                      composed: np.ndarray) -> float:
    """
    Max violation of  max(u+v-1, 0) <= C <= min(u,v),  which holds for
    EVERY valid copula and every parameter value. Returns 0.0 when the
    sandwich holds; certified invariant #3 alongside additivity and
    nesting. Cost: three elementwise passes, milliseconds at n=1149.
    """
    lower = np.maximum(u + v - 1.0, 0.0)
    upper = np.minimum(u, v)
    viol_low = float(np.max(lower - composed, initial=0.0))
    viol_high = float(np.max(composed - upper, initial=0.0))
    return max(viol_low, viol_high, 0.0)


# ── Composer ─────────────────────────────────────────────────────────

def compose_filters(
    time_weights,
    cost_weights=None,
    *,
    family: str = "independence",
    theta: float | None = None,
    scaling: float = 1.0,
    epsilon: float | None,
    check_frechet: bool = True,
    frechet_atol: float = 5e-6,
):
    """
    Compose a time-tolerance filter and a money-tolerance filter into
    a joint trip-weight matrix.

    Parameters
    ----------
    time_weights : (n, n) marginal survival filter F_t(T), unscaled.
    cost_weights : (n, n) marginal survival filter F_c(C), unscaled,
        or None for cost-free modes (collapses to time_weights).
    family, theta : copula choice. theta is required for the
        parametric families ("frank", "gumbel") and forbidden otherwise (fail loud, no silent defaults).
    scaling : mode-availability factor, applied AFTER composition.
    epsilon : REQUIRED keyword, single owner (CLI). Applied once to
        the composed, scaled matrix. None disables sparsification.
    check_frechet : run the sandwich invariant on the raw copula
        output (before scaling/epsilon, where the bound is exact).
    """
    u = ensure_dense(time_weights).astype(DTYPE, copy=False)

    if not (0.0 <= scaling <= 1.0):
        raise ValueError(f"scaling must be in [0,1], got {scaling}")

    if cost_weights is None:
        # C(u, 1) = u for every copula: pure time filter, exactly.
        composed = u.copy()
    else:
        v = ensure_dense(cost_weights).astype(DTYPE, copy=False)
        if u.shape != v.shape:
            raise ValueError(
                f"Marginal shape mismatch: time {u.shape} vs cost {v.shape}"
            )
        for name, m in (("time", u), ("cost", v)):
            lo, hi = float(m.min()), float(m.max())
            if lo < 0.0 or hi > 1.0 + 1e-6:
                raise ValueError(
                    f"{name} marginal not a probability filter: "
                    f"range [{lo}, {hi}]. Did a scaled curve leak in? "
                    f"Pass scaling= to the composer instead."
                )

        if family in _PARAMETRIC_COPULAS:
            if theta is None:
                raise ValueError(f"family='{family}' requires theta.")
            composed = _PARAMETRIC_COPULAS[family](u, v, theta)
        else:
            if theta is not None:
                raise ValueError(
                    f"theta given but family '{family}' takes no parameter."
                )
            composed = get_copula(family)(u, v)

        # independence, comonotone and countermonotone satisfy the bounds
        # by construction; only the parametric families need the check
        if check_frechet and family in _PARAMETRIC_COPULAS:
            viol = frechet_violation(u, v, composed)
            if viol > frechet_atol:
                raise AssertionError(
                    f"Frechet–Hoeffding sandwich violated by {viol:.3e} "
                    f"(family={family}, theta={theta}). Copula "
                    f"evaluation or parameter handling is broken."
                )

    if scaling != 1.0:
        composed = composed * DTYPE(scaling)

    if epsilon is not None:
        composed[composed < epsilon] = 0.0

    return maybe_to_sparse(composed)