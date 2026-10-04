"""
Survival-function families for threshold marginals, and their hazard
tools.

A decay function normalised to 1 at zero impedance is the survival
function f(z) = Pr(X >= z) of a non-negative threshold X
(docs/model_theory.md). Its log-derivative, the hazard
h(z) = -d log f / dz, is the acceptance lost per extra unit of
impedance, the elasticity is eta = -z h(z), and separability of time and
cost is additivity of log f. Every family below is a proper survival
function (f(0) = 1, non-increasing, -> 0) and so can be composed as a
probability marginal; each has closed-form log-survival and hazard.

    exponential  rate b                    h = b            (constant)
    weibull      shape k, scale s          h = (k/s)(z/s)^(k-1)
    lomax        shape a, scale s          h = a / (s + z)  (decreasing)
    triangular      low, mode, high        triangular density on [low, high]
                                           with its peak at mode (survival
                                           1 - CDF); quadratic_ramp is the
                                           mode = midpoint case
    quadratic_ramp  low, high              f = 1 up to low, then a smooth
                                           two-piece quadratic (C1) ramp
                                           to 0 at high
    piecewise_linear     knots (z, f)      linear interpolation of a
                                           survival function given at
                                           knots; 0 beyond the last knot
    piecewise_quadratic  knots (z, f)      shape-preserving C1 quadratic
                                           spline through the same knots
    pareto       shape a, threshold z0     f = 1 (z <= z0), (z/z0)^-a
                                           beyond: constant elasticity a;
                                           the SURVIVAL form of the power
                                           law f = z^-a, which is not
                                           normalisable at 0
    tanner       rho, chi, scale s         f = (1+z/s)^-rho exp(-chi z)
                                           the gamma friction factor
                                           a z^-rho e^-chi z shifted so
                                           that f(0) = 1; rho = 0 is
                                           exponential, chi = 0 Lomax
    gamma        shape v, scale t          gamma-distributed threshold;
                                           hazard rises to 1/t for v > 1
    lognormal    mu, sigma                 ln X ~ N(mu, sigma^2)
    loglogistic  shape k, scale s          f = 1 / (1 + (z/s)^k)
    step         threshold t*              f = 1 for z <= t*: the isochrone

All arrays are float64 here; decay_curves wraps them to the model dtype.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy import integrate, special, stats

_TINY = 1e-300


@dataclass(frozen=True)
class Family:
    name: str
    params: tuple[str, ...]
    log_survival: Callable          # (z, *params) -> log f(z), <= 0
    hazard: Callable                # (z, *params) -> h(z)
    validate: Callable              # (*params) -> None or ValueError
    mean: Callable | None = None    # (*params) -> E[X] = integral of f
    shape: Callable | None = None   # (*params) -> hazard class string
    variadic: bool = False          # knot families: any even number of params


def _positive(name, value):
    if not (np.isfinite(value) and value > 0):
        raise ValueError(f"{name} must be positive and finite, got {value}")


def _nonneg(name, value):
    if not (np.isfinite(value) and value >= 0):
        raise ValueError(f"{name} must be non-negative and finite, "
                         f"got {value}")


def _z(z):
    return np.maximum(np.asarray(z, dtype=np.float64), 0.0)


# ── exponential ──────────────────────────────────────────────────────

def _exp_logs(z, b):
    return -b * _z(z)


def _exp_h(z, b):
    return np.full_like(_z(z), b)


# ── weibull ──────────────────────────────────────────────────────────

def _weibull_logs(z, k, s):
    with np.errstate(over="ignore"):
        return -((_z(z) / s) ** k)


def _weibull_h(z, k, s):
    with np.errstate(divide="ignore", invalid="ignore"):
        return (k / s) * (_z(z) / s) ** (k - 1.0)


# ── lomax ────────────────────────────────────────────────────────────

def _lomax_logs(z, a, s):
    return -a * np.log1p(_z(z) / s)


def _lomax_h(z, a, s):
    return a / (s + _z(z))


# ── pareto (power law beyond a threshold) ────────────────────────────

def _pareto_logs(z, a, z0):
    z = _z(z)
    with np.errstate(divide="ignore"):
        return np.where(z > z0, -a * np.log(np.maximum(z, z0) / z0), 0.0)


def _pareto_h(z, a, z0):
    z = _z(z)
    with np.errstate(divide="ignore"):
        return np.where(z > z0, a / np.maximum(z, z0), 0.0)


# ── tanner (gamma friction factor, shifted) ──────────────────────────

def _tanner_logs(z, rho, chi, s):
    z = _z(z)
    return -rho * np.log1p(z / s) - chi * z


def _tanner_h(z, rho, chi, s):
    return rho / (s + _z(z)) + chi


# ── gamma distribution ───────────────────────────────────────────────

def _gamma_logs(z, v, t):
    return stats.gamma.logsf(_z(z), a=v, scale=t)


def _gamma_h(z, v, t):
    z = _z(z)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        h = np.exp(stats.gamma.logpdf(z, a=v, scale=t)
                   - stats.gamma.logsf(z, a=v, scale=t))
    # far tail: logsf underflows and h converges to the rate 1/t
    return np.where(np.isfinite(h), h, 1.0 / t)


# ── lognormal ────────────────────────────────────────────────────────

def _lognormal_logs(z, mu, sigma):
    return stats.lognorm.logsf(_z(z), s=sigma, scale=math.exp(mu))


def _lognormal_h(z, mu, sigma):
    z = _z(z)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        h = np.exp(stats.lognorm.logpdf(z, s=sigma, scale=math.exp(mu))
                   - stats.lognorm.logsf(z, s=sigma, scale=math.exp(mu)))
    return np.where(np.isfinite(h), h, 0.0)


# ── log-logistic ─────────────────────────────────────────────────────

def _loglogistic_logs(z, k, s):
    with np.errstate(over="ignore"):
        return -np.log1p((_z(z) / s) ** k)


def _loglogistic_h(z, k, s):
    z = _z(z)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        u = (z / s) ** k
        h = (k / z) * u / (1.0 + u)
    return np.where(z > 0, h, 0.0 if k > 1 else (np.inf if k < 1 else 1.0 / s))


# ── step (isochrone) ─────────────────────────────────────────────────

def _step_logs(z, t):
    return np.where(_z(z) <= t, 0.0, -np.inf)


def _step_h(z, t):
    return np.where(_z(z) < t, 0.0, np.inf)


# ── means, hazard classes ────────────────────────────────────────────

def _mean_numeric(logs, *params):
    upper = np.inf
    val, _ = integrate.quad(lambda x: math.exp(logs(np.array(x), *params)),
                            0.0, upper, limit=200)
    return float(val)


FAMILIES: dict[str, Family] = {}


def _register(fam: Family):
    FAMILIES[fam.name] = fam


_register(Family(
    "exponential", ("rate",), _exp_logs, _exp_h,
    lambda b: _positive("rate", b),
    mean=lambda b: 1.0 / b, shape=lambda b: "constant"))
_register(Family(
    "weibull", ("shape", "scale"), _weibull_logs, _weibull_h,
    lambda k, s: (_positive("shape", k), _positive("scale", s)),
    mean=lambda k, s: s * special.gamma(1.0 + 1.0 / k),
    shape=lambda k, s: ("constant" if k == 1 else
                        "increasing" if k > 1 else "decreasing")))
_register(Family(
    "lomax", ("alpha", "scale"), _lomax_logs, _lomax_h,
    lambda a, s: (_positive("alpha", a), _positive("scale", s)),
    mean=lambda a, s: s / (a - 1.0) if a > 1 else math.inf,
    shape=lambda a, s: "decreasing"))
_register(Family(
    "pareto", ("alpha", "z0"), _pareto_logs, _pareto_h,
    lambda a, z0: (_positive("alpha", a), _positive("z0", z0)),
    mean=lambda a, z0: z0 * a / (a - 1.0) if a > 1 else math.inf,
    shape=lambda a, z0: "decreasing"))
def _validate_tanner(r, c, s):
    _nonneg("rho", r)
    _nonneg("chi", c)
    _positive("scale", s)
    if r == 0 and c == 0:
        raise ValueError("tanner needs rho > 0 or chi > 0 (otherwise the "
                         "impedance never deters)")


def _validate_lognormal(mu, sigma):
    if not np.isfinite(mu):
        raise ValueError(f"mu must be finite, got {mu}")
    _positive("sigma", sigma)


_register(Family(
    "tanner", ("rho", "chi", "scale"), _tanner_logs, _tanner_h,
    _validate_tanner,
    mean=lambda r, c, s: (1.0 / c if r == 0 else
                          s / (r - 1.0) if (c == 0 and r > 1) else
                          math.inf if c == 0 else
                          _mean_numeric(_tanner_logs, r, c, s)),
    shape=lambda r, c, s: ("constant" if r == 0 else "decreasing")))
_register(Family(
    "gamma", ("shape", "scale"), _gamma_logs, _gamma_h,
    lambda v, t: (_positive("shape", v), _positive("scale", t)),
    mean=lambda v, t: v * t,
    shape=lambda v, t: ("constant" if v == 1 else
                        "increasing" if v > 1 else "decreasing")))
_register(Family(
    "lognormal", ("mu", "sigma"), _lognormal_logs, _lognormal_h,
    _validate_lognormal,
    mean=lambda mu, sg: math.exp(mu + sg * sg / 2.0),
    shape=lambda mu, sg: "unimodal"))
_register(Family(
    "loglogistic", ("shape", "scale"), _loglogistic_logs, _loglogistic_h,
    lambda k, s: (_positive("shape", k), _positive("scale", s)),
    mean=lambda k, s: (s * (math.pi / k) / math.sin(math.pi / k)
                       if k > 1 else math.inf),
    shape=lambda k, s: "unimodal" if k > 1 else "decreasing"))
_register(Family(
    "step", ("threshold",), _step_logs, _step_h,
    lambda t: _nonneg("threshold", t),
    mean=lambda t: float(t), shape=lambda t: "degenerate"))


# ── quadratic ramp (smooth two-piece quadratic between low and high) ──

def _ramp_u(z, low, high):
    return np.clip((_z(z) - low) / (high - low), 0.0, 1.0)


def _ramp_survival(z, low, high):
    u = _ramp_u(z, low, high)
    return np.where(u <= 0.5, 1.0 - 2.0 * u * u, 2.0 * (1.0 - u) ** 2)


def _ramp_logs(z, low, high):
    with np.errstate(divide="ignore"):
        return np.log(_ramp_survival(z, low, high))


def _ramp_h(z, low, high):
    z = _z(z)
    u = np.clip((z - low) / (high - low), 0.0, 1.0)
    w = high - low
    with np.errstate(divide="ignore", invalid="ignore"):
        lower = 4.0 * u / (w * (1.0 - 2.0 * u * u))
        upper = 2.0 / (w * (1.0 - u))
    h = np.where(u <= 0.5, lower, upper)
    return np.where((z <= low), 0.0, np.where(z >= high, np.inf, h))


def _validate_ramp(low, high):
    _nonneg("low", low)
    _positive("high", high)
    if not high > low:
        raise ValueError(f"quadratic_ramp needs low < high, got "
                         f"({low}, {high})")


# ── triangular density with a free mode ──────────────────────────────

def _tri_cdf(z, a, c, b):
    z = _z(z)
    with np.errstate(divide="ignore", invalid="ignore"):
        rising = (z - a) ** 2 / ((b - a) * (c - a))
        falling = 1.0 - (b - z) ** 2 / ((b - a) * (b - c))
    out = np.where(z <= a, 0.0,
                   np.where(z >= b, 1.0,
                            np.where(z <= c, rising, falling)))
    return np.clip(np.nan_to_num(out, nan=0.0), 0.0, 1.0)


def _tri_logs(z, a, c, b):
    with np.errstate(divide="ignore"):
        return np.log(1.0 - _tri_cdf(z, a, c, b))


def _tri_h(z, a, c, b):
    z = _z(z)
    with np.errstate(divide="ignore", invalid="ignore"):
        pdf = np.where(z <= c, 2.0 * (z - a) / ((b - a) * (c - a)),
                       2.0 * (b - z) / ((b - a) * (b - c)))
        h = pdf / (1.0 - _tri_cdf(z, a, c, b))
    pdf_ok = np.where((z > a) & (z < b), h, 0.0)
    return np.where(z >= b, np.inf, np.where(z <= a, 0.0,
                                             np.nan_to_num(pdf_ok, nan=0.0)))


def _validate_tri(low, mode, high):
    _nonneg("low", low)
    _positive("high", high)
    if not (np.isfinite(mode) and low <= mode <= high and low < high):
        raise ValueError(f"triangular needs low <= mode <= high and "
                         f"low < high, got ({low}, {mode}, {high})")


# ── piecewise linear / piecewise quadratic through knots ─────────────

def _knots(params):
    a = np.asarray(params, dtype=np.float64)
    return a[0::2], a[1::2]


def _validate_knots(*params):
    if len(params) < 4 or len(params) % 2:
        raise ValueError("knots need at least two (z, f) pairs "
                         "(an even number of values)")
    x, y = _knots(params)
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise ValueError("knots must be finite")
    if x[0] != 0.0 or y[0] != 1.0:
        raise ValueError("the first knot must be (0, 1): a survival "
                         "function is 1 at zero impedance")
    if np.any(np.diff(x) <= 0):
        raise ValueError("knot impedances must be strictly increasing")
    if np.any(y < 0) or np.any(y > 1):
        raise ValueError("knot survival values must lie in [0, 1]")
    if np.any(np.diff(y) > 0):
        raise ValueError("knot survival values must be non-increasing")


def _pl_survival(z, *params):
    x, y = _knots(params)
    return np.interp(_z(z), x, y, right=0.0)


def _pl_logs(z, *params):
    with np.errstate(divide="ignore"):
        return np.log(_pl_survival(z, *params))


def _pl_h(z, *params):
    x, y = _knots(params)
    z = _z(z)
    k = np.clip(np.searchsorted(x, z, side="right") - 1, 0, len(x) - 2)
    slope = (y[k + 1] - y[k]) / (x[k + 1] - x[k])
    f = _pl_survival(z, *params)
    with np.errstate(divide="ignore", invalid="ignore"):
        h = -slope / f
    return np.where(z > x[-1], np.inf, np.where(f > 0, h, np.inf))


def _pl_mean(*params):
    x, y = _knots(params)
    return float(np.sum(np.diff(x) * (y[:-1] + y[1:]) / 2.0))


def _qspline(params):
    """Shape-preserving C1 quadratic spline through the knots: returns
    piece starts, ends and coefficients (a, b, c) of f = a + b u + c u^2
    with u = z - start. Slopes at the knots are limited (harmonic mean of
    neighbouring secants, capped at twice a secant) and every knot
    interval gets one extra breakpoint chosen so that the spline is
    monotone; see the module notes."""
    x, y = _knots(params)
    n = len(x)
    h = np.diff(x)
    delta = np.diff(y) / h                      # secant slopes, <= 0
    d = np.zeros(n)
    for i in range(1, n - 1):
        a, b = delta[i - 1], delta[i]
        d[i] = 2.0 * a * b / (a + b) if a * b > 0 else 0.0
    if n == 2:
        d[0] = d[1] = delta[0]
    else:
        def end_slope(dl, d_next):
            e = (3.0 * dl - d_next) / 2.0
            if e * dl <= 0:
                return 0.0
            return float(np.sign(dl) * min(abs(e), 2.0 * abs(dl)))
        d[0] = end_slope(delta[0], d[1])
        d[-1] = end_slope(delta[-1], d[-2])
    starts, ends, A, B, C = [], [], [], [], []
    for i in range(n - 1):
        dl = delta[i]
        if dl == 0.0:
            starts.append(x[i])
            ends.append(x[i + 1])
            A.append(y[i])
            B.append(0.0)
            C.append(0.0)
            continue
        al, be = d[i] / dl, d[i + 1] / dl
        t = (1.0 - be) / (al - be) if (al - 1.0) * (be - 1.0) < 0 else 0.5
        xi = x[i] + t * h[i]
        s = 2.0 * dl - t * d[i] - (1.0 - t) * d[i + 1]
        # [x_i, xi]
        starts.append(x[i])
        ends.append(xi)
        A.append(y[i])
        B.append(d[i])
        C.append((s - d[i]) / (2.0 * t * h[i]))
        # [xi, x_{i+1}]
        y_xi = y[i] + (d[i] + s) / 2.0 * t * h[i]
        starts.append(xi)
        ends.append(x[i + 1])
        A.append(y_xi)
        B.append(s)
        C.append((d[i + 1] - s) / (2.0 * (1.0 - t) * h[i]))
    return (np.array(starts), np.array(ends), np.array(A), np.array(B),
            np.array(C), float(x[-1]))


def _pq_eval(z, params):
    starts, ends, A, B, C, last = _qspline(params)
    z = _z(z)
    k = np.clip(np.searchsorted(starts, z, side="right") - 1, 0,
                len(starts) - 1)
    u = z - starts[k]
    f = A[k] + B[k] * u + C[k] * u * u
    slope = B[k] + 2.0 * C[k] * u
    f = np.where(z > last, 0.0, np.clip(f, 0.0, 1.0))
    return f, slope


def _pq_survival(z, *params):
    return _pq_eval(z, params)[0]


def _pq_logs(z, *params):
    with np.errstate(divide="ignore"):
        return np.log(_pq_survival(z, *params))


def _pq_h(z, *params):
    f, slope = _pq_eval(z, params)
    z = _z(z)
    last = _knots(params)[0][-1]
    with np.errstate(divide="ignore", invalid="ignore"):
        h = -slope / f
    return np.where(z > last, np.inf, np.where(f > 0, h, np.inf))


def _pq_mean(*params):
    starts, ends, A, B, C, _ = _qspline(params)
    w = ends - starts
    # exact integral of a quadratic over each piece
    return float(np.sum(A * w + B * w ** 2 / 2.0 + C * w ** 3 / 3.0))


_register(Family(
    "quadratic_ramp", ("low", "high"), _ramp_logs, _ramp_h, _validate_ramp,
    mean=lambda low, high: low + (high - low) / 2.0,
    shape=lambda low, high: "piecewise"))
_register(Family(
    "triangular", ("low", "mode", "high"), _tri_logs, _tri_h, _validate_tri,
    mean=lambda a, c, b: (a + b + c) / 3.0,
    shape=lambda a, c, b: "piecewise"))
_register(Family(
    "piecewise_linear", ("knots",), _pl_logs, _pl_h, _validate_knots,
    mean=_pl_mean, shape=lambda *p: "piecewise", variadic=True))
_register(Family(
    "piecewise_quadratic", ("knots",), _pq_logs, _pq_h, _validate_knots,
    mean=_pq_mean, shape=lambda *p: "piecewise", variadic=True))


def get_family(name: str) -> Family:
    try:
        return FAMILIES[name]
    except KeyError:
        raise KeyError(f"Unknown family {name!r}; known: "
                       f"{sorted(FAMILIES)}") from None


def validate_params(name: str, params) -> None:
    """Raise ValueError for parameters outside the family's domain."""
    fam = get_family(name)
    if not fam.variadic and len(params) != len(fam.params):
        raise ValueError(f"{name} takes {fam.params}, got {len(params)} "
                         f"parameter(s).")
    fam.validate(*[float(p) for p in params])


# ── Evaluation ───────────────────────────────────────────────────────

def survival(name: str, params, z) -> np.ndarray:
    """f(z) = Pr(X >= z), float64."""
    fam = get_family(name)
    validate_params(name, params)
    return np.exp(fam.log_survival(z, *[float(p) for p in params]))


def cumulative_hazard(name: str, params, z) -> np.ndarray:
    """Lambda(z) = -log f(z)."""
    fam = get_family(name)
    validate_params(name, params)
    return -fam.log_survival(z, *[float(p) for p in params])


def hazard(name: str, params, z) -> np.ndarray:
    """h(z) = -d log f / dz."""
    fam = get_family(name)
    validate_params(name, params)
    return fam.hazard(z, *[float(p) for p in params])


def elasticity(name: str, params, z) -> np.ndarray:
    """eta(z) = -z h(z): the % loss in acceptance per % impedance."""
    return -_z(z) * hazard(name, params, z)


def mean_threshold(name: str, params) -> float:
    """E[X], the mean acceptable impedance (integral of f; inf when the
    tail is too heavy)."""
    fam = get_family(name)
    validate_params(name, params)
    if fam.mean is None:
        return _mean_numeric(fam.log_survival, *[float(p) for p in params])
    return float(fam.mean(*[float(p) for p in params]))


def hazard_shape(name: str, params) -> str:
    """'constant', 'increasing', 'decreasing', 'unimodal' or
    'degenerate': the hazard class of the family at these parameters."""
    fam = get_family(name)
    validate_params(name, params)
    return fam.shape(*[float(p) for p in params])


# ── Diagnostics of a margin ──────────────────────────────────────────

def implied_vot(time_family: tuple, cost_family: tuple, t, c) -> np.ndarray:
    """Local value of time of a separable surface f = S_T(t) S_M(c):
    the marginal rate of substitution along an iso-f contour,
    VOT = -dc/dt = h_T(t) / h_M(c) (money per unit time). Constant only
    when both hazards are constant (exponential); for power marginals it
    is (delta/gamma) c/t."""
    ht = hazard(time_family[0], time_family[1], t)
    hm = hazard(cost_family[0], cost_family[1], c)
    with np.errstate(divide="ignore", invalid="ignore"):
        return ht / hm


def ttt_transform(name: str, params, n: int = 400, upper: float | None = None):
    """Scaled total-time-on-test transform of the threshold distribution,
    phi(u) = (1/mu) * integral_0^{F^-1(u)} f(x) dx for u = F in (0, 1).
    Exponential is the diagonal phi(u) = u, increasing hazard (Weibull
    k > 1) is concave, decreasing hazard (Lomax) convex, the isochrone
    phi = 1. Needs a finite mean. Returns (u, phi)."""
    mu = mean_threshold(name, params)
    if not np.isfinite(mu):
        raise ValueError("The scaled TTT transform needs a finite mean "
                         "(the family's tail is too heavy at these "
                         "parameters).")
    if upper is None:
        upper = mu
        while (survival(name, params, np.array([upper]))[0] > 1e-6
               and upper < 1e9):
            upper *= 2.0
    # geometric grid: the tail of a heavy-tailed family matters and a
    # uniform grid over a huge range would integrate it badly
    z = np.concatenate([[0.0], np.geomspace(mu * 1e-5, upper, 20 * n)])
    f = survival(name, params, z)
    cum = integrate.cumulative_trapezoid(f, z, initial=0.0)
    u = 1.0 - f
    idx = np.unique(np.linspace(0, len(z) - 1, n).astype(int))
    return u[idx], cum[idx] / mu


def moment_matched_scale(name: str, shape_params, target_mean: float,
                         *, scale_index: int = -1) -> tuple:
    """Parameters with the given mean by rescaling one parameter (the
    last by default): families become comparable on their first moment,
    so that differences are dispersion alone.
    Mean is proportional to the scale for the scale families; for tanner
    or pareto pass the appropriate `scale_index`."""
    if target_mean <= 0:
        raise ValueError("target_mean must be positive.")
    params = [float(p) for p in shape_params]
    if name == "lognormal":
        # scale is exp(mu); the mean is exp(mu + sigma^2 / 2)
        validate_params(name, (0.0, params[1]))
        return (math.log(target_mean) - params[1] ** 2 / 2.0, params[1])
    trial = list(params)
    trial[scale_index] = 1.0
    m1 = mean_threshold(name, tuple(trial))
    if not np.isfinite(m1) or m1 <= 0:
        raise ValueError("Cannot moment-match: the mean is infinite for "
                         "these shape parameters.")
    params[scale_index] = target_mean / m1
    return tuple(params)
