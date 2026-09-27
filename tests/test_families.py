"""Survival families, their hazard tools, and the curve integration."""


import numpy as np
import pytest
from scipy import integrate

from ikob2.core import families as fam
from ikob2.core.decay_curves import (
    apply_decay,
    exponential,
    get_decay_function,
    weibull,
)
from ikob2.core.numerics import DTYPE, ensure_dense
from ikob2.domain.filter_config import (
    CurveSpec,
    FilterConfigError,
)
from ikob2.engine.runner import evaluate_marginal

CASES = {
    "exponential": (1 / 30,),
    "weibull": (2.5, 34.0),
    "lomax": (2.2, 36.0),
    "pareto": (1.5, 10.0),
    "tanner": (0.8, 0.02, 15.0),
    "gamma": (3.0, 10.0),
    "lognormal": (3.3, 0.6),
    "loglogistic": (2.5, 30.0),
    "step": (30.0,),
}
GRID = np.linspace(0.0, 400.0, 4001)


@pytest.mark.parametrize("name,params", CASES.items())
def test_every_family_is_a_survival_function(name, params):
    s = fam.survival(name, params, GRID)
    assert s[0] == pytest.approx(1.0)
    assert np.all(s >= 0) and np.all(s <= 1 + 1e-12)
    assert np.all(np.diff(s) <= 1e-12)
    assert s[-1] < 0.05 or name in ("lomax", "loglogistic", "pareto")
    assert np.all(np.isfinite(fam.cumulative_hazard(name, params, GRID[:200]))
                  ) or name == "step"


@pytest.mark.parametrize("name,params", [c for c in CASES.items()
                                         if c[0] != "step"])
def test_hazard_matches_numeric_log_derivative(name, params):
    z = np.linspace(2.0, 120.0, 400)
    lam = fam.cumulative_hazard(name, params, z)
    numeric = np.gradient(lam, z)
    h = fam.hazard(name, params, z)
    keep = (h > 1e-6) & np.isfinite(h)
    np.testing.assert_allclose(h[keep][2:-2], numeric[keep][2:-2],
                               rtol=2e-2, atol=1e-5)


@pytest.mark.parametrize("name,params", [c for c in CASES.items()
                                         if c[0] != "step"])
def test_analytic_mean_matches_integral_of_survival(name, params):
    mean = fam.mean_threshold(name, params)
    val, _ = integrate.quad(lambda x: float(fam.survival(name, params,
                                                         np.array([x]))[0]),
                            0, np.inf, limit=300)
    assert mean == pytest.approx(val, rel=2e-3)


def test_step_is_the_isochrone():
    s = fam.survival("step", (30.0,), np.array([0, 29.9, 30.0, 30.1]))
    np.testing.assert_array_equal(s, [1, 1, 1, 0])
    assert fam.mean_threshold("step", (30.0,)) == 30.0


# ── nesting relations ────────────────────────────────────────────────

def test_family_nesting():
    z = np.linspace(0, 200, 300)
    np.testing.assert_allclose(fam.survival("weibull", (1.0, 30.0), z),
                               fam.survival("exponential", (1 / 30,), z))
    np.testing.assert_allclose(fam.survival("gamma", (1.0, 30.0), z),
                               fam.survival("exponential", (1 / 30,), z),
                               rtol=1e-9)
    np.testing.assert_allclose(fam.survival("tanner", (0.0, 0.05, 10.0), z),
                               fam.survival("exponential", (0.05,), z))
    np.testing.assert_allclose(fam.survival("tanner", (1.7, 0.0, 12.0), z),
                               fam.survival("lomax", (1.7, 12.0), z))
    # pareto and lomax share the tail exponent
    far = np.array([1e5, 2e5])
    e = np.diff(np.log(fam.survival("pareto", (1.5, 10.0), far))) \
        / np.diff(np.log(far))
    assert e[0] == pytest.approx(-1.5)


# ── hazard shapes (paper Table 1 and S2) ─────────────────────────────

def test_hazard_shapes():
    z = np.linspace(1, 200, 400)
    assert fam.hazard_shape("exponential", (0.1,)) == "constant"
    h = fam.hazard("weibull", (2.5, 34.0), z)
    assert fam.hazard_shape("weibull", (2.5, 34.0)) == "increasing"
    assert np.all(np.diff(h) > 0)
    assert fam.hazard_shape("weibull", (0.7, 34.0)) == "decreasing"
    assert fam.hazard_shape("lomax", (2.2, 36.0)) == "decreasing"
    assert np.all(np.diff(fam.hazard("lomax", (2.2, 36.0), z)) < 0)
    # gamma with shape > 1: rising, converging to the rate 1/scale
    hg = fam.hazard("gamma", (3.0, 10.0), np.array([1, 10, 50, 300.0, 1e4]))
    assert np.all(np.diff(hg) > 0) and hg[-1] == pytest.approx(0.1, rel=1e-2)
    # lognormal and log-logistic (k > 1): unimodal, tending to 0
    for name, p in (("lognormal", (3.3, 0.6)), ("loglogistic", (2.5, 30.0))):
        h = fam.hazard(name, p, np.linspace(1, 2000, 4000))
        peak = int(np.argmax(h))
        assert 0 < peak < len(h) - 1 and h[-1] < h[peak] / 3
        assert fam.hazard_shape(name, p) == "unimodal"
    # tanner: decreasing towards chi
    ht = fam.hazard("tanner", (0.8, 0.02, 15.0), np.array([1, 100, 1e6]))
    assert ht[0] > ht[1] > ht[2] and ht[2] == pytest.approx(0.02, rel=1e-3)


def test_power_law_elasticity_is_constant_beyond_threshold():
    z = np.array([20.0, 50.0, 100.0])
    eta = fam.elasticity("pareto", (1.5, 10.0), z)
    np.testing.assert_allclose(eta, -1.5)
    np.testing.assert_allclose(
        fam.elasticity("exponential", (0.1,), z), -0.1 * z)


# ── implied VOT (Fixed-VOT Trap) ─────────────────────────────────────

def test_implied_vot_constant_only_for_exponential_margins():
    t = np.array([10.0, 30.0, 60.0])
    c = np.array([2.0, 6.0, 12.0])
    vot = fam.implied_vot(("exponential", (0.05,)),
                          ("exponential", (0.5,)), t, c)
    np.testing.assert_allclose(vot, 0.05 / 0.5)         # h_T / h_M constant


def test_implied_vot_of_power_marginals_is_delta_over_gamma_c_over_t():
    t = np.array([20.0, 40.0, 80.0])
    c = np.array([4.0, 4.0, 32.0])
    delta, gamma = 1.8, 0.9                 # hazards delta/t and gamma/c
    vot = fam.implied_vot(("pareto", (delta, 5.0)),
                          ("pareto", (gamma, 1.0)), t, c)
    np.testing.assert_allclose(vot, (delta / gamma) * c / t)
    # varies across the surface, unlike exponential
    assert vot.max() / vot.min() > 2


def test_implied_vot_varies_for_weibull_margins():
    t = np.array([10.0, 40.0])
    c = np.array([5.0, 5.0])
    vot = fam.implied_vot(("weibull", (2.5, 34.0)),
                          ("weibull", (1.5, 8.0)), t, c)
    assert vot[1] > vot[0]


# ── TTT transform ────────────────────────────────────────────────────

def test_ttt_shapes():
    u, phi = fam.ttt_transform("exponential", (0.05,))
    np.testing.assert_allclose(phi, u, atol=2e-3)         # the diagonal
    u, phi = fam.ttt_transform("weibull", (2.5, 34.0))
    assert np.all(phi[1:-1] >= u[1:-1] - 1e-9)             # concave (IFR)
    u, phi = fam.ttt_transform("lomax", (2.2, 36.0))
    assert np.all(phi[1:-1] <= u[1:-1] + 1e-9)             # convex (DFR)
    with pytest.raises(ValueError, match="finite mean"):
        fam.ttt_transform("lomax", (0.8, 36.0))


def test_moment_matching_gives_common_mean():
    for name, shape in (("weibull", (2.5, 1.0)), ("lomax", (2.2, 1.0)),
                        ("gamma", (3.0, 1.0)), ("lognormal", (1.0, 0.6)),
                        ("loglogistic", (2.5, 1.0))):
        p = fam.moment_matched_scale(name, shape, 30.0)
        assert fam.mean_threshold(name, p) == pytest.approx(30.0, rel=1e-6)
    with pytest.raises(ValueError, match="infinite"):
        fam.moment_matched_scale("lomax", (0.9, 1.0), 30.0)


# ── validation ───────────────────────────────────────────────────────

@pytest.mark.parametrize("name,params", [
    ("weibull", (0.0, 10.0)), ("weibull", (2.0, -1.0)), ("lomax", (1.0, 0.0)),
    ("pareto", (-1.0, 5.0)), ("tanner", (0.0, 0.0, 5.0)),
    ("tanner", (-0.1, 0.1, 5.0)), ("gamma", (0.0, 1.0)),
    ("lognormal", (1.0, 0.0)), ("lognormal", (np.nan, 1.0)),
    ("loglogistic", (2.0, np.inf)), ("step", (-1.0,)),
    ("weibull", (2.0,)),
])
def test_invalid_parameters_are_rejected(name, params):
    with pytest.raises(ValueError):
        fam.validate_params(name, params)
    with pytest.raises(KeyError, match="Unknown family"):
        fam.get_family("nope")


# ── integration with curves, config and runner ───────────────────────

@pytest.mark.parametrize("name,params", [c for c in CASES.items()
                                         if c[0] not in ("exponential",)])
def test_decay_curves_match_the_family_and_stay_probabilities(name, params):
    z = np.linspace(0, 300, 500).astype(DTYPE)
    got = get_decay_function(name)(z, *params)
    assert got.dtype == DTYPE
    np.testing.assert_allclose(got, fam.survival(name, params, z),
                               rtol=1e-5, atol=1e-7)
    assert got.min() >= 0 and got.max() <= 1


def test_weibull_and_exponential_curves_agree_with_families():
    z = np.linspace(0, 200, 300).astype(DTYPE)
    np.testing.assert_allclose(
        weibull(z, 2.5, 34.0), fam.survival("weibull", (2.5, 34.0), z),
        rtol=1e-5, atol=1e-7)
    np.testing.assert_allclose(
        exponential(z, 0.03), fam.survival("exponential", (0.03,), z),
        rtol=1e-5)


def test_negative_costs_are_treated_as_zero_and_apply_decay_works():
    z = np.array([[-5.0, 0.0], [10.0, 500.0]], dtype=DTYPE)
    out = ensure_dense(apply_decay(z, "gamma", (3.0, 10.0), epsilon=None))
    assert out[0, 0] == 1.0 and out[0, 1] == 1.0
    assert 0 < out[1, 0] < 1 and out[1, 1] < 1e-6


def test_evaluate_marginal_applies_atom_to_new_families():
    c = np.array([[0.0, 30.0]], dtype=DTYPE)
    out = evaluate_marginal(c, CurveSpec("lognormal", (3.3, 0.6), atom=0.2))
    base = fam.survival("lognormal", (3.3, 0.6), np.array([30.0]))[0]
    assert out[0, 0] == 1.0
    assert out[0, 1] == pytest.approx(0.8 * base, rel=1e-5)


def _curve(block):
    """Parse a curve block as a run configuration would."""
    return CurveSpec.from_dict(block, "time")


def test_curve_blocks_accept_all_families_and_validate_parameters():
    blocks = [
        {"curve": "lomax", "alpha": 2.2, "scale": 36},
        {"curve": "pareto", "alpha": 1.5, "z0": 10},
        {"curve": "tanner", "rho": 0.8, "chi": 0.02, "scale": 15},
        {"curve": "gamma", "shape": 3, "scale": 10},
        {"curve": "lognormal", "mu": 3.3, "sigma": 0.6},
        {"curve": "loglogistic", "shape": 2.5, "scale": 30},
        {"curve": "step", "threshold": 30},
    ]
    for b in blocks:
        cf = _curve(b)
        assert cf.curve == b["curve"]
    with pytest.raises(FilterConfigError, match="shape must be positive"):
        _curve({"curve": "gamma", "shape": -1, "scale": 10})
    with pytest.raises(FilterConfigError, match="missing parameter"):
        _curve({"curve": "tanner", "rho": 1})
