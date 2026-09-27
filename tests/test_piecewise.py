"""Piecewise-linear and piecewise-quadratic decay, and the quadratic ramp."""


import numpy as np
import pytest
from scipy import integrate

from ikob2.core import families as fam
from ikob2.core.decay_curves import apply_decay, get_decay_function, uniform
from ikob2.core.numerics import DTYPE, ensure_dense
from ikob2.domain.filter_config import (
    CurveSpec,
    FilterConfigError,
)
from ikob2.engine.runner import evaluate_marginal

# survival function of stated acceptable times at 15-minute bin edges
KNOTS = (0, 1, 15, 0.92, 30, 0.5, 45, 0.15, 60, 0.02, 90, 0.0)
GRID = np.linspace(0.0, 120.0, 12001)


# ── piecewise linear ─────────────────────────────────────────────────

def test_piecewise_linear_interpolates_and_is_zero_beyond_the_last_knot():
    z = np.array([0, 7.5, 15, 22.5, 30, 60, 75, 90, 90.1, 200.0])
    got = fam.survival("piecewise_linear", KNOTS, z)
    np.testing.assert_allclose(
        got, [1, 0.96, 0.92, 0.71, 0.5, 0.02, 0.01, 0.0, 0.0, 0.0],
        atol=1e-12)


def test_last_knot_above_zero_is_a_jump():
    kn = (0, 1, 30, 0.4)
    assert fam.survival("piecewise_linear", kn, np.array([30.0]))[0] == 0.4
    assert fam.survival("piecewise_linear", kn, np.array([30.001]))[0] == 0.0
    assert fam.hazard("piecewise_linear", kn, np.array([31.0]))[0] == np.inf


def test_two_knot_piecewise_linear_is_the_uniform_curve():
    z = np.linspace(0, 40, 200)
    pl = fam.survival("piecewise_linear", (0, 1, 5, 1, 15, 0), z)
    np.testing.assert_allclose(pl, uniform(z.astype(DTYPE), 5.0, 15.0),
                               atol=1e-6)


def test_piecewise_linear_hazard_and_mean():
    z = np.array([5.0, 20.0, 40.0])
    f = fam.survival("piecewise_linear", KNOTS, z)
    slope = np.array([(0.92 - 1) / 15, (0.5 - 0.92) / 15, (0.15 - 0.5) / 15])
    np.testing.assert_allclose(fam.hazard("piecewise_linear", KNOTS, z),
                               -slope / f)
    area = (15 * (1 + .92) + 15 * (.92 + .5) + 15 * (.5 + .15)
            + 15 * (.15 + .02) + 30 * (.02 + 0)) / 2
    assert fam.mean_threshold("piecewise_linear", KNOTS) == pytest.approx(area)
    val, _ = integrate.quad(lambda x: float(fam.survival(
        "piecewise_linear", KNOTS, np.array([x]))[0]), 0, 90, limit=400)
    assert val == pytest.approx(area, rel=1e-6)


@pytest.mark.parametrize("bad,match", [
    ((0, 1), "at least two"),
    ((0, 1, 10, 0.5, 20), "even number"),
    ((1, 1, 10, 0.5), "first knot"),
    ((0, 0.9, 10, 0.5), "first knot"),
    ((0, 1, 10, 0.5, 10, 0.2), "strictly increasing"),
    ((0, 1, 10, 0.5, 5, 0.2), "strictly increasing"),
    ((0, 1, 10, 0.5, 20, 0.7), "non-increasing"),
    ((0, 1, 10, 1.2), "lie in"),
    ((0, 1, 10, -0.1), "lie in"),
    ((0, 1, np.nan, 0.5), "finite"),
])
@pytest.mark.parametrize("name", ["piecewise_linear", "piecewise_quadratic"])
def test_knot_validation(name, bad, match):
    with pytest.raises(ValueError, match=match):
        fam.validate_params(name, bad)


# ── piecewise quadratic ──────────────────────────────────────────────

KNOT_SETS = [
    KNOTS,
    (0, 1, 10, 1, 20, 0.6, 30, 0.6, 50, 0.1, 80, 0.0),   # flat segments
    (0, 1, 5, 0.2, 60, 0.15, 61, 0.0),                    # steep then long
    (0, 1, 30, 0.4),                                      # jump at the end
    (0, 1, 60, 0),                                        # two knots
    (0, 1, 1, 0.99, 2, 0.5, 3, 0.01, 4, 0),               # sharp bend
]


@pytest.mark.parametrize("kn", KNOT_SETS)
def test_quadratic_spline_passes_through_the_knots(kn):
    x = np.array(kn[0::2], dtype=float)
    y = np.array(kn[1::2], dtype=float)
    got = fam.survival("piecewise_quadratic", kn, x)
    np.testing.assert_allclose(got, y, atol=1e-12)


@pytest.mark.parametrize("kn", KNOT_SETS)
def test_quadratic_spline_is_a_monotone_survival_function(kn):
    f = fam.survival("piecewise_quadratic", kn, GRID)
    assert f[0] == 1.0
    assert f.min() >= 0.0 and f.max() <= 1.0
    assert np.all(np.diff(f) <= 1e-12)
    assert f[-1] == 0.0


@pytest.mark.parametrize("kn", KNOT_SETS)
def test_quadratic_spline_is_c1(kn):
    starts, ends, A, B, C, last = fam._qspline(kn)
    eps = 1e-9
    for j in range(1, len(starts)):
        left = B[j - 1] + 2 * C[j - 1] * (starts[j] - starts[j - 1])
        assert left == pytest.approx(B[j], abs=1e-7)
        vleft = (A[j - 1] + B[j - 1] * (starts[j] - starts[j - 1])
                 + C[j - 1] * (starts[j] - starts[j - 1]) ** 2)
        assert vleft == pytest.approx(A[j], abs=1e-9)
    assert eps > 0


def test_flat_segments_stay_flat():
    kn = KNOT_SETS[1]       # flat on [0, 10] and [20, 30], falling between
    z2 = np.linspace(20, 30, 50)
    np.testing.assert_allclose(
        fam.survival("piecewise_quadratic", kn, z2), 0.6)
    z1 = np.linspace(0, 10, 50)
    np.testing.assert_allclose(
        fam.survival("piecewise_quadratic", kn, z1), 1.0)


def test_two_knots_give_the_straight_line():
    z = np.linspace(0, 70, 300)
    np.testing.assert_allclose(
        fam.survival("piecewise_quadratic", (0, 1, 60, 0), z),
        fam.survival("piecewise_linear", (0, 1, 60, 0), z), atol=1e-12)


def test_quadratic_is_closer_than_linear_to_a_smooth_curve():
    truth = lambda z: fam.survival("weibull", (2.5, 34.0), z)   # noqa: E731
    xs = np.arange(0, 111, 10.0)
    kn = tuple(v for x, y in zip(xs, truth(xs)) for v in (x, float(y)))
    z = np.linspace(0, 110, 2000)
    err_q = np.abs(fam.survival("piecewise_quadratic", kn, z) - truth(z)).max()
    err_l = np.abs(fam.survival("piecewise_linear", kn, z) - truth(z)).max()
    assert err_q < err_l and err_q < 0.02


def test_quadratic_mean_matches_numeric_integral_and_hazard_is_continuous():
    val, _ = integrate.quad(lambda x: float(fam.survival(
        "piecewise_quadratic", KNOTS, np.array([x]))[0]), 0, 90, limit=400)
    assert fam.mean_threshold("piecewise_quadratic", KNOTS) == pytest.approx(
        val, rel=1e-6)
    z = np.linspace(0.5, 80, 4000)
    h = fam.hazard("piecewise_quadratic", KNOTS, z)
    assert np.all(np.isfinite(h)) and np.all(h >= 0)
    assert np.abs(np.diff(h)).max() < 0.25        # no jumps between pieces
    # piecewise LINEAR hazard, by contrast, jumps at the knots
    hl = fam.hazard("piecewise_linear", KNOTS, z)
    assert np.abs(np.diff(hl)).max() > 0.02


# ── quadratic ramp ───────────────────────────────────────────────────

def test_quadratic_ramp_values_and_smoothness():
    z = np.array([0, 10, 20, 30, 40, 50, 60.0])
    np.testing.assert_allclose(
        fam.survival("quadratic_ramp", (10, 50), z),
        [1, 1, 0.875, 0.5, 0.125, 0, 0], atol=1e-12)
    # C1 at the midpoint: equal slopes from both sides
    eps = 1e-6
    a = fam.survival("quadratic_ramp", (10, 50), np.array([30 - eps, 30]))
    b = fam.survival("quadratic_ramp", (10, 50), np.array([30, 30 + eps]))
    assert np.diff(a)[0] / eps == pytest.approx(np.diff(b)[0] / eps, rel=1e-4)
    assert fam.mean_threshold("quadratic_ramp", (10, 50)) == pytest.approx(30.0)
    val, _ = integrate.quad(lambda x: float(fam.survival(
        "quadratic_ramp", (10, 50), np.array([x]))[0]), 0, 50, limit=200)
    assert val == pytest.approx(30.0, rel=1e-6)


def test_quadratic_ramp_hazard_and_validation():
    z = np.array([5.0, 20.0, 40.0, 55.0])
    h = fam.hazard("quadratic_ramp", (10, 50), z)
    assert h[0] == 0 and 0 < h[1] < h[2] < np.inf and h[3] == np.inf
    for bad in ((-1, 10), (10, 10), (20, 10), (0, np.inf)):
        with pytest.raises(ValueError):
            fam.validate_params("quadratic_ramp", bad)


# ── curves, config, runner ───────────────────────────────────────────

@pytest.mark.parametrize("name,params", [
    ("piecewise_linear", KNOTS), ("piecewise_quadratic", KNOTS),
    ("quadratic_ramp", (10.0, 50.0))])
def test_decay_curves_are_float32_probability_marginals(name, params):
    z = np.linspace(-5, 150, 400).astype(DTYPE)
    got = get_decay_function(name)(z, *params)
    assert got.dtype == DTYPE and got.min() >= 0 and got.max() <= 1
    np.testing.assert_allclose(got, fam.survival(name, params, z),
                               rtol=1e-5, atol=1e-7)
    out = ensure_dense(apply_decay(z.reshape(1, -1), name, params,
                                   epsilon=None))
    np.testing.assert_allclose(out.ravel(), got)


def _curve(block):
    """Parse a curve block as a run configuration would."""
    return CurveSpec.from_dict(block, "time")


def test_curve_blocks_knots_and_ramp():
    knots = [[0, 1], [15, 0.92], [30, 0.5], [60, 0.02], [90, 0]]
    for curve in ("piecewise_linear", "piecewise_quadratic"):
        cf = _curve({"curve": curve, "knots": knots, "atom": 0.1}
        )
        assert cf.curve == curve and cf.atom == 0.1
        assert cf.params == (0, 1, 15, 0.92, 30, 0.5, 60, 0.02, 90, 0)
    cf = _curve({"curve": "quadratic_ramp", "low": 10, "high": 50}
    )
    assert cf.params == (10.0, 50.0)


def test_curve_block_knot_errors():
    with pytest.raises(FilterConfigError, match="needs 'knots'"):
        _curve({"curve": "piecewise_linear"})
    with pytest.raises(FilterConfigError, match="pairs"):
        _curve({"curve": "piecewise_linear", "knots": [1, 2, 3]})
    with pytest.raises(FilterConfigError, match="non-increasing"):
        _curve({"curve": "piecewise_quadratic",
                       "knots": [[0, 1], [10, 0.2], [20, 0.5]]})
    with pytest.raises(FilterConfigError, match="Unknown key"):
        _curve({"curve": "piecewise_linear", "low": 1,
                       "knots": [[0, 1], [10, 0]]})
    with pytest.raises(FilterConfigError, match="low < high"):
        _curve({"curve": "quadratic_ramp", "low": 20, "high": 10})


def test_evaluate_marginal_with_knots_and_atom():
    c = np.array([[0.0, 30.0, 100.0]], dtype=DTYPE)
    out = evaluate_marginal(c, CurveSpec("piecewise_quadratic", KNOTS,
                                         atom=0.2))
    assert out[0, 0] == 1.0
    assert out[0, 1] == pytest.approx(0.8 * 0.5, rel=1e-5)
    assert out[0, 2] == 0.0


# ── triangular with a free mode ──────────────────────────────────────

def test_triangular_survival_values():
    # low 10, mode 20, high 50: F(20) = 10/40 = 0.25
    z = np.array([0, 10, 15, 20, 35, 50, 60.0])
    f = fam.survival("triangular", (10, 20, 50), z)
    np.testing.assert_allclose(
        f, [1, 1, 1 - 25 / 400, 0.75, 1 - (1 - 15 ** 2 / (40 * 30)), 0, 0],
        atol=1e-12)
    assert fam.survival("triangular", (10, 20, 50), np.array([35.0]))[0] \
        == pytest.approx(15 ** 2 / (40 * 30))


def test_triangular_with_midpoint_mode_is_the_quadratic_ramp():
    z = np.linspace(0, 70, 500)
    np.testing.assert_allclose(
        fam.survival("triangular", (10, 30, 50), z),
        fam.survival("quadratic_ramp", (10, 50), z), atol=1e-12)
    h1 = fam.hazard("triangular", (10, 30, 50), z[(z > 10) & (z < 50)])
    h2 = fam.hazard("quadratic_ramp", (10, 50), z[(z > 10) & (z < 50)])
    np.testing.assert_allclose(h1, h2, rtol=1e-9)


@pytest.mark.parametrize("params", [(10, 20, 50), (10, 10, 50), (10, 50, 50),
                                    (0, 5, 60), (10, 30, 50)])
def test_triangular_is_a_survival_function_with_the_right_mean(params):
    z = np.linspace(0, 80, 8001)
    f = fam.survival("triangular", params, z)
    assert f[0] == 1.0 and f.min() >= 0 and f.max() <= 1
    assert np.all(np.diff(f) <= 1e-12) and f[-1] == 0.0
    a, c, b = params
    mean = fam.mean_threshold("triangular", params)
    assert mean == pytest.approx((a + b + c) / 3)
    val, _ = integrate.quad(lambda x: float(fam.survival(
        "triangular", params, np.array([x]))[0]), 0, b, limit=400)
    assert val == pytest.approx(mean, rel=1e-6)


def test_triangular_mode_moves_the_weight_and_hazard_is_pdf_over_survival():
    low_mode = fam.survival("triangular", (10, 12, 50), np.array([30.0]))[0]
    high_mode = fam.survival("triangular", (10, 48, 50), np.array([30.0]))[0]
    assert low_mode < high_mode                # more mass at low thresholds
    z = np.array([15.0, 25.0, 40.0])
    a, c, b = 10.0, 25.0, 50.0
    pdf = np.array([2 * (15 - a) / ((b - a) * (c - a)),
                    2 * (25 - a) / ((b - a) * (c - a)),
                    2 * (b - 40) / ((b - a) * (b - c))])
    f = fam.survival("triangular", (a, c, b), z)
    np.testing.assert_allclose(fam.hazard("triangular", (a, c, b), z), pdf / f)
    assert fam.hazard("triangular", (a, c, b), np.array([5.0]))[0] == 0
    assert fam.hazard("triangular", (a, c, b), np.array([55.0]))[0] == np.inf


def test_right_angled_triangular_edge_modes():
    # mode = low: density falls linearly from low to high
    z = np.array([10.0, 30.0, 50.0])
    f = fam.survival("triangular", (10, 10, 50), z)
    # S(z) = ((high - z) / (high - low))^2
    np.testing.assert_allclose(f, [1, 0.25, 0], atol=1e-12)
    # mode = high: S(z) = 1 - ((z-low)/(high-low))^2
    g = fam.survival("triangular", (10, 50, 50), z)
    np.testing.assert_allclose(g, [1, 0.75, 0], atol=1e-12)


@pytest.mark.parametrize("bad", [(20, 10, 50), (10, 60, 50), (10, 10, 10),
                                 (-1, 5, 50), (10, np.nan, 50)])
def test_triangular_validation(bad):
    with pytest.raises(ValueError):
        fam.validate_params("triangular", bad)


def test_triangular_through_curves_and_blocks():
    z = np.linspace(0, 70, 300).astype(DTYPE)
    got = get_decay_function("triangular")(z, 10.0, 20.0, 50.0)
    assert got.dtype == DTYPE
    np.testing.assert_allclose(got, fam.survival("triangular",
                                                 (10, 20, 50), z),
                               rtol=1e-5, atol=1e-7)
    cf = _curve({"curve": "triangular", "low": 10, "mode": 20, "high": 50,
                   "atom": 0.1})
    assert cf.params == (10.0, 20.0, 50.0) and cf.atom == 0.1
    with pytest.raises(FilterConfigError, match="low <= mode <= high"):
        _curve({"curve": "triangular", "low": 10, "mode": 60,
                       "high": 50})
    with pytest.raises(FilterConfigError, match="missing parameter"):
        _curve({"curve": "triangular", "low": 10, "high": 50})
