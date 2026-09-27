"""Weibull margin and Gumbel-Hougaard copula (threshold-gate model)."""


import numpy as np
import pytest

from ikob2.core.compose import (
    comonotone,
    compose_filters,
    frechet_violation,
    gumbel_hougaard,
)
from ikob2.core.decay_curves import apply_decay, exponential, weibull
from ikob2.core.numerics import DTYPE, ensure_dense
from ikob2.domain.filter_config import (
    CopulaSpec,
    FilterConfigError,
    CurveSpec,
)


# ── Weibull ──────────────────────────────────────────────────────────

def test_weibull_matches_closed_form():
    t = np.array([0.0, 10.0, 30.0, 60.0, 120.0], dtype=DTYPE)
    k, eta = 2.5, 40.0
    np.testing.assert_allclose(
        weibull(t, k, eta), np.exp(-(t / eta) ** k), rtol=1e-5)


def test_weibull_is_a_survival_function():
    t = np.linspace(0, 300, 500, dtype=DTYPE)
    s = weibull(t, 2.0, 45.0)
    assert s[0] == 1.0
    assert np.all(np.diff(s) <= 0)
    assert s.min() >= 0.0 and s.max() <= 1.0
    assert s[-1] < 1e-6


def test_weibull_shape_one_is_exponential():
    t = np.linspace(0, 100, 50, dtype=DTYPE)
    eta = 30.0
    np.testing.assert_allclose(
        weibull(t, 1.0, eta), exponential(t, 1.0 / eta), rtol=1e-5)


def test_weibull_median_is_scale_times_ln2_root():
    k, eta = 3.0, 50.0
    median = eta * np.log(2.0) ** (1.0 / k)
    assert float(weibull(np.array([median]), k, eta)[0]) == pytest.approx(0.5, abs=1e-6)


def test_weibull_extreme_inputs_do_not_overflow():
    t = np.array([0.0, 1e6, 1e12], dtype=DTYPE)
    with np.errstate(all="raise"):
        s = weibull(t, 8.0, 5.0)
    assert np.all(np.isfinite(s))
    assert s[0] == 1.0 and s[1] < 1e-30 and s[2] < 1e-30


@pytest.mark.parametrize("shape,scale", [(0, 10), (-1, 10), (2, 0), (2, -5),
                                          (np.nan, 10), (2, np.inf)])
def test_weibull_rejects_invalid_params(shape, scale):
    with pytest.raises(ValueError):
        weibull(np.ones(3, dtype=DTYPE), shape, scale)


def test_weibull_through_apply_decay():
    cost = np.array([[0.0, 30.0], [30.0, 0.0]], dtype=DTYPE)
    got = ensure_dense(apply_decay(cost, "weibull", (2.0, 30.0), epsilon=None))
    np.testing.assert_allclose(got, [[1, np.exp(-1)], [np.exp(-1), 1]], rtol=1e-5)


# ── Gumbel-Hougaard ──────────────────────────────────────────────────

def _grid(n=41):
    g = np.linspace(0.0, 1.0, n, dtype=DTYPE)
    u, v = np.meshgrid(g, g)
    return u, v


def test_gumbel_matches_paper_formula():
    u = np.array([0.8, 0.5, 0.3], dtype=DTYPE)
    v = np.array([0.6, 0.9, 0.2], dtype=DTYPE)
    for theta in (1.25, 1.5, 2.0, 4.0):
        lt, lm = -np.log(u.astype(float)), -np.log(v.astype(float))
        expected = np.exp(-(lt ** theta + lm ** theta) ** (1.0 / theta))
        np.testing.assert_allclose(
            gumbel_hougaard(u, v, theta), expected, rtol=1e-5)


def test_gumbel_theta_one_is_independence():
    u, v = _grid()
    np.testing.assert_allclose(gumbel_hougaard(u, v, 1.0), u * v, atol=1e-7)


def test_gumbel_large_theta_approaches_comonotone_and_is_stable():
    u, v = _grid()
    with np.errstate(all="raise"):
        big = gumbel_hougaard(u, v, 1e4)
    assert np.all(np.isfinite(big))
    np.testing.assert_allclose(big, comonotone(u, v), atol=2e-3)


def test_gumbel_boundary_conditions():
    g = np.linspace(0.0, 1.0, 11, dtype=DTYPE)
    one = np.ones_like(g)
    zero = np.zeros_like(g)
    for theta in (1.5, 4.0):
        np.testing.assert_allclose(gumbel_hougaard(g, one, theta), g, atol=1e-6)
        np.testing.assert_allclose(gumbel_hougaard(one, g, theta), g, atol=1e-6)
        np.testing.assert_array_equal(gumbel_hougaard(g, zero, theta), zero)
        np.testing.assert_array_equal(gumbel_hougaard(zero, zero, theta), zero)


def test_gumbel_is_symmetric_and_increasing_in_theta():
    u, v = _grid()
    for theta in (1.25, 2.0):
        np.testing.assert_allclose(
            gumbel_hougaard(u, v, theta), gumbel_hougaard(v, u, theta), atol=1e-7)
    # positive dependence raises the joint survival (paper Sec. 2.4)
    prev = u * v
    for theta in (1.25, 1.5, 2.0, 4.0):
        cur = gumbel_hougaard(u, v, theta)
        assert np.all(cur >= prev - 1e-6)
        prev = cur


@pytest.mark.parametrize("theta", [1.0, 1.25, 2.0, 4.0, 50.0])
def test_gumbel_respects_frechet_bounds(theta):
    u, v = _grid()
    assert frechet_violation(u, v, gumbel_hougaard(u, v, theta)) <= 5e-6


@pytest.mark.parametrize("theta", [0.99, 0.0, -1.0, np.inf, np.nan])
def test_gumbel_rejects_invalid_theta(theta):
    with pytest.raises(ValueError):
        gumbel_hougaard(np.ones(2, dtype=DTYPE), np.ones(2, dtype=DTYPE), theta)


def test_compose_filters_gumbel_and_theta_rules():
    u = np.full((3, 3), 0.7, dtype=DTYPE)
    v = np.full((3, 3), 0.4, dtype=DTYPE)
    out = ensure_dense(compose_filters(u, v, family="gumbel", theta=2.0,
                                       epsilon=None))
    np.testing.assert_allclose(out, gumbel_hougaard(u, v, 2.0), rtol=1e-6)

    with pytest.raises(ValueError, match="requires theta"):
        compose_filters(u, v, family="gumbel", epsilon=None)
    with pytest.raises(ValueError, match="takes no parameter"):
        compose_filters(u, v, family="independence", theta=2.0, epsilon=None)


# ── Config integration ───────────────────────────────────────────────

def test_blocks_parse_weibull_and_gumbel():
    time = CurveSpec.from_dict(
        {"curve": "weibull", "shape": 2.5, "scale": 40.0}, "time")
    assert time.curve == "weibull" and time.params == (2.5, 40.0)
    assert CopulaSpec.from_dict({"family": "gumbel", "theta": 1.5},
                                "copula") == CopulaSpec("gumbel", 1.5)


def test_blocks_reject_bad_weibull_and_gumbel():
    with pytest.raises(FilterConfigError, match="missing parameter"):
        CurveSpec.from_dict({"curve": "weibull", "shape": 2.0}, "time")
    with pytest.raises(FilterConfigError, match="theta >= 1"):
        CopulaSpec.from_dict({"family": "gumbel", "theta": 0.5}, "copula")
    with pytest.raises(FilterConfigError, match="requires theta"):
        CopulaSpec.from_dict({"family": "gumbel"}, "copula")


# ── Uniform (piecewise-linear) cost margin ───────────────────────────

from ikob2.core.decay_curves import uniform  # noqa: E402


def test_uniform_piecewise_linear_values():
    c = np.array([0.0, 2.0, 5.0, 10.0, 15.0, 20.0, 99.0], dtype=DTYPE)
    np.testing.assert_allclose(
        uniform(c, 5.0, 15.0), [1, 1, 1, 0.5, 0, 0, 0], atol=1e-7)


def test_uniform_is_left_continuous_at_the_bounds():
    # a trip costing exactly `low` clears the gate; exactly `high` does not
    s = uniform(np.array([5.0, 15.0], dtype=DTYPE), 5.0, 15.0)
    assert s[0] == 1.0 and s[1] == 0.0


def test_uniform_is_a_survival_function():
    c = np.linspace(0, 300, 600, dtype=DTYPE)
    s = uniform(c, 4.87, 23.52)
    assert np.all(np.diff(s) <= 0)
    assert s.min() >= 0 and s.max() <= 1
    assert uniform(np.zeros(1, dtype=DTYPE), 4.87, 23.52)[0] == 1.0


def test_uniform_free_travel_clears_even_with_zero_low():
    # envelope like single/Q2: km max motorised low = 0.00 -> low = 0
    assert uniform(np.zeros(1, dtype=DTYPE), 0.0, 25.53)[0] == 1.0


def test_uniform_degenerate_interval_is_a_step():
    c = np.array([0.0, 7.0, 7.01, 50.0], dtype=DTYPE)
    np.testing.assert_array_equal(uniform(c, 7.0, 7.0), [1, 1, 0, 0])


def test_uniform_mean_is_interval_midpoint():
    # E[X] = integral of S; midpoint for uniform (grid integral)
    c = np.linspace(0, 40, 400001)
    s = uniform(c, 10.0, 30.0).astype(np.float64)
    assert float(np.trapezoid(s, c)) == pytest.approx(20.0, rel=1e-4)


def test_uniform_fare_cut_gain_matches_threshold_crossing_identity():
    # Paper Sec. 2.3: a fare cut c -> c' changes acceptance by
    # Pr(c' <= X_M < c) = S(c') - S(c) for the cost margin alone.
    low, high, c, c2 = 5.0, 15.0, 12.0, 6.0
    gain = float(uniform(np.array([c2], dtype=DTYPE), low, high)[0]
                 - uniform(np.array([c], dtype=DTYPE), low, high)[0])
    assert gain == pytest.approx((c - c2) / (high - low), rel=1e-5)


@pytest.mark.parametrize("low,high", [(-1, 5), (5, 4), (0, np.inf), (np.nan, 3)])
def test_uniform_rejects_invalid_bounds(low, high):
    with pytest.raises(ValueError):
        uniform(np.ones(2, dtype=DTYPE), low, high)


def test_uniform_cost_margin_block_and_composition():
    cost = CurveSpec.from_dict(
        {"curve": "uniform", "low": 5.0, "high": 15.0}, "cost")
    assert cost.curve == "uniform" and cost.params == (5.0, 15.0)

    # gate: time survival x cost survival under independence
    t = np.full((2, 2), 30.0, dtype=DTYPE)
    m = np.full((2, 2), 10.0, dtype=DTYPE)
    joint = ensure_dense(compose_filters(
        weibull(t, 2.0, 45.0), uniform(m, 5.0, 15.0), epsilon=None))
    np.testing.assert_allclose(joint, np.exp(-(30 / 45) ** 2) * 0.5, rtol=1e-5)


# ── Atom at zero ─────────────────────────────────────────────────────

from ikob2.core.decay_curves import with_atom  # noqa: E402
from ikob2.domain.filter_config import INDEPENDENCE, ClassFilter  # noqa: E402
from ikob2.domain.segments import Segment  # noqa: E402
from ikob2.engine.runner import SegmentedRunner, evaluate_marginal  # noqa: E402


def test_atom_scales_positive_values_and_spares_zero():
    c = np.array([0.0, 1.0, 10.0, 100.0], dtype=DTYPE)
    base = uniform(c, 5.0, 15.0)
    out = with_atom(base, c, 0.25)
    np.testing.assert_allclose(out, [1.0, 0.75, 0.75 * 0.5, 0.0], atol=1e-7)


def test_atom_zero_is_identity_and_one_excludes_all_priced_trips():
    c = np.array([0.0, 3.0], dtype=DTYPE)
    base = uniform(c, 5.0, 15.0)
    assert with_atom(base, c, 0.0) is base
    np.testing.assert_array_equal(with_atom(base, c, 1.0), [1.0, 0.0])


def test_atom_result_is_still_a_survival_function():
    c = np.linspace(0, 100, 500, dtype=DTYPE)
    s = with_atom(weibull(c, 2.0, 40.0), c, 0.3)
    # non-increasing on the positive axis, S(0+) = 1 - atom
    pos = s[c > 0]
    assert np.all(np.diff(pos) <= 0)
    assert pos[0] == pytest.approx(0.7, abs=1e-3)


@pytest.mark.parametrize("atom", [-0.1, 1.1, np.nan])
def test_atom_rejects_out_of_range(atom):
    with pytest.raises(ValueError):
        with_atom(np.ones(2, dtype=DTYPE), np.ones(2, dtype=DTYPE), atom)
    with pytest.raises(FilterConfigError):
        CurveSpec("uniform", (1.0, 2.0), atom=atom)


def test_curvespec_atom_is_part_of_identity():
    a = CurveSpec("uniform", (5.0, 15.0))
    b = CurveSpec("uniform", (5.0, 15.0), atom=0.2)
    assert a == CurveSpec("uniform", (5.0, 15.0), atom=0.0)
    assert a != b and hash(a) != hash(b)


def test_atom_in_curve_block():
    block = {"curve": "uniform", "low": 5.0, "high": 15.0, "atom": 0.15}
    assert CurveSpec.from_dict(block, "cost").atom == 0.15
    with pytest.raises(FilterConfigError, match="atom"):
        CurveSpec.from_dict({**block, "atom": 2.0}, "cost")


def test_runner_applies_atom_to_priced_pairs_only():
    # Pairs: diagonal is free (cost 0), off-diagonal costs 8 EUR, well
    # inside the envelope's flat part -> continuous S_M = 1 there, so the
    # only thing reducing the priced weight is the atom.
    n = 3
    time = np.full((n, n), 10.0, dtype=DTYPE)
    money = np.full((n, n), 8.0, dtype=DTYPE)
    np.fill_diagonal(money, 0.0)
    pop = np.array([100.0, 200.0, 300.0], dtype=DTYPE)
    jobs = np.array([50.0, 60.0, 70.0], dtype=DTYPE)
    pi = 0.2

    from ikob2.domain.state import ModelState
    state = ModelState.create(
        generalized_cost=time, population=pop, opportunities=jobs,
        decay_type="exponential", decay_beta=0.01, decay_epsilon=0.0)

    def run(atom):
        cf = ClassFilter(
            CurveSpec("weibull", (2.0, 60.0)),
            CurveSpec("uniform", (10.0, 20.0), atom=atom),
            INDEPENDENCE)
        seg = Segment("s", "laag", cf,
                      time_cost_id="time", money_cost_id="money")
        return SegmentedRunner(decay_epsilon=None).run(
            state, [seg], {"s": pop},
            cost_matrices={"time": time, "money": money})

    r0, r1 = run(0.0), run(pi)

    # hand-built weights
    tw = np.exp(-(10.0 / 60.0) ** 2)
    for res, factor in ((r0, 1.0), (r1, 1.0 - pi)):
        D = np.full((n, n), tw * factor)
        np.fill_diagonal(D, tw)
        V = D.T @ pop
        expected = D @ (jobs / V)
        np.testing.assert_allclose(res.per_segment["s"], expected, rtol=1e-4)
    assert not np.allclose(r0.per_segment["s"], r1.per_segment["s"])


def test_evaluate_marginal_applies_atom():
    c = np.array([[0.0, 8.0], [8.0, 0.0]], dtype=DTYPE)
    out = evaluate_marginal(c, CurveSpec("uniform", (10.0, 20.0), atom=0.4))
    np.testing.assert_allclose(out, [[1.0, 0.6], [0.6, 1.0]], rtol=1e-6)
