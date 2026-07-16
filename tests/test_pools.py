import numpy as np
import pytest

from ikob2.core.decay_curves import apply_decay, logistic
from ikob2.core.numerics import DTYPE, ensure_dense
from ikob2.data.validation import validate_pools
from ikob2.domain.segments import CarAccess, DecayParams, Income, Preference, Segment
from ikob2.domain.state import ModelState
from ikob2.engine.runner import SegmentedRunner


def _segment(name, pool="default", decay=None):
    return Segment(
        name=name, income=Income.LOW, car_access=CarAccess.WITH_CAR,
        preference=Preference.CAR,
        decay=decay or DecayParams("linear_threshold", (20.0,)),
        pool=pool,
    )


def _state(n=2, cost=None, population=None, opportunities=None):
    cost = cost if cost is not None else np.array([[0.0, 10.0], [10.0, 0.0]])
    return ModelState.create(
        generalized_cost=cost,
        population=population if population is not None else np.ones(n),
        opportunities=(opportunities if opportunities is not None
                       else np.ones(n)),
        decay_type="linear_threshold",
        decay_beta=20.0,
    )


# ── Fix 1: logistic curve is reachable through the pipeline ──────────

def test_logistic_through_apply_decay_matches_direct_formula():
    rng = np.random.default_rng(0)
    cost = rng.uniform(0, 170, size=(30, 30)).astype(DTYPE)
    alpha, omega, scaling = 0.125, 45.0, 0.95   # legacy work-constants shape

    got = ensure_dense(apply_decay(cost, "logistic", (alpha, omega, scaling),
                                   cutoff=180.0, epsilon=1e-3))
    expected = scaling / (1.0 + np.exp(alpha * (cost - omega)))
    expected[expected < 1e-3] = 0.0
    np.testing.assert_allclose(got, expected, rtol=1e-6)


def test_logistic_decayparams_is_hashable_for_batching():
    a = DecayParams.logistic(0.125, 45.0, 0.95)
    b = DecayParams.logistic(0.125, 45.0, 0.95)
    assert a == b and hash(a) == hash(b)
    assert _segment("x", decay=a).weight_key == _segment("y", decay=b).weight_key


def test_scalar_params_still_accepted():
    assert DecayParams("exponential", 0.05).params == (0.05,)


# ── Fix 2: zero-competition zones contribute zero, not O/floor ───────

def test_zero_competition_zone_contributes_zero():
    # Identity decay matrix (threshold below off-diagonal cost).
    seg = _segment("s", decay=DecayParams("linear_threshold", (5.0,)))
    state = _state(population=np.array([1.0, 0.0]),
                   opportunities=np.array([0.0, 5.0]))
    result = SegmentedRunner().run(
        state, [seg], {"s": np.array([1.0, 0.0])},
        opportunities={"default": np.array([0.0, 5.0])},
    )
    a = result.per_segment["s"]
    assert np.all(np.isfinite(a))
    # Zone 2 has 5 jobs but zero reachable population: its contribution
    # must be 0 everywhere, and zone 1 has no jobs -> all zeros.
    np.testing.assert_allclose(a, [0.0, 0.0])


# ── Pools: hand-computed two-pool case ───────────────────────────────

def test_two_pools_hand_computed():
    # D = all-ones (costs 0/10, threshold 20).
    seg_a = _segment("A", pool="p")
    seg_b = _segment("B", pool="q")
    state = _state()
    pops = {"A": np.array([1.0, 0.0]), "B": np.array([0.0, 2.0])}
    opps = {"p": np.array([3.0, 0.0]), "q": np.array([0.0, 4.0])}

    result = SegmentedRunner().run(state, [seg_a, seg_b], pops,
                                   opportunities=opps)

    # V_p = 1s.T @ [1,0] = [1,1];  A_A = 1s @ [3/1, 0] = [3,3]
    # V_q = 1s.T @ [0,2] = [2,2];  A_B = 1s @ [0, 4/2] = [2,2]
    np.testing.assert_allclose(result.per_segment["A"], [3.0, 3.0])
    np.testing.assert_allclose(result.per_segment["B"], [2.0, 2.0])
    np.testing.assert_allclose(result.competition["p"], [1.0, 1.0])
    np.testing.assert_allclose(result.competition["q"], [2.0, 2.0])
    # total = (3*[1,0] + 2*[0,2]) / [1,2] = [3, 2]
    np.testing.assert_allclose(result.total, [3.0, 2.0])


def test_single_pool_default_matches_explicit_default_pool():
    segs = [_segment("A"), _segment("B")]
    state = _state()
    pops = {"A": np.array([1.0, 2.0]), "B": np.array([3.0, 4.0])}

    implicit = SegmentedRunner().run(state, segs, pops)
    explicit = SegmentedRunner().run(
        state, segs, pops, opportunities={"default": state.opportunities})

    for name in ("A", "B"):
        np.testing.assert_array_equal(
            implicit.per_segment[name], explicit.per_segment[name])
    np.testing.assert_array_equal(implicit.total, explicit.total)


def test_matrix_shared_across_pools_but_results_differ():
    # Same weight_key in two pools: one decay matrix, pool-specific A.
    seg_p = _segment("P", pool="p")
    seg_q = _segment("Q", pool="q")
    state = _state()
    pops = {"P": np.array([1.0, 0.0]), "Q": np.array([1.0, 0.0])}
    opps = {"p": np.array([10.0, 0.0]), "q": np.array([0.0, 10.0])}

    result = SegmentedRunner().run(state, [seg_p, seg_q], pops,
                                   opportunities=opps)
    assert not np.allclose(result.per_segment["P"], result.per_segment["Q"])


# ── Misconfiguration ─────────────────────────────────────────────────

def test_segment_referencing_unknown_pool_raises():
    seg = _segment("A", pool="nonexistent")
    state = _state()
    with pytest.raises(ValueError, match="validation failed"):
        SegmentedRunner().run(state, [seg], {"A": np.ones(2)},
                              opportunities={"default": np.ones(2)})


def test_unused_pool_warns():
    report = validate_pools(
        [_segment("A", pool="p")],
        {"p": np.ones(2), "orphan": np.ones(2)},
    )
    assert report.ok
    assert any("orphan" in w for w in report.warnings)


def test_supply_double_count_warns():
    total = np.array([3.0, 4.0])
    report = validate_pools(
        [_segment("A", pool="p"), _segment("B", pool="q")],
        {"p": total, "q": total},          # same total vector to both
        total_opportunities=total,
    )
    assert report.ok                        # warning, not error
    assert any("double-counted" in w for w in report.warnings)


def test_zero_population_pool_warns():
    report = validate_pools(
        [_segment("A", pool="p")],
        {"p": np.ones(2)},
        populations={"A": np.zeros(2)},
    )
    assert any("zero total population" in w for w in report.warnings)