import numpy as np

from ikob2.core.numerics import DTYPE, safe_divide
from ikob2.domain.segments import CarAccess, DecayParams, Income, Preference, Segment
from ikob2.domain.state import ModelState
from ikob2.engine.runner import SegmentedRunner


def make_problem(n=40, seed=0):
    rng = np.random.default_rng(seed)
    xy = rng.uniform(0, 20, size=(n, 2))
    cost = np.linalg.norm(xy[:, None] - xy[None, :], axis=2)
    state = ModelState.create(
        generalized_cost=cost,
        population=rng.uniform(1, 100, n),
        opportunities=rng.uniform(1, 100, n),
        decay_type="exponential",
        decay_beta=0.1,
    )
    fast = DecayParams("exponential", beta=0.05)
    slow = DecayParams("exponential", beta=0.15)
    segments = [
        Segment("car_high", Income.HIGH, CarAccess.WITH_CAR, Preference.CAR, fast),
        Segment("car_low", Income.LOW, CarAccess.WITH_CAR, Preference.CAR, fast),
        Segment("pt_low", Income.LOW, CarAccess.NO_CAR, Preference.PT, slow),
    ]
    populations = {
        s.name: rng.uniform(1, 50, n).astype(DTYPE) for s in segments
    }
    return state, segments, populations


def test_segments_sharing_weight_key_share_result():
    state, segments, populations = make_problem()
    result = SegmentedRunner().run(state, segments, populations)

    # car_high and car_low share a weight_key -> identical accessibility
    assert np.array_equal(result.per_segment["car_high"],
                          result.per_segment["car_low"])
    # pt_low has a different beta -> different result
    assert not np.allclose(result.per_segment["car_high"],
                           result.per_segment["pt_low"])


def test_batched_matches_unbatched_reference():
    state, segments, populations = make_problem()
    result = SegmentedRunner().run(state, segments, populations)

    # Unbatched reference: run each segment's decay against summed
    # competition computed per-segment (no population pre-summing).
    from ikob2.core.decay_curves import apply_decay
    from ikob2.core.numerics import ensure_dense

    n = state.n_zones
    competition = np.zeros(n, dtype=np.float64)
    decays = {}
    for seg in segments:
        d = ensure_dense(apply_decay(
            state.generalized_cost, seg.decay.decay_type, seg.decay.beta,
            cutoff=seg.decay.cutoff, epsilon=seg.decay.epsilon,
        ))
        decays[seg.name] = d
        competition += d.T @ populations[seg.name].astype(np.float64)
    competition = np.maximum(competition, 1e-6)

    adjusted = state.opportunities / competition
    # float32 batching only changes summation order -> allclose, not equal
    assert np.allclose(result.competition, competition, rtol=1e-4)
    for seg in segments:
        expected = decays[seg.name] @ adjusted
        assert np.allclose(result.per_segment[seg.name], expected, rtol=1e-4)


def test_total_is_population_weighted_mean():
    state, segments, populations = make_problem()
    result = SegmentedRunner().run(state, segments, populations)

    num = np.zeros(state.n_zones, dtype=DTYPE)
    den = np.zeros(state.n_zones, dtype=DTYPE)
    for seg in segments:
        num += result.per_segment[seg.name] * populations[seg.name]
        den += populations[seg.name]
    assert np.allclose(result.total, safe_divide(num, den), rtol=1e-5)


def test_pin_decay_gives_same_answer():
    state, segments, populations = make_problem()
    r1 = SegmentedRunner(pin_decay=False).run(state, segments, populations)
    r2 = SegmentedRunner(pin_decay=True).run(state, segments, populations)
    for seg in segments:
        assert np.array_equal(r1.per_segment[seg.name], r2.per_segment[seg.name])