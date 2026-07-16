"""
tests/engine/test_runner_reuse.py

Regression tests for SegmentedRunner reuse across sequential run()
calls. The bug class: weight_key identifies a cost matrix by cost_id (a
string), so a registry recipe surviving from a previous run can silently
serve decay weights computed from a stale cost matrix. No exception, no
warning — just wrong numbers. These tests exist to make that impossible
to reintroduce.
"""

import numpy as np
import pytest

from ikob2.domain.segments import CarAccess, DecayParams, Income, Preference, Segment
from ikob2.domain.state import ModelState
from ikob2.engine.runner import SegmentedRunner
from ikob2.variants.base import MultiplyGeneralizedCost


# ── Fixtures ─────────────────────────────────────────────────────────

N_ZONES = 12


def make_state(seed: int = 42) -> ModelState:
    rng = np.random.default_rng(seed)
    xy = rng.uniform(0, 60, size=(N_ZONES, 2))
    cost = np.linalg.norm(xy[:, None, :] - xy[None, :, :], axis=2)
    return ModelState.create(
        generalized_cost=cost,
        population=rng.integers(50, 500, N_ZONES).astype(float),
        opportunities=rng.integers(10, 300, N_ZONES).astype(float),
        decay_type="exponential",
        decay_beta=0.05,
    )


def make_segments() -> list[Segment]:
    fast = DecayParams("exponential", beta=0.04)
    slow = DecayParams("exponential", beta=0.08)
    return [
        # a and b share a weight_key on purpose (aliasing + batching tests)
        Segment("seg_a", Income.HIGH, CarAccess.WITH_CAR, Preference.CAR, fast),
        Segment("seg_b", Income.LOW, CarAccess.WITH_CAR, Preference.CAR, fast),
        Segment("seg_c", Income.LOW, CarAccess.NO_CAR, Preference.PT, slow),
    ]


def make_populations(state: ModelState, segments, seed: int = 1):
    rng = np.random.default_rng(seed)
    shares = rng.dirichlet(np.ones(len(segments)), size=state.n_zones)
    return {
        seg.name: (state.population * shares[:, i]).astype(np.float32)
        for i, seg in enumerate(segments)
    }


@pytest.fixture
def state():
    return make_state()


@pytest.fixture
def segments():
    return make_segments()


@pytest.fixture
def populations(state, segments):
    return make_populations(state, segments)


# ── The staleness regression ─────────────────────────────────────────

@pytest.mark.parametrize("pin_decay", [False, True])
def test_second_run_sees_new_cost_matrix(state, segments, populations, pin_decay):
    """A reused runner must recompute decay matrices when a variant
    changes the cost matrix — the strongest form of the assertion is
    equality with a FRESH runner, not just difference from baseline."""
    reused = SegmentedRunner(pin_decay=pin_decay)
    baseline = reused.run(state, segments, populations)
    varied = reused.run(
        state, segments, populations,
        variants=[MultiplyGeneralizedCost(1.5)],
    )

    fresh = SegmentedRunner(pin_decay=pin_decay).run(
        state, segments, populations,
        variants=[MultiplyGeneralizedCost(1.5)],
    )

    # The buggy runner returns varied == baseline. Guard against that
    # explicitly so the failure message points at staleness, not at a
    # subtle numeric mismatch.
    assert not np.allclose(varied.total, baseline.total), (
        "Second run returned baseline results despite a cost variant: "
        "the registry served a stale decay matrix."
    )

    np.testing.assert_allclose(varied.total, fresh.total, rtol=1e-6)
    np.testing.assert_allclose(varied.competition, fresh.competition, rtol=1e-6)
    for seg in segments:
        np.testing.assert_allclose(
            varied.per_segment[seg.name], fresh.per_segment[seg.name], rtol=1e-6
        )


@pytest.mark.parametrize("pin_decay", [False, True])
def test_baseline_rerun_is_reproducible(state, segments, populations, pin_decay):
    """Reuse with identical inputs must be a no-op — protects against
    over-correcting the staleness fix into non-determinism."""
    runner = SegmentedRunner(pin_decay=pin_decay)
    r1 = runner.run(state, segments, populations)
    r2 = runner.run(state, segments, populations)
    np.testing.assert_array_equal(r1.total, r2.total)
    np.testing.assert_array_equal(r1.competition, r2.competition)


def test_interleaved_runs_do_not_cross_contaminate(state, segments, populations):
    """baseline -> variant -> baseline: the third run must reproduce the
    first exactly. Catches one-way fixes that refresh recipes on the
    variant run but leave the variant matrix behind for the next."""
    runner = SegmentedRunner(pin_decay=True)
    r1 = runner.run(state, segments, populations)
    runner.run(state, segments, populations,
               variants=[MultiplyGeneralizedCost(1.5)])
    r3 = runner.run(state, segments, populations)
    np.testing.assert_array_equal(r1.total, r3.total)


# ── Pin lifecycle ────────────────────────────────────────────────────

def test_no_pins_leak_after_successful_run(state, segments, populations):
    runner = SegmentedRunner(pin_decay=True)
    runner.run(state, segments, populations)
    assert runner.registry.pinned_size_mb() == 0.0


def test_no_pins_leak_after_mid_run_failure(state, populations):
    """First key pins fine, second key's recipe raises (unknown decay
    type) during pass 1. The try/finally must still unpin the first."""
    good = DecayParams("exponential", beta=0.04)
    bad = DecayParams("no_such_curve", beta=0.04)
    segments = [
        Segment("seg_a", Income.HIGH, CarAccess.WITH_CAR, Preference.CAR, good),
        Segment("seg_b", Income.LOW, CarAccess.WITH_CAR, Preference.CAR, good),
        Segment("seg_c", Income.LOW, CarAccess.NO_CAR, Preference.PT, bad),
    ]
    runner = SegmentedRunner(pin_decay=True)

    with pytest.raises(ValueError, match="decay type"):
        runner.run(state, segments, populations)

    assert runner.registry.pinned_size_mb() == 0.0, (
        "A pinned decay matrix survived a failed run."
    )


# ── Result aliasing ──────────────────────────────────────────────────

def test_no_aliasing_between_segments_sharing_key(state, segments, populations):
    """seg_a and seg_b share a weight_key. Their per_segment arrays must
    be independent buffers: a caller normalising one in place must not
    corrupt the other."""
    result = SegmentedRunner().run(state, segments, populations)
    a = result.per_segment["seg_a"]
    b = result.per_segment["seg_b"]

    np.testing.assert_array_equal(a, b)   # same key -> same values...
    assert not np.shares_memory(a, b)     # ...but not the same memory

    expected_b = b.copy()
    a += 999.0                            # simulate in-place consumer
    np.testing.assert_array_equal(result.per_segment["seg_b"], expected_b)