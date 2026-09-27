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

from ikob2.domain.filter_config import INDEPENDENCE, ClassFilter, CurveSpec
from ikob2.domain.segments import Segment
from ikob2.domain.state import ModelState
from ikob2.engine.runner import SegmentedRunner


def _scaled(state, factor):
    """The state with its cost matrix multiplied by `factor`."""
    return state.with_updates(
        generalized_cost=(state.generalized_cost * factor).astype(np.float32))


def _filter(curve, *params, scaling=1.0):
    """Time-only, independence-copula filter (the reference filter)."""
    return ClassFilter(CurveSpec(curve, tuple(float(p) for p in params)),
                       None, INDEPENDENCE, scaling)


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
        decay_epsilon=0.0,
    )


def make_segments() -> list[Segment]:
    fast = _filter("exponential", 0.04)
    slow = _filter("exponential", 0.08)
    return [
        # a and b share a weight_key on purpose (aliasing + batching tests)
        Segment("seg_a", "hoog", fast,
                time_cost_id="time"),
        Segment("seg_b", "laag", fast,
                time_cost_id="time"),
        Segment("seg_c", "laag", slow,
                time_cost_id="time"),
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
    """A reused runner must recompute decay matrices when the cost
    matrix changes — the strongest form of the assertion is
    equality with a FRESH runner, not just difference from baseline."""
    reused = SegmentedRunner(decay_epsilon=None, pin_decay=pin_decay)
    baseline = reused.run(state, segments, populations)
    varied = reused.run(_scaled(state, 1.5), segments, populations)

    fresh = SegmentedRunner(decay_epsilon=None, pin_decay=pin_decay).run(
        _scaled(state, 1.5), segments, populations)

    # The buggy runner returns varied == baseline. Guard against that
    # explicitly so the failure message points at staleness, not at a
    # subtle numeric mismatch.
    assert not np.allclose(varied.total, baseline.total), (
        "Second run returned baseline results despite a changed cost matrix: "
        "the registry served a stale decay matrix."
    )

    np.testing.assert_allclose(varied.total, fresh.total, rtol=1e-6)
    for pool in fresh.competition:
        np.testing.assert_allclose(
            varied.competition[pool], fresh.competition[pool], rtol=1e-6)
    for seg in segments:
        np.testing.assert_allclose(
            varied.per_segment[seg.name], fresh.per_segment[seg.name], rtol=1e-6
        )


@pytest.mark.parametrize("pin_decay", [False, True])
def test_baseline_rerun_is_reproducible(state, segments, populations, pin_decay):
    """Reuse with identical inputs must be a no-op — protects against
    over-correcting the staleness fix into non-determinism."""
    runner = SegmentedRunner(decay_epsilon=None, pin_decay=pin_decay)
    r1 = runner.run(state, segments, populations)
    r2 = runner.run(state, segments, populations)
    np.testing.assert_array_equal(r1.total, r2.total)
    for pool in r1.competition:
        np.testing.assert_array_equal(r1.competition[pool], r2.competition[pool])


def test_interleaved_runs_do_not_cross_contaminate(state, segments, populations):
    """baseline -> scaled cost -> baseline: the third run must reproduce the
    first exactly. Catches one-way fixes that refresh recipes on the
    second run but leave its matrix behind for the next."""
    runner = SegmentedRunner(decay_epsilon=None, pin_decay=True)
    r1 = runner.run(state, segments, populations)
    runner.run(_scaled(state, 1.5), segments, populations)
    r3 = runner.run(state, segments, populations)
    np.testing.assert_array_equal(r1.total, r3.total)


# ── Pin lifecycle ────────────────────────────────────────────────────

def test_no_pins_leak_after_successful_run(state, segments, populations):
    runner = SegmentedRunner(decay_epsilon=None, pin_decay=True)
    runner.run(state, segments, populations)
    assert runner.registry.pinned_size_mb() == 0.0


def test_no_pins_leak_after_mid_run_failure(state, populations):
    """First key pins fine, second key's recipe raises (unknown decay
    type) during pass 1. The try/finally must still unpin the first."""
    good = _filter("exponential", 0.04)
    bad = _filter("no_such_curve", 0.04)
    segments = [
        Segment("seg_a", "hoog", good,
                time_cost_id="time"),
        Segment("seg_b", "laag", good,
                time_cost_id="time"),
        Segment("seg_c", "laag", bad,
                time_cost_id="time"),
    ]
    runner = SegmentedRunner(decay_epsilon=None, pin_decay=True)

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
    result = SegmentedRunner(decay_epsilon=None).run(state, segments, populations)
    a = result.per_segment["seg_a"]
    b = result.per_segment["seg_b"]

    np.testing.assert_array_equal(a, b)   # same key -> same values...
    assert not np.shares_memory(a, b)     # ...but not the same memory

    expected_b = b.copy()
    a += 999.0                            # simulate in-place consumer
    np.testing.assert_array_equal(result.per_segment["seg_b"], expected_b)