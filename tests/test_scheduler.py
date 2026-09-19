"""
tests/engine/test_scheduler.py

Round-trip tests for the process-parallel path: results must match a
direct in-process SimulationRunner, one failing run must not discard
the batch, and every Variant must survive pickling (spawn-based pools
serialise everything; a lambda-carrying variant only fails on
Windows/macOS unless a test catches it here).
"""

import pickle

import numpy as np
import pytest

from ikob2.domain.state import ModelState
from ikob2.engine.runner import SimulationRunner
from ikob2.engine.scheduler import Scheduler
from ikob2.engine.experiment_expander import RunSpec
from ikob2.variants.base import (
    AddOpportunities,
    CompositeVariant,
    MultiplyGeneralizedCost,
    ScaleMonetaryCost,
    ScalePopulation,
    SetDecayBeta,
    SetTvom,
    Variant,
)

N_ZONES = 5


def make_state(seed: int = 7) -> ModelState:
    rng = np.random.default_rng(seed)
    xy = rng.uniform(0, 30, size=(N_ZONES, 2))
    cost = np.linalg.norm(xy[:, None, :] - xy[None, :, :], axis=2)
    return ModelState.create(
        generalized_cost=cost,
        population=rng.integers(50, 500, N_ZONES).astype(float),
        opportunities=rng.integers(10, 300, N_ZONES).astype(float),
        decay_type="exponential",
        decay_beta=0.05,
        decay_epsilon=0.0,
    )


# ── Round trip ───────────────────────────────────────────────────────

def test_scheduler_matches_in_process_runner():
    state = make_state()
    specs = [
        RunSpec("baseline", (), {"kind": "baseline"}),
        RunSpec("beta_003", (SetDecayBeta(0.03),), {"kind": "sensitivity"}),
        RunSpec("beta_010", (SetDecayBeta(0.10),), {"kind": "sensitivity"}),
        RunSpec("cost_15", (MultiplyGeneralizedCost(1.5),), {"kind": "sensitivity"}),
    ]

    results = Scheduler(max_workers=2).run_all(state, specs)

    assert set(results) == {s.run_id for s in specs}
    runner = SimulationRunner()
    for spec in specs:
        expected = runner.run(state, list(spec.variants))
        np.testing.assert_allclose(
            results[spec.run_id], expected, rtol=1e-6,
            err_msg=f"Worker result diverged for {spec.run_id}",
        )


# ── Failure isolation ────────────────────────────────────────────────

def test_one_failing_run_does_not_discard_the_batch():
    """ScaleMonetaryCost raises on a state without money_component —
    a picklable, deterministic in-worker failure. The other runs must
    still come back."""
    state = make_state()
    specs = [
        RunSpec("good_1", (SetDecayBeta(0.04),), {}),
        RunSpec("bad", (ScaleMonetaryCost(2.0),), {}),
        RunSpec("good_2", (SetDecayBeta(0.06),), {}),
    ]

    results = Scheduler(max_workers=2).run_all(state, specs)

    assert "bad" not in results
    assert set(results) == {"good_1", "good_2"}
    expected = SimulationRunner().run(state, [SetDecayBeta(0.06)])
    np.testing.assert_allclose(results["good_2"], expected, rtol=1e-6)


# ── Variant picklability ─────────────────────────────────────────────

VARIANT_CASES = [
    SetDecayBeta(0.07),
    MultiplyGeneralizedCost(1.1),
    SetTvom(9.5),
    ScaleMonetaryCost(2.0),
    AddOpportunities(-50.0),
    ScalePopulation(1.2),
    CompositeVariant([SetDecayBeta(0.07), MultiplyGeneralizedCost(1.1)]),
]


@pytest.mark.parametrize("variant", VARIANT_CASES, ids=lambda v: v.name)
def test_variant_survives_pickle(variant):
    restored = pickle.loads(pickle.dumps(variant))
    assert isinstance(restored, Variant)
    assert restored.name == variant.name


@pytest.mark.parametrize(
    "variant",
    [v for v in VARIANT_CASES if not isinstance(
        v, (SetTvom, ScaleMonetaryCost))],
    ids=lambda v: v.name,
)
def test_pickled_variant_produces_identical_state(variant):
    """Not just importable — the round-tripped variant must transform a
    state identically. (SetTvom/ScaleMonetaryCost excluded: they need
    the time/money decomposition, covered by pickling alone above.)"""
    state = make_state()
    restored = pickle.loads(pickle.dumps(variant))
    a = variant(state)
    b = restored(state)
    np.testing.assert_array_equal(
        np.asarray(a.generalized_cost.todense()) if hasattr(a.generalized_cost, "todense")
        else a.generalized_cost,
        np.asarray(b.generalized_cost.todense()) if hasattr(b.generalized_cost, "todense")
        else b.generalized_cost,
    )
    np.testing.assert_array_equal(a.population, b.population)
    np.testing.assert_array_equal(a.opportunities, b.opportunities)
    assert a.decay_params == b.decay_params
    assert a.decay_epsilon == b.decay_epsilon


def test_composite_variant_is_picklable_and_ordered():
    """Order matters: cost×1.5 then beta-set ≠ beta-set semantics if
    someone later makes variants non-commuting. Assert the restored
    composite preserves sequence."""
    composite = CompositeVariant([SetDecayBeta(0.03), MultiplyGeneralizedCost(2.0)])
    restored = pickle.loads(pickle.dumps(composite))
    assert [v.name for v in restored.variants] == [v.name for v in composite.variants]