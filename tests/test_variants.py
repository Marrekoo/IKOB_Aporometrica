import numpy as np
import pytest

from ikob2.domain.state import ModelState
from ikob2.variants.base import (
    AddOpportunities,
    CompositeVariant,
    MultiplyGeneralizedCost,
    ScaleMonetaryCost,
    SetDecayBeta,
)


def make_state(n=2):
    return ModelState.create(
        generalized_cost=np.ones((n, n)),
        population=np.ones(n),
        opportunities=np.ones(n),
        decay_type="exponential",
        decay_beta=0.1,
        decay_epsilon=0.0,
    )


def test_decay_variant_changes_beta():
    state = make_state()

    variant = SetDecayBeta(0.5)
    new_state = variant(state)

    assert new_state.decay_params == 0.5
    assert state.decay_params == 0.1  # original unchanged


def test_multiply_cost():
    state = make_state()
    new_state = MultiplyGeneralizedCost(2.0)(state)

    assert np.allclose(new_state.generalized_cost, 2.0)
    assert np.allclose(state.generalized_cost, 1.0)  # original unchanged


def test_add_opportunities_clamps_at_zero():
    state = make_state()
    new_state = AddOpportunities(-5.0)(state)

    assert np.all(new_state.opportunities == 0.0)


def test_composite_applies_in_order():
    state = make_state()
    composite = CompositeVariant([SetDecayBeta(0.3), MultiplyGeneralizedCost(3.0)])
    new_state = composite(state)

    assert new_state.decay_params == 0.3
    assert np.allclose(new_state.generalized_cost, 3.0)


def test_scale_money_requires_decomposition():
    state = make_state()
    with pytest.raises(ValueError):
        ScaleMonetaryCost(2.0)(state)