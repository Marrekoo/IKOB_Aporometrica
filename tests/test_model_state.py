import numpy as np
import pytest

from ikob2.core.numerics import DTYPE
from ikob2.domain.state import ModelState


def make_state(n=3, **overrides):
    kwargs = dict(
        generalized_cost=np.ones((n, n)),
        population=np.ones(n),
        opportunities=np.ones(n),
        decay_type="exponential",
        decay_beta=0.1,
    )
    kwargs.update(overrides)
    return ModelState.create(**kwargs)


def test_model_state_validates():
    state = make_state(n=3)
    state.validate()  # should not raise


def test_create_coerces_dtype():
    # float64 inputs must come out as the model DTYPE
    state = make_state(n=3)
    assert state.generalized_cost.dtype == DTYPE
    assert state.population.dtype == DTYPE
    assert state.opportunities.dtype == DTYPE


def test_direct_construction_with_wrong_dtype_fails():
    # Direct construction bypasses coercion; validate() must catch it.
    n = 2
    state = ModelState(
        n_zones=n,
        generalized_cost=np.ones((n, n)),        # float64
        population=np.ones(n, dtype=DTYPE),
        opportunities=np.ones(n, dtype=DTYPE),
        decay_type="exponential",
        decay_beta=0.1,
    )
    with pytest.raises(TypeError):
        state.validate()


def test_shape_mismatch_rejected():
    with pytest.raises(ValueError):
        make_state(n=3, generalized_cost=np.ones((2, 2)))


def test_time_money_must_come_together():
    n = 2
    with pytest.raises(ValueError):
        make_state(n=n, time_component=np.ones((n, n)))  # no money_component


def test_with_recomputed_cost():
    n = 2
    t = np.ones((n, n))
    m = np.full((n, n), 2.0)
    state = make_state(
        n=n,
        generalized_cost=t + 0.5 * m,
        time_component=t,
        money_component=m,
        tvom=0.5,
    )
    new_state = state.with_recomputed_cost(tvom=1.0)
    assert np.allclose(new_state.generalized_cost, t + 1.0 * m)
    # original untouched
    assert np.allclose(state.generalized_cost, t + 0.5 * m)