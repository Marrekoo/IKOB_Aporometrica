import numpy as np

from ikob2.core.accessibility import (
    build_decay_matrix,
    compute_accessibility,
    compute_naive_accessibility,
)
from ikob2.domain.state import ModelState


def make_state(n=2, beta=1.0):
    cost = np.array([[0.0, 1.0],
                     [1.0, 0.0]])
    return ModelState.create(
        generalized_cost=cost,
        population=np.ones(n),
        opportunities=np.array([10.0, 20.0]),
        decay_type="exponential",
        decay_beta=beta,
        decay_epsilon=0.0,
    )


def test_accessibility_exponential():
    # by hand: weights exp(-cost) = [[1, e^-1], [e^-1, 1]], competition per
    # destination 1 + e^-1 (one person per zone), A_i = sum_j w_ij O_j / V_j
    state = make_state()
    A = compute_accessibility(state)
    w = np.exp(-1.0)
    expected = np.array([10.0 + 20.0 * w, 10.0 * w + 20.0]) / (1.0 + w)
    np.testing.assert_allclose(A, expected, rtol=1e-6)


def test_accessibility_accepts_prebuilt_decay_matrix():
    state = make_state()
    D = build_decay_matrix(state)
    assert np.allclose(compute_accessibility(state, decay_matrix=D),
                       compute_accessibility(state))


def test_epsilon_zeroes_far_weights():
    # off-diagonal weight is exp(-1) ~ 0.37, below epsilon; only
    # intra-zonal (cost 0, weight 1) survives sparsification
    state = make_state().with_updates(decay_epsilon=0.5)
    A = compute_naive_accessibility(state)
    assert np.allclose(A, state.opportunities, rtol=1e-5)


def test_shen_conservation():
    # Competition-adjusted accessibility conserves total opportunities:
    # sum_i P_i * A_i == sum_j O_j  (Shen 1998 identity)
    rng = np.random.default_rng(0)
    n = 50
    xy = rng.uniform(0, 10, size=(n, 2))
    cost = np.linalg.norm(xy[:, None] - xy[None, :], axis=2)
    state = ModelState.create(
        generalized_cost=cost,
        population=rng.uniform(1, 100, n),
        opportunities=rng.uniform(1, 100, n),
        decay_type="exponential",
        decay_beta=0.1,
        decay_epsilon=0.0,   # truncation breaks the identity slightly
    )
    A = compute_accessibility(state)
    assert np.isclose(float(state.population @ A),
                      float(state.opportunities.sum()), rtol=1e-3)

def test_unreachable_opportunities_contribute_zero():
    """A zone with jobs but zero competition (nobody can reach it)
    must contribute exactly nothing — the safe_divide contract
    downstream of compute_competition's raw zeros. A floor such as
    1e-6 would inject O * 1e6 here."""
    D = np.array([
        [1.0, 0.0],
        [1.0, 0.0],
    ], dtype=np.float32)          # nobody reaches destination 1
    state = make_state()          # opportunities = [10, 20]

    acc = compute_accessibility(state, decay_matrix=D)
    assert np.all(np.isfinite(acc))

    # zone 1's 20 jobs must be invisible: identical to 0 jobs there
    state_no_jobs = make_state().with_updates(
        opportunities=np.array([10.0, 0.0], dtype=np.float32)
    )
    acc_no_jobs = compute_accessibility(state_no_jobs, decay_matrix=D)
    assert np.array_equal(acc, acc_no_jobs)