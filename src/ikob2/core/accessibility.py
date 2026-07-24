"""
Competition-adjusted gravity accessibility (Shen 1998).

Pass 1: V_j = (D.T @ P)
Pass 2: A_i = D @ (O / V)
"""

import numpy as np

from ikob2.core.competition import compute_competition
from ikob2.core.decay_curves import apply_decay
from ikob2.core.numerics import DTYPE, matvec, safe_divide
from ikob2.domain.state import ModelState


def build_decay_matrix(state: ModelState):
    return apply_decay(
        state.generalized_cost,
        state.decay_type,
        state.decay_params,
        epsilon=state.decay_epsilon,
    )


def compute_accessibility(state: ModelState, decay_matrix=None) -> np.ndarray:
    state.validate()

    D = decay_matrix if decay_matrix is not None else build_decay_matrix(state)

    competition = compute_competition(D, state.population)
    adjusted_opportunities = safe_divide(state.opportunities, competition)

    accessibility = matvec(D, adjusted_opportunities)

    if state.zone_weights is not None:
        accessibility *= state.zone_weights

    return accessibility.astype(DTYPE, copy=False)


def compute_naive_accessibility(state: ModelState, decay_matrix=None) -> np.ndarray:
    state.validate()
    D = decay_matrix if decay_matrix is not None else build_decay_matrix(state)
    return matvec(D, state.opportunities)