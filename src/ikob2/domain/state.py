"""
Immutable snapshot of all numeric inputs required by the core model.

Rewritten version:
- Supports multi-parameter decay curves (e.g. logistic).
- Removes legacy decay_cutoff.
- decay_params may be float or tuple.
- No artificial truncation behaviour baked into state.
"""

from dataclasses import dataclass, replace, field
from typing import Any, Dict, Union

import numpy as np

from ikob2.core.numerics import DTYPE, as_dtype, is_sparse

Matrix = Union[np.ndarray, "scipy.sparse.spmatrix"]  # noqa


def _check_matrix(name: str, m, expected_shape) -> None:
    if m.dtype != DTYPE:
        raise TypeError(f"{name} must be {np.dtype(DTYPE).name}, got {m.dtype}")
    if tuple(m.shape) != tuple(expected_shape):
        raise ValueError(
            f"{name} has shape {tuple(m.shape)}, expected {tuple(expected_shape)}"
        )


def _check_vector(name: str, v, n: int) -> None:
    if is_sparse(v):
        raise TypeError(f"{name} must be dense 1-D array, got sparse")
    if v.dtype != DTYPE:
        raise TypeError(f"{name} must be {np.dtype(DTYPE).name}, got {v.dtype}")
    if v.shape != (n,):
        raise ValueError(f"{name} has shape {v.shape}, expected ({n},)")


@dataclass(frozen=True)
class ModelState:
    # ── Geometry ─────────────────────────────
    n_zones: int

    # ── Core matrices ────────────────────────
    generalized_cost: Matrix

    # ── Zone-level attributes ────────────────
    population: np.ndarray
    opportunities: np.ndarray

    # ── Decay specification ──────────────────
    decay_type: str
    decay_params: float | tuple
    decay_epsilon: float | None

    # ── Optional decomposition ───────────────
    time_component: Matrix | None = None
    money_component: Matrix | None = None
    tvom: float | None = None

    # ── Optional extras ──────────────────────
    zone_weights: np.ndarray | None = None
    metadata: Dict[str, Any] | None = field(default=None)

    # ─────────────────────────────────────────
    # Construction
    # ─────────────────────────────────────────

    @classmethod
    def create(
        cls,
        generalized_cost,
        population,
        opportunities,
        decay_type: str,
        decay_beta,
        *,
        decay_epsilon: float | None,
        time_component=None,
        money_component=None,
        tvom: float | None = None,
        zone_weights=None,
        metadata: Dict[str, Any] | None = None,
    ) -> "ModelState":

        def _mat(m):
            if m is None:
                return None
            if is_sparse(m):
                return m.astype(DTYPE)
            return as_dtype(m)

        population = as_dtype(population)
        opportunities = as_dtype(opportunities)
        n = len(population)

        # Support float or tuple
        if isinstance(decay_beta, (int, float)):
            decay_params = float(decay_beta)
        else:
            decay_params = tuple(float(x) for x in decay_beta)

        state = cls(
            n_zones=n,
            generalized_cost=_mat(generalized_cost),
            population=population,
            opportunities=opportunities,
            decay_type=decay_type,
            decay_params=decay_params,
            decay_epsilon=float(decay_epsilon),
            time_component=_mat(time_component),
            money_component=_mat(money_component),
            tvom=None if tvom is None else float(tvom),
            zone_weights=None if zone_weights is None else as_dtype(zone_weights),
            metadata=metadata,
        )

        state.validate()
        return state

    # ─────────────────────────────────────────
    # Validation
    # ─────────────────────────────────────────

    def validate(self) -> None:
        n = self.n_zones

        _check_matrix("generalized_cost", self.generalized_cost, (n, n))
        _check_vector("population", self.population, n)
        _check_vector("opportunities", self.opportunities, n)

        if self.decay_epsilon < 0:
            raise ValueError("decay_epsilon must be >= 0")

        if self.time_component is not None:
            _check_matrix("time_component", self.time_component, (n, n))
        if self.money_component is not None:
            _check_matrix("money_component", self.money_component, (n, n))

        if (self.time_component is None) != (self.money_component is None):
            raise ValueError("time_component and money_component must be given together")

        if self.time_component is not None and self.tvom is None:
            raise ValueError("tvom required when time/money components are set")

        if self.zone_weights is not None:
            _check_vector("zone_weights", self.zone_weights, n)

    # ─────────────────────────────────────────
    # Functional updates
    # ─────────────────────────────────────────

    def with_updates(self, **changes) -> "ModelState":
        new_state = replace(self, **changes)
        new_state.validate()
        return new_state