"""
Pure transformations of ModelState.

Unchanged design; the concrete variants below are consolidated here
from the previous zone/decay/cost variant modules, with sparse-safety
guards: scalar multiplication is CSR-safe, but scalar ADDITION on a
sparse matrix would densify it — such variants use ensure_dense
deliberately or reject sparse input.
"""

from abc import ABC, abstractmethod

import numpy as np

from ikob2.core.numerics import DTYPE, is_sparse
from ikob2.domain.state import ModelState


class Variant(ABC):
    """A pure transformation of ModelState. Must not mutate input state."""

    name: str

    def __init__(self, name: str | None = None):
        self.name = name or self.__class__.__name__

    @abstractmethod
    def apply(self, state: ModelState) -> ModelState:
        """Returns a new modified ModelState."""

    def __call__(self, state: ModelState) -> ModelState:
        return self.apply(state)

    def __repr__(self) -> str:
        return f"<Variant {self.name}>"


class CompositeVariant(Variant):
    def __init__(self, variants: list[Variant]):
        names = "+".join(v.name for v in variants)
        super().__init__(name=f"Composite({names})")
        self.variants = variants

    def apply(self, state: ModelState) -> ModelState:
        new_state = state
        for variant in self.variants:
            new_state = variant(new_state)
        return new_state


# ── Decay parameter variants ─────────────────────────────────────────

class SetDecayBeta(Variant):
    def __init__(self, beta: float):
        super().__init__(name=f"SetDecayBeta({beta})")
        self.beta = float(beta)

    def apply(self, state: ModelState) -> ModelState:
        return state.with_updates(decay_beta=self.beta)


class SetDecayCutoff(Variant):
    def __init__(self, cutoff: float):
        super().__init__(name=f"SetDecayCutoff({cutoff})")
        self.cutoff = float(cutoff)

    def apply(self, state: ModelState) -> ModelState:
        return state.with_updates(decay_cutoff=self.cutoff)


# ── Cost variants ────────────────────────────────────────────────────

class MultiplyGeneralizedCost(Variant):
    """Sparse-safe: csr * scalar stays CSR."""

    def __init__(self, factor: float):
        super().__init__(name=f"MultiplyCost({factor})")
        self.factor = float(factor)

    def apply(self, state: ModelState) -> ModelState:
        new_cost = state.generalized_cost * DTYPE(self.factor)
        if not is_sparse(new_cost):
            new_cost = np.asarray(new_cost, dtype=DTYPE)
        return state.with_updates(generalized_cost=new_cost)


class SetTvom(Variant):
    """Change the value of time and rebuild gtt = t + tvom * m.

    Requires the state to carry the time/money decomposition."""

    def __init__(self, tvom: float):
        super().__init__(name=f"SetTvom({tvom})")
        self.tvom = float(tvom)

    def apply(self, state: ModelState) -> ModelState:
        return state.with_recomputed_cost(tvom=self.tvom)


class ScaleMonetaryCost(Variant):
    """E.g. road pricing / fare scenarios: scale the money component and
    rebuild generalized cost. Requires the decomposition."""

    def __init__(self, factor: float):
        super().__init__(name=f"ScaleMoney({factor})")
        self.factor = float(factor)

    def apply(self, state: ModelState) -> ModelState:
        if state.money_component is None:
            raise ValueError(f"{self.name} requires a money_component on the state")
        new_money = np.asarray(
            state.money_component * DTYPE(self.factor), dtype=DTYPE
        )
        return state.with_recomputed_cost(money_component=new_money)


# ── Zone attribute variants ──────────────────────────────────────────

class AddOpportunities(Variant):
    """Clamped at zero so negative deltas can't produce negative supply."""

    def __init__(self, delta: float):
        super().__init__(name=f"AddOpp({delta})")
        self.delta = float(delta)

    def apply(self, state: ModelState) -> ModelState:
        new_opp = np.maximum(
            state.opportunities + DTYPE(self.delta), 0.0
        ).astype(DTYPE)
        return state.with_updates(opportunities=new_opp)


class ScalePopulation(Variant):
    def __init__(self, factor: float):
        super().__init__(name=f"ScalePop({factor})")
        self.factor = float(factor)

    def apply(self, state: ModelState) -> ModelState:
        new_pop = (state.population * DTYPE(self.factor)).astype(DTYPE)
        return state.with_updates(population=new_pop)