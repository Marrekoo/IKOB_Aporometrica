"""
Population segments for the engine.

A segment's spatial weight matrix is determined by
(time_cost_id, money_cost_id, ClassFilter): a composed tolerance filter
(time marginal, optional money-cost marginal, copula, scaling), with the
frozen, value-based ClassFilter as the unit of equality. Segments with
equal weight_key share one composed matrix.

The registry key vocabulary (weight_key, time_marginal_key,
cost_marginal_key) lives here so the runner and the cache name matrices
the same way.

epsilon is not part of weight_key: it has a single owner (the runner) and
is applied once, to the composed matrix, inside compose_filters, so it is
uniform across a run and carries no identity.

pool is not part of weight_key either: segments in different pools with
equal filters share one composed matrix; pools multiply matvecs, never
matrices.
"""

from dataclasses import dataclass

from ikob2.domain.filter_config import ClassFilter


@dataclass(frozen=True)
class Segment:
    """A population slice with its own composed tolerance filter.

    time_cost_id / money_cost_id name entries in the runner's
    cost_matrices mapping. money_cost_id and class_filter.cost are a
    biconditional, enforced loudly:
      * cost filter without money_cost_id -> cannot be evaluated;
      * money_cost_id without cost filter -> dead config that LOOKS
        like it does something (the same rule the loader applies to
        class-level copulas without cost blocks).
    This also keeps weight_key normalised for free: money_cost_id is
    None exactly when the filter is time-only, so time-only segments
    with different *hypothetical* money skims still dedup together.

    pool: name of the competition pool. The runner requires an
    opportunities vector for every pool referenced by any segment.
    """
    name: str
    income: str | None            # income class, e.g. "D3"
    class_filter: ClassFilter
    time_cost_id: str = "time"
    money_cost_id: str | None = None
    pool: str = "default"
    household_type: str | None = None

    def __post_init__(self):
        if self.class_filter.cost is not None and self.money_cost_id is None:
            raise ValueError(
                f"Segment '{self.name}' has a cost filter but no "
                f"money_cost_id; the cost marginal has no matrix to "
                f"evaluate on."
            )
        if self.class_filter.cost is None and self.money_cost_id is not None:
            raise ValueError(
                f"Segment '{self.name}' names money_cost_id "
                f"'{self.money_cost_id}' but has no cost filter; this "
                f"is dead configuration. Remove the id or add the filter."
            )

    # ── Registry key vocabulary (single source of truth) ─────────────

    @property
    def weight_key(self) -> tuple:
        """Segments with equal weight_key share one COMPOSED matrix.

        pool is deliberately absent (matvecs multiply, matrices don't).
        epsilon is deliberately absent (run-global, no identity — see
        module docstring before changing this).
        """
        return (self.time_cost_id, self.money_cost_id, self.class_filter)

    @property
    def time_marginal_key(self) -> tuple:
        """Registry key of this segment's time marginal (pre-copula,
        dense, unsparsified)."""
        return ("marginal", self.time_cost_id, self.class_filter.time)

    @property
    def cost_marginal_key(self) -> tuple | None:
        """Registry key of the cost marginal, or None for time-only
        filters."""
        if self.class_filter.cost is None:
            return None
        return ("marginal", self.money_cost_id, self.class_filter.cost)

    def __str__(self) -> str:
        return self.name
