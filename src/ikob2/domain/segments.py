"""
Structured population segments — filter-composition edition.

IDENTITY SHIFT. A segment's spatial weight matrix used to be
determined by (cost_id, DecayParams). It is now determined by
(time_cost_id, money_cost_id, ClassFilter): the weights are a COMPOSED
tolerance filter — time marginal, optional money-cost marginal, copula,
scaling — and the frozen, value-based ClassFilter is the unit of
equality. Two segments with equal weight_key share one composed matrix;
this is the SAME criterion FilterConfig.additivity_armed() tests, so
the engine's dedup and the Hansen-additivity invariant agree on what
"the same filter" means by construction, not by convention.

Registry key vocabulary lives HERE (weight_key, time_marginal_key,
cost_marginal_key) so the runner and the cache can never drift on how
a matrix is named.

EPSILON REVERSAL — read before "fixing" this back. epsilon used to be
a REQUIRED DecayParams field with no default, because it determined
the matrix and a module-level default once disagreed with another
default and broke single-Shen/segmented nesting. Under composition,
epsilon has a single owner (the CLI/runner) and is applied ONCE, to
the composed matrix, inside compose_filters. It is therefore uniform
across every matrix of a run and carries no identity information;
re-adding it to weight_key would fragment dedup without discriminating
anything. The old bug class is now prevented by there being exactly
one epsilon in existence per run — a stronger guarantee than requiring
it per segment ever was.

SCALING moved with it: the legacy logistic carried (alpha, omega,
scaling) positionally; CurveSpec is (alpha, omega) and scaling lives
on ClassFilter, applied after composition. The loader rejects scaling
inside curve blocks; segments inherit that rule by carrying ClassFilter
verbatim.

pool remains deliberately NOT part of weight_key: segments in
different pools with equal filters share one composed matrix. Pools
multiply matvecs, never matrices.

DecayParams is RETAINED, legacy-only: the single-population path
(ModelState.decay_params -> compute_accessibility) still speaks it,
and its epsilon is still per-instance there. Unifying that path under
single-owner epsilon is a separate change; do not half-do it here.
"""

from dataclasses import dataclass
from enum import StrEnum

from ikob2.domain.filter_config import (
    INCOME_CLASSES,
    ClassFilter,
    CurveSpec,
    FilterConfig,
)


class Income(StrEnum):
    LOW = "laag"
    MID_LOW = "middellaag"
    MID_HIGH = "middelhoog"
    HIGH = "hoog"


# The filter loader validates configs against INCOME_CLASSES; segments
# resolve their ClassFilter via Income.value. If these two vocabularies
# drift, resolution fails at runtime in confusing ways — so lock them
# at import time instead.
if tuple(m.value for m in Income) != tuple(INCOME_CLASSES):
    raise ImportError(
        f"Income enum {tuple(m.value for m in Income)} and "
        f"filter_config.INCOME_CLASSES {tuple(INCOME_CLASSES)} have "
        f"diverged. These must stay in lockstep."
    )


class CarAccess(StrEnum):
    WITH_CAR = "with_car"
    FREE_CAR = "free_car"
    NO_CAR = "no_car"           # license, no car (car-share/taxi proxy)
    NO_LICENSE = "no_license"


class Preference(StrEnum):
    CAR = "car"
    NEUTRAL = "neutral"
    BIKE = "bike"
    PT = "pt"


# ── Legacy decay identity (single-population path ONLY) ──────────────

@dataclass(frozen=True)
class DecayParams:
    """LEGACY. Everything that determines a decay matrix, given a cost
    matrix — for the un-segmented ModelState/compute_accessibility path.

    Segments no longer use this class; they carry a ClassFilter.
    Differences to be aware of when migrating:
      * logistic params here are (alpha, omega, scaling); CurveSpec
        logistic is (alpha, omega), scaling on ClassFilter;
      * epsilon here is per-instance and part of identity; in the
        filter world epsilon is run-global (see module docstring).
    """
    decay_type: str
    params: tuple[float, ...]
    epsilon: float | None

    def __post_init__(self):
        if isinstance(self.params, (int, float)):
            object.__setattr__(self, "params", (float(self.params),))
        else:
            object.__setattr__(
                self, "params", tuple(float(p) for p in self.params)
            )

    @classmethod
    def exponential(cls, beta: float, **kw) -> "DecayParams":
        return cls("exponential", (beta,), **kw)

    @classmethod
    def power(cls, beta: float, **kw) -> "DecayParams":
        return cls("power", (beta,), **kw)

    @classmethod
    def logistic(cls, alpha: float, omega: float,
                 scaling: float = 1.0, **kw) -> "DecayParams":
        """The legacy IKOB sigmoid (Tables 9-11 constants go here)."""
        return cls("logistic", (alpha, omega, scaling), **kw)


# ── Segment ──────────────────────────────────────────────────────────

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
    income: Income
    car_access: CarAccess
    preference: Preference
    class_filter: ClassFilter
    has_free_pt: bool = False
    time_cost_id: str = "default"
    money_cost_id: str | None = None
    pool: str = "default"

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


# ── FilterConfig glue ────────────────────────────────────────────────

def resolve_class_filter(fc: FilterConfig, mode: str,
                         income: Income) -> ClassFilter:
    """Look up the ClassFilter for (mode, income class).

    The loader guarantees every mode defines exactly the canonical
    classes, so once the mode exists this cannot miss on income."""
    try:
        per_class = fc.filters[mode]
    except KeyError:
        raise KeyError(
            f"FilterConfig defines no mode '{mode}'; "
            f"available: {sorted(fc.filters)}."
        ) from None
    return per_class[income.value]