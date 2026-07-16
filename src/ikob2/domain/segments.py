"""
Structured population segments.

DecayParams now carries a params TUPLE instead of a single beta, so
multi-parameter curves (notably the legacy logistic with alpha, omega,
scaling) work through the pipeline. Still frozen/hashable: it
participates in weight_key batching.

Segment gains `pool`: the competition pool this segment belongs to.
Segments in the same pool compete for that pool's opportunity supply
(runner receives a pool -> opportunities mapping). Pooling was
previously an implicit consequence of legacy loop structure (by income
class); it is now an explicit, user-controlled modelling choice.

pool is deliberately NOT part of weight_key: segments in different
pools with equal (cost_id, decay) share one decay matrix. Pools
multiply matvecs, never matrices.
"""

from dataclasses import dataclass
from enum import StrEnum


class Income(StrEnum):
    LOW = "laag"
    MID_LOW = "middellaag"
    MID_HIGH = "middelhoog"
    HIGH = "hoog"


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


@dataclass(frozen=True)
class DecayParams:
    """Everything that determines a decay matrix, given a cost matrix.

    params holds the curve parameters positionally, matching the curve
    function's signature after `cost`:
        exponential / power:  (beta,)
        linear_threshold:     (threshold,)
        logistic:             (alpha, omega, scaling)
    A bare float is accepted and normalised to a 1-tuple.
    """
    decay_type: str
    params: tuple[float, ...]
    cutoff: float = 180.0
    epsilon: float = 1e-3

    def __post_init__(self):
        if isinstance(self.params, (int, float)):
            object.__setattr__(self, "params", (float(self.params),))
        else:
            object.__setattr__(
                self, "params", tuple(float(p) for p in self.params)
            )

    # ── Convenience constructors ─────────────────────────────────────

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


@dataclass(frozen=True)
class Segment:
    """A population slice with its own travel behaviour.

    pool: name of the competition pool. The runner requires an
    opportunities vector for every pool referenced by any segment.
    """
    name: str
    income: Income
    car_access: CarAccess
    preference: Preference
    decay: DecayParams
    has_free_pt: bool = False
    cost_id: str = "default"
    pool: str = "default"

    @property
    def weight_key(self) -> tuple:
        """Segments with equal weight_key share one decay matrix."""
        return (self.cost_id, self.decay)

    def __str__(self) -> str:
        return self.name