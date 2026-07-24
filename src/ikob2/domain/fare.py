"""
Out-of-pocket cost model: an affine transform of a distance skim.

fee + rate * detour * km is not an approximation — it is the actual
tariff structure for the modes that matter (OV-chipkaart pricing IS
boarding fee + per-km rate; car is per-km fuel with no fee). Free
travel is FareModel(0, 0): money cost identically zero, cost
marginal ~1 everywhere, composed filter collapses toward the time
marginal. No special case exists downstream for it.

detour absorbs the one honest imperfection of the proxy: the only
distance skim is the car freeflow network, while PT fares follow
routed PT kilometres. A uniform routed-km/car-km ratio (~1.2-1.4)
is defensible; OD pairs whose true detour deviates strongly (water
barriers, missing rail links) are mispriced, and those correlate
with equity-relevant geography. Documented limitation, not a bug.

Frozen on purpose: the dataclass IS the identity. matrix_id becomes
the cost_matrices dict key, hence the (matrix_id, CurveSpec)
marginal-cache key in both the runner registry and run.py's
FilterMatrixCache, hence the composed-matrix identity — two runs or
(later) two segments with different fare assumptions can never share
a composed matrix. Change a parameter, change the key.
"""

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class FareModel:
    fee: float                  # boarding/starting fee (EUR)
    rate_per_km: float          # marginal cost (EUR/km)
    detour: float = 1.0         # routed-km / car-network-km ratio
    skim_id: str = "afstand_auto_freeflow"

    def __post_init__(self):
        for name in ("fee", "rate_per_km", "detour"):
            value = getattr(self, name)
            if not math.isfinite(value):
                raise ValueError(f"FareModel.{name} must be finite, "
                                 f"got {value!r}")
        if self.fee < 0 or self.rate_per_km < 0:
            raise ValueError(f"Negative fare parameters: {self}")
        if self.detour <= 0:
            raise ValueError(f"detour must be positive: {self}")

    @property
    def matrix_id(self) -> str:
        """Cache/registry key. %g keeps 0.0 and 0 identical."""
        return (f"fare({self.skim_id},fee={self.fee:g},"
                f"rate={self.rate_per_km:g},detour={self.detour:g})")