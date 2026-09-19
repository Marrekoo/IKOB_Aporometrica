"""
Car time and money for the threshold gate, after the legacy IKOB.

Legacy IKOB (ikob/utils.compute_car_gtt, single_weights, config
defaults) builds the car generalised time as

    gtt = drive time + parking search time
          + tvom x [ (var + road pricing) x distance
                     + additional costs + parking costs ]

with, per default, a variable cost of 16 ct/km for a fossil car and
5 ct/km for an electric one, an optional per-km road charge, per-zone
parking costs, and parking search times by urbanisation class of the
zone: arrival {1: 12, 2: 8, 3: 4, 4: 0, 5: 0} minutes and departure a
quarter of that (`urbanization_grade_to_parking_times`; the source notes
that the values are undocumented). Households without a car use a
shared car (0.33 EUR/km + 0.05 EUR/min) or a taxi (2.40 EUR/km +
0.40 EUR/min): a per-km and a per-minute charge, the kappa_1 and kappa_2
terms of the paper's cost equation.

The threshold gate needs time and money SEPARATELY, so this module
returns both instead of one generalised time. Where the legacy adds
`parking[i, arrival] + parking[j, departure]` (origin arrival, destination
departure), this uses the physical reading: leaving origin i costs its
departure search time, arriving at destination j its arrival search time.

Distance: r5py's matrices have no distance. `DetourModel` turns the
crow-fly distance between zone centroids into a route distance with a
detour factor per distance band, calibrated on routed samples.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# KWB urbanisation class (1 = most urban) -> minutes to find parking on
# arrival (legacy IKOB values).
PARKING_ARRIVAL_MIN = {1: 12.0, 2: 8.0, 3: 4.0, 4: 0.0, 5: 0.0}
DEPARTURE_FACTOR = 0.25


@dataclass(frozen=True)
class CarCostModel:
    """Per-km money cost of using a car, and per-minute charges for
    shared-car and taxi users."""
    variable_eur_per_km: float = 0.16       # fossil; electric ~ 0.05
    road_pricing_eur_per_km: float = 0.0
    per_minute_eur: float = 0.0             # shared car 0.05, taxi 0.40

    def __post_init__(self):
        for name in ("variable_eur_per_km", "road_pricing_eur_per_km",
                     "per_minute_eur"):
            v = getattr(self, name)
            if not (np.isfinite(v) and v >= 0):
                raise ValueError(f"{name} must be non-negative and finite, "
                                 f"got {v}")

    @property
    def matrix_id(self) -> str:
        return (f"car(var={self.variable_eur_per_km:g},"
                f"pricing={self.road_pricing_eur_per_km:g},"
                f"min={self.per_minute_eur:g})")


# Legacy alternatives for households without a car.
FOSSIL_CAR = CarCostModel(0.16)
ELECTRIC_CAR = CarCostModel(0.05)
SHARED_CAR = CarCostModel(0.33, per_minute_eur=0.05)
TAXI = CarCostModel(2.40, per_minute_eur=0.40)


def parking_times(urbanisation) -> tuple[np.ndarray, np.ndarray]:
    """(arrival, departure) search minutes per zone from the KWB
    urbanisation class; unknown/missing classes count as class 5 (no
    search time)."""
    u = np.asarray(urbanisation, dtype=float)
    arrival = np.zeros(u.shape)
    for grade, minutes in PARKING_ARRIVAL_MIN.items():
        arrival[u == grade] = minutes
    return arrival, arrival * DEPARTURE_FACTOR


def car_time_and_cost(
    drive_min: np.ndarray,
    distance_km: np.ndarray,
    model: CarCostModel,
    *,
    origin_urbanisation=None,
    dest_urbanisation=None,
    dest_parking_cost_eur=None,
) -> tuple[np.ndarray, np.ndarray]:
    """(time, cost) matrices origins x destinations.

    time = drive + departure search at the origin + arrival search at the
    destination (each only if the urbanisation classes are given), NaN
    where the drive time is NaN. cost = (variable + road pricing) x
    distance + parking cost at the destination + per-minute charge x
    time.
    """
    drive = np.asarray(drive_min, dtype=np.float64)
    dist = np.asarray(distance_km, dtype=np.float64)
    if drive.shape != dist.shape:
        raise ValueError(f"drive {drive.shape} and distance {dist.shape} "
                         f"differ in shape.")
    time = drive.copy()
    if origin_urbanisation is not None:
        _, dep = parking_times(origin_urbanisation)
        time = time + dep[:, None]
    if dest_urbanisation is not None:
        arr, _ = parking_times(dest_urbanisation)
        time = time + arr[None, :]
    cost = (model.variable_eur_per_km + model.road_pricing_eur_per_km) * dist
    if dest_parking_cost_eur is not None:
        cost = cost + np.asarray(dest_parking_cost_eur,
                                 dtype=np.float64)[None, :]
    cost = cost + model.per_minute_eur * time
    return time.astype(np.float32), cost.astype(np.float32)


# ── distance from crow-fly ───────────────────────────────────────────

def crowfly_km(origin_xy, dest_xy) -> np.ndarray:
    """Crow-fly distance (km) between RD New points (metres)."""
    o = np.asarray(origin_xy, dtype=float)
    d = np.asarray(dest_xy, dtype=float)
    return np.hypot(o[:, None, 0] - d[None, :, 0],
                    o[:, None, 1] - d[None, :, 1]) / 1000.0


@dataclass(frozen=True)
class DetourModel:
    """Route distance / crow-fly distance as a function of the crow-fly
    distance: linear interpolation between band medians (`km` are band
    centres, `factor` the detour factors), constant outside."""
    km: tuple[float, ...]
    factor: tuple[float, ...]
    meta: dict = field(default_factory=dict, compare=False, hash=False)

    def __post_init__(self):
        if len(self.km) != len(self.factor) or not self.km:
            raise ValueError("km and factor must be equally long, non-empty.")
        if list(self.km) != sorted(self.km):
            raise ValueError("km must be increasing.")
        if min(self.factor) < 1.0:
            raise ValueError("A route cannot be shorter than the crow-fly "
                             "distance: factors must be >= 1.")

    def route_km(self, crow_km) -> np.ndarray:
        crow = np.asarray(crow_km, dtype=float)
        f = np.interp(crow, self.km, self.factor)
        return crow * f

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(
            {"km": list(self.km), "factor": list(self.factor),
             "meta": self.meta}, indent=1))

    @classmethod
    def load(cls, path: str | Path) -> "DetourModel":
        d = json.loads(Path(path).read_text())
        return cls(tuple(d["km"]), tuple(d["factor"]), d.get("meta", {}))

    @classmethod
    def constant(cls, factor: float = 1.3) -> "DetourModel":
        return cls((0.0,), (float(factor),), {"source": "constant"})


def calibrate_detour(crow_km, route_km,
                     edges=(0.0, 1.0, 3.0, 8.0, 20.0, 50.0, 1e9),
                     min_pairs: int = 20) -> DetourModel:
    """Median route/crow-fly ratio per crow-fly distance band from routed
    pairs. Bands with fewer than `min_pairs` pairs are dropped (the
    neighbours interpolate over them)."""
    crow = np.asarray(crow_km, dtype=float).ravel()
    route = np.asarray(route_km, dtype=float).ravel()
    ok = np.isfinite(crow) & np.isfinite(route) & (crow > 0.05) \
        & (route >= crow * 0.999)
    crow, route = crow[ok], route[ok]
    km, fac, counts = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (crow >= lo) & (crow < hi)
        if sel.sum() < min_pairs:
            continue
        km.append(float(np.median(crow[sel])))
        fac.append(float(np.median(route[sel] / crow[sel])))
        counts.append(int(sel.sum()))
    if not km:
        raise ValueError("No distance band has enough routed pairs.")
    fac = [max(f, 1.0) for f in fac]
    return DetourModel(tuple(km), tuple(fac),
                       {"pairs": int(ok.sum()), "band_counts": counts,
                        "source": "calibrated on routed pairs"})
