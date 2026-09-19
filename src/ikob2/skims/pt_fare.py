"""
Public transport fares from distances.

Two parts, after the paper (Section 3.4):

  * rail: a fare that tapers with distance. The paper gives the average
    fare per kilometre as about 2.60 EUR over one kilometre and 0.20 EUR
    over one hundred; between these anchors the fare follows a power law
    through fare(1 km) = 2.60 and fare(100 km) = 20, and never falls
    below the 1 km fare (a minimum fare). This is an ASSUMPTION for the
    shape between the anchors: replace it by the operator's tariff table
    (`rail_table`, points km -> EUR, linear in between) when available;
  * bus, tram, metro, ferry ("regional"): a boarding charge of about
    1.08 EUR plus 0.18 EUR per kilometre. Following the earlier R draft
    the charge is paid ONCE per journey by default (a 35-minute transfer
    window makes a single boarding the practical case);
    `boardings="count"` charges every boarding.

Inputs are in-vehicle kilometres per class and the number of boardings
onto regional lines from `PtRouter.journeys`. The fare is applied when a
run is set up, so fare assumptions can change without new routing.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PtFareModel:
    rail_eur_per_km_at_1km: float = 2.60
    rail_eur_per_km_at_100km: float = 0.20
    rail_table: tuple[tuple[float, float], ...] | None = None
    regional_boarding_eur: float = 1.08
    regional_eur_per_km: float = 0.18
    boardings: str = "single"            # "single" or "count"

    def __post_init__(self):
        for name in ("rail_eur_per_km_at_1km", "rail_eur_per_km_at_100km",
                     "regional_boarding_eur", "regional_eur_per_km"):
            v = getattr(self, name)
            if not (math.isfinite(v) and v >= 0):
                raise ValueError(f"{name} must be non-negative and finite, "
                                 f"got {v}")
        if not (self.rail_eur_per_km_at_100km < self.rail_eur_per_km_at_1km):
            raise ValueError("The rail fare must taper: per-km fare at 100 "
                             "km below the fare at 1 km.")
        if self.boardings not in ("single", "count"):
            raise ValueError("boardings must be 'single' or 'count'.")
        if self.rail_table is not None:
            km = [k for k, _ in self.rail_table]
            eur = [e for _, e in self.rail_table]
            if len(km) < 2 or km != sorted(km) or len(set(km)) != len(km):
                raise ValueError("rail_table needs increasing km, at least "
                                 "two points.")
            if eur != sorted(eur) or min(eur) < 0:
                raise ValueError("rail_table fares must be non-decreasing "
                                 "and non-negative.")

    @property
    def matrix_id(self) -> str:
        table = "table" if self.rail_table else (
            f"{self.rail_eur_per_km_at_1km:g}-{self.rail_eur_per_km_at_100km:g}")
        return (f"ptfare(rail={table},reg={self.regional_boarding_eur:g}+"
                f"{self.regional_eur_per_km:g}/km,{self.boardings})")

    def rail_fare(self, km) -> np.ndarray:
        km = np.asarray(km, dtype=float)
        if self.rail_table:
            pts = np.array(self.rail_table, dtype=float)
            fare = np.interp(km, pts[:, 0], pts[:, 1])
            # beyond the table: extend with the last segment's slope
            slope = (pts[-1, 1] - pts[-2, 1]) / (pts[-1, 0] - pts[-2, 0])
            fare = np.where(km > pts[-1, 0],
                            pts[-1, 1] + slope * (km - pts[-1, 0]), fare)
            fare1 = float(np.interp(1.0, pts[:, 0], pts[:, 1]))
        else:
            a = self.rail_eur_per_km_at_1km
            b = self.rail_eur_per_km_at_100km
            exponent = 1.0 - math.log(a / b) / math.log(100.0)
            fare1 = a
            with np.errstate(invalid="ignore"):
                fare = a * np.maximum(km, 0.0) ** exponent
        fare = np.where(km > 0, np.maximum(fare, fare1), 0.0)
        return np.where(np.isnan(km), np.nan, fare)

    def fare(self, rail_km, other_km, other_boardings) -> np.ndarray:
        """Fare (EUR) per journey; NaN where an input is NaN."""
        rail = self.rail_fare(rail_km)
        other_km = np.asarray(other_km, dtype=float)
        boards = np.asarray(other_boardings, dtype=float)
        n = np.minimum(boards, 1.0) if self.boardings == "single" else boards
        regional = np.where(boards > 0,
                            n * self.regional_boarding_eur
                            + self.regional_eur_per_km * other_km, 0.0)
        return (rail + regional).astype(np.float32)
