"""
Public transport fares from distances.

Two parts, after the paper (Section 3.4):

  * rail: the NS single fare by distance (second class), a table read
    linearly between its points: up to 8 km 2.70 EUR (the minimum fare),
    15 km 4.40, 30 km 7.60, 50 km 11.80, 80 km 17.90, 100 km 21.30,
    150 km 26.90, and 200 km and beyond 29.40 (the cap). Any other table
    can be given (`rail_table`, km -> EUR); with `rail_table=None` a
    tapering power law through the paper's anchors (about 2.60 EUR per
    kilometre over one kilometre, 0.20 over one hundred) is used instead;
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
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np


def load_ns_table(path: str | Path | None = None
                  ) -> tuple[tuple[float, float], ...]:
    """(tariff units, EUR) rows of the NS price list CSV."""
    path = Path(path) if path else Path(__file__).with_name(
        "ns_2026_2e_klas.csv")
    rows = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and line != "te,eur":
            te, eur = line.split(",")
            rows.append((float(te), float(eur)))
    return tuple(rows)


def parse_ns_tariff(text: str) -> tuple[tuple[float, float], ...]:
    """Rows of the second-class table from the text of the NS price list
    (`pdftotext -layout`): lines like ' 15   € 4,60   € 3,68 ...'; the row
    '0 t/m 8' counts as 8 tariff units. First price column (full fare
    including VAT)."""
    out = {}
    for line in text.splitlines():
        m = re.match(r"^\s*(\d+)(?:\s+t/m\s+(\d+))?\s+€\s*([\d.,]+)\s+€",
                     line)
        if m:
            te = int(m.group(2) or m.group(1))
            out[te] = float(m.group(3).replace(".", "").replace(",", "."))
    return tuple((float(k), out[k]) for k in sorted(out))


NS_RAIL_TABLE = load_ns_table()


@dataclass(frozen=True)
class PtFareModel:
    rail_eur_per_km_at_1km: float = 2.60
    rail_eur_per_km_at_100km: float = 0.20
    rail_table: tuple[tuple[float, float], ...] | None = NS_RAIL_TABLE
    rail_beyond_table: str = "cap"       # "cap" or "linear" (last slope)
    rail_discount: float = 0.0           # e.g. 0.2 / 0.4 for NS discounts
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
        if not (0.0 <= self.rail_discount < 1.0):
            raise ValueError("rail_discount must be in [0, 1).")
        if self.rail_beyond_table not in ("cap", "linear"):
            raise ValueError("rail_beyond_table must be 'cap' or 'linear'.")
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
        table = ("ns" if self.rail_table == NS_RAIL_TABLE else "table") \
            if self.rail_table else (
            f"{self.rail_eur_per_km_at_1km:g}-{self.rail_eur_per_km_at_100km:g}")
        disc = f"-{self.rail_discount:g}" if self.rail_discount else ""
        return (f"ptfare(rail={table}{disc},reg={self.regional_boarding_eur:g}+"
                f"{self.regional_eur_per_km:g}/km,{self.boardings})")

    def rail_fare(self, km) -> np.ndarray:
        km = np.asarray(km, dtype=float)
        if self.rail_table:
            pts = np.array(self.rail_table, dtype=float)
            # np.interp holds the first value below and the last above:
            # the minimum fare below the first point, a cap above the last
            fare = np.interp(km, pts[:, 0], pts[:, 1])
            if self.rail_beyond_table == "linear":
                slope = (pts[-1, 1] - pts[-2, 1]) / (pts[-1, 0] - pts[-2, 0])
                fare = np.where(km > pts[-1, 0],
                                pts[-1, 1] + slope * (km - pts[-1, 0]), fare)
            fare1 = float(pts[0, 1])
        else:
            a = self.rail_eur_per_km_at_1km
            b = self.rail_eur_per_km_at_100km
            exponent = 1.0 - math.log(a / b) / math.log(100.0)
            fare1 = a
            with np.errstate(invalid="ignore"):
                fare = a * np.maximum(km, 0.0) ** exponent
        fare = np.where(km > 0, np.maximum(fare, fare1), 0.0)
        fare = fare * (1.0 - self.rail_discount)
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
