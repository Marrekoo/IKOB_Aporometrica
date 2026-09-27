"""
Public transport fares from distances.

Two parts, after the paper (Section 3.4):

  * rail: the NS single fare, second class, full tariff, from 1 January
    2026 (`ns_2026_2e_klas.csv`, tariff units -> EUR), read linearly
    between whole units, one tariff unit per rail kilometre: 3.00 EUR up to
    8 units, 4.60 at 15, 8.00 at 30, 12.40 at 50, 19.10 at 80, 22.70 at 100,
    28.80 at 150 and 33.30 at 200, held beyond (`rail_beyond_table`). Any
    other table can be given (`rail_table`, km -> EUR); with
    `rail_table=None` a tapering power law through two anchors (about 2.60
    EUR per kilometre over one kilometre, 0.20 over one hundred) is used
    instead;
  * bus, tram, metro, ferry ("regional"): a boarding charge of about
    1.08 EUR plus 0.18 EUR per kilometre. The charge is paid ONCE per
    journey by default (a 35-minute transfer window makes a single
    boarding the practical case); `boardings="count"` charges every
    boarding.

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

from ikob2.params import DEFAULTS


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
    """Public transport fare per journey: rail by the NS tariff table (or a
    tapering power law through two anchors), read linearly between tariff
    units, with an optional discount; bus, tram, metro and ferry by a
    boarding charge plus a rate per km, the boarding once per journey or per
    boarding. Defaults from the `pt_fare` parameters."""
    rail_eur_per_km_at_1km: float = DEFAULTS.pt_fare.rail_eur_per_km_at_1km
    rail_eur_per_km_at_100km: float = DEFAULTS.pt_fare.rail_eur_per_km_at_100km
    rail_table: tuple[tuple[float, float], ...] | None = NS_RAIL_TABLE
    rail_beyond_table: str = DEFAULTS.pt_fare.rail_beyond_table  # cap, linear
    rail_discount: float = DEFAULTS.pt_fare.rail_discount   # NS 0.2 / 0.4
    regional_boarding_eur: float = DEFAULTS.pt_fare.regional_boarding_eur
    regional_eur_per_km: float = DEFAULTS.pt_fare.regional_eur_per_km
    boardings: str = DEFAULTS.pt_fare.boardings   # single, count"

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

    @classmethod
    def from_params(cls, p) -> "PtFareModel":
        """From the `pt_fare` parameters: a rail table CSV (km, eur)
        replaces the NS table; `rail_anchors` selects the power law."""
        import pandas as pd

        table = NS_RAIL_TABLE
        if p.rail_anchors:
            table = None
        elif p.rail_table:
            df = pd.read_csv(p.rail_table)
            table = tuple(zip(df["km"].astype(float), df["eur"].astype(float)))
        return cls(
            rail_eur_per_km_at_1km=p.rail_eur_per_km_at_1km,
            rail_eur_per_km_at_100km=p.rail_eur_per_km_at_100km,
            rail_table=table, rail_beyond_table=p.rail_beyond_table,
            rail_discount=p.rail_discount,
            regional_boarding_eur=p.regional_boarding_eur,
            regional_eur_per_km=p.regional_eur_per_km, boardings=p.boardings)

    @property
    def matrix_id(self) -> str:
        """Identity of the fare matrix this model produces, used as the cost
        matrix key."""
        table = ("ns" if self.rail_table == NS_RAIL_TABLE else "table") \
            if self.rail_table else (
            f"{self.rail_eur_per_km_at_1km:g}-{self.rail_eur_per_km_at_100km:g}")
        disc = f"-{self.rail_discount:g}" if self.rail_discount else ""
        return (f"ptfare(rail={table}{disc},reg={self.regional_boarding_eur:g}+"
                f"{self.regional_eur_per_km:g}/km,{self.boardings})")

    def rail_fare(self, km) -> np.ndarray:
        """Rail fare (EUR) for in-vehicle rail km: the tariff table or the
        power law, at least the minimum fare above 0 km, 0 at 0 km, NaN
        where km is NaN, times (1 - rail_discount)."""
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
