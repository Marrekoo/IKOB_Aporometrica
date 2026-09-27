"""
One-way journeys per home-based tour in ODiN: the basis of `legs_per_tour`.

    python envelope/journeys_per_tour.py <ODiN csv> [<ODiN csv> ...]

An envelope per tour (`envelope.unit = "tour"`) divides a household's monthly
mobility residual by its number of TOURS, where a tour is a home-based chain:
a person's regular trips (ODiN verplaatsingen, `Verpl` = 1) in order, a new
tour starting after every trip whose destination is home (`Doel` = 1), after
dropping touring (`MotiefV` 9) and business trips (2, 3); only tours with a
priced main mode (car driver or passenger, train, bus/tram/metro, other:
`KHvm` 1, 2, 3, 4, 7) count. So X_M is then EUR per home-based tour, while the
accessibility model prices one one-way journey (door to door) at a time. The
number of journeys per tour converts one into the other
(`accessibility.legs_per_tour`).

Prints aggregates only (ODiN microdata stay local).
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

HOME, TOURING, BUSINESS = 1, 9, (2, 3)
PRICED = (1, 2, 3, 4, 7)
COLS = ["OPID", "VerplNr", "Verpl", "Doel", "KHvm", "MotiefV", "FactorV", "AfstV"]


def home_based_tours(trips: pd.DataFrame) -> pd.DataFrame:
    """One row per home-based tour with a priced mode: journeys, weight, km."""
    t = trips[(trips["Verpl"] == 1) & trips["VerplNr"].notna()
              & (trips["MotiefV"] != TOURING) & ~trips["MotiefV"].isin(BUSINESS)]
    t = t.sort_values(["OPID", "VerplNr"])
    after_home = (t["Doel"] == HOME).groupby(t["OPID"]).shift(fill_value=False)
    t = t.assign(tour=after_home.groupby(t["OPID"]).cumsum())
    g = t.groupby(["OPID", "tour"]).agg(
        journeys=("VerplNr", "size"), weight=("FactorV", "first"),
        km=("AfstV", lambda s: s.sum() / 10.0),
        priced=("KHvm", lambda s: s.isin(PRICED).any()))
    return g[g["priced"]]


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("odin", nargs="+")
    args = p.parse_args(argv)
    trips = pd.concat([pd.read_csv(f, sep=";", usecols=COLS, decimal=",",
                                   low_memory=False) for f in args.odin])
    tours = home_based_tours(trips)
    n = tours["journeys"]
    print(f"priced home-based tours: {len(tours)}")
    print(f"journeys per tour: mean {n.mean():.2f}, "
          f"weighted (FactorV) {np.average(n, weights=tours['weight']):.2f}")
    print("share by journeys per tour:",
          n.clip(upper=5).value_counts(normalize=True).sort_index().round(3).to_dict())
    print(f"median km per tour {tours['km'].median():.1f}")


if __name__ == "__main__":
    main()
