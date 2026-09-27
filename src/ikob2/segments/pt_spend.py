"""Public transport fare spending per person and year, by income decile, from ODiN.

Needed for the public cost of a fare discount: the accessibility model counts
acceptable journeys, not trips, so the number of paid trips comes from ODiN.

    spend[k] = trips_per_year[k] x mean_fare[k]

for income decile k (`HHGestInkG`, the standardised disposable household
income in 10% groups). A trip is a regular ODiN tour (`Verpl` = 1: one movement,
one direction, one purpose) with at least one public transport leg (train, bus,
tram, metro). Its fare comes from the model's own fare rules
(`skims.pt_fare.PtFareModel`): the rail kilometres of its train legs, the
kilometres of its bus/tram/metro legs and the number of those boardings
(`AfstR` in hectometres, `Rvm` the leg mode). The number of trips per person and
year is the weighted number of such tours (`FactorV`) over the weighted number
of persons (`FactorP`), all ages, both ODiN years pooled. Residents of the
study municipality are shrunk towards the national rate,

    x = (n_local x_local + k x_national) / (n_local + k),

with n_local the number of local respondents (k defaults to 100 persons).

It is a level of paid travel of the average person in the decile, not of
the persons who travel by public transport; a discount is costed at this
baseline volume (no induced trips).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ikob2.params import DEFAULTS
from ikob2.skims.pt_fare import PtFareModel

_COLS = ["OPID", "HHGestInkG", "WoGem", "FactorP", "VerplID", "Verpl", "RitID",
         "Rvm", "AfstR", "FactorV"]
RAIL = (2,)
OTHER_PT = (3, 4, 5)          # bus, tram, metro


def read_odin(path: str | Path) -> pd.DataFrame:
    raw = pd.read_csv(path, sep=";", usecols=_COLS, decimal=",", low_memory=False)
    return raw


def pt_trips(legs: pd.DataFrame, fare_model: PtFareModel) -> pd.DataFrame:
    """One row per regular tour with a public transport leg: person, decile,
    municipality, tour weight and fare (EUR)."""
    d = legs[(legs["Verpl"] == 1) & legs["RitID"].notna()].copy()
    d["km"] = d["AfstR"].clip(lower=0) / 10.0
    d["rail"] = d["Rvm"].isin(RAIL)
    d["other"] = d["Rvm"].isin(OTHER_PT)
    d["rail_km"] = np.where(d["rail"], d["km"], 0.0)
    d["other_km"] = np.where(d["other"], d["km"], 0.0)
    g = d.groupby("VerplID").agg(
        OPID=("OPID", "first"), inc=("HHGestInkG", "first"), gem=("WoGem", "first"),
        w=("FactorV", "first"), rail_km=("rail_km", "sum"), other_km=("other_km", "sum"),
        boardings=("other", "sum"), any_pt=("rail", "max"), other_any=("other", "max"))
    g = g[g["any_pt"] | g["other_any"]].copy()
    g["fare"] = fare_model.fare(g["rail_km"].to_numpy(), g["other_km"].to_numpy(),
                                g["boardings"].to_numpy())
    return g[["OPID", "inc", "gem", "w", "fare"]]


def pt_spend_by_decile(legs: pd.DataFrame, fare_model: PtFareModel | None = None,
                       municipality: int | None = DEFAULTS.ownership.municipality,
                       prior: float = DEFAULTS.ownership.pt_spend_prior
                       ) -> pd.DataFrame:
    """Table income_class, trips_per_year, mean_fare_eur, spend_eur_year with the
    national and local respondent counts (see the module doc)."""
    fare_model = fare_model or PtFareModel()
    persons = legs.drop_duplicates("OPID")[["OPID", "HHGestInkG", "WoGem", "FactorP"]]
    persons = persons[persons["HHGestInkG"].between(1, 10)]
    trips = pt_trips(legs, fare_model)
    trips = trips[trips["inc"].between(1, 10)]
    rows = []
    for k in range(1, 11):
        rec = {"income_class": f"D{k}"}
        for scope, pmask, tmask in (
                ("national", persons["HHGestInkG"] == k, trips["inc"] == k),
                ("local", (persons["HHGestInkG"] == k) & (persons["WoGem"] == municipality),
                 (trips["inc"] == k) & (trips["gem"] == municipality))):
            p, t = persons[pmask], trips[tmask]
            n_pers = float(p["FactorP"].sum())
            rec[f"n_{scope}"] = len(p)
            rec[f"trips_{scope}"] = float(t["w"].sum()) / n_pers if n_pers else np.nan
            rec[f"fare_{scope}"] = (float(np.average(t["fare"], weights=t["w"]))
                                    if len(t) else np.nan)
        nl = rec["n_local"]
        for what in ("trips", "fare"):
            loc = 0.0 if np.isnan(rec[f"{what}_local"]) else rec[f"{what}_local"]
            rec[what] = (nl * loc + prior * rec[f"{what}_national"]) / (nl + prior)
        rows.append(rec)
    out = pd.DataFrame(rows)
    out["trips_per_year"] = out["trips"]
    out["mean_fare_eur"] = out["fare"]
    out["spend_eur_year"] = out["trips_per_year"] * out["mean_fare_eur"]
    return out[["income_class", "trips_per_year", "mean_fare_eur", "spend_eur_year",
                "n_national", "n_local", "trips_national", "trips_local",
                "fare_national", "fare_local"]]
