"""
Car availability per segment from ODiN.

For each household type x income decile (the segments) this is the weighted
share of adults (18+) living in a household with at least one passenger car
(`basis="household_car"`), or with a car and their own driving licence
(`basis="car_and_licence"`). ODiN 2022 and 2023 are pooled; the person
weight FactorP is used. Sparse cells in the study municipality are
shrunk towards the national cell:

    p = (n_local p_local + k p_national) / (n_local + k)

with n_local the number of local respondents and k the prior strength
(default 30). Result: a table household_type x income_class with the share,
the local and national respondent counts and the local and national raw
shares.

The share is a modelled availability: "no car in the household" is not
"cannot travel by car" (lifts, car sharing), and ODiN records the household,
not who may use the car.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ikob2.segments.config import HOUSEHOLD_TYPES

_HHSAM = {1: "single", 2: "couple", 3: "couple_children",
          4: "couple_children", 5: "couple", 6: "single_parent",
          7: "single_parent"}          # 8 = other household: not a segment
_COLS = ["OPID", "HHSam", "HHGestInkG", "HHAuto", "OPRijbewijsAu",
         "FactorP", "WoGem", "Leeftijd"]


def read_odin_persons(path: str | Path) -> pd.DataFrame:
    """One row per adult respondent with numeric codes. Accepts the
    semicolon CSV of the cleaned ODiN pool (decimal comma in the weights)."""
    raw = pd.read_csv(path, sep=";", usecols=_COLS, dtype=str)
    p = raw.drop_duplicates("OPID").copy()
    for c in _COLS:
        if c != "OPID":
            p[c] = pd.to_numeric(p[c].str.replace(",", "."), errors="coerce")
    return p[(p["Leeftijd"] >= 18)]


def car_availability(persons: pd.DataFrame, municipality: int | None = 344,
                     *, basis: str = "household_car", prior: float = 30.0,
                     ) -> pd.DataFrame:
    """Availability per (household_type, income_class); see module doc."""
    if basis not in ("household_car", "car_and_licence"):
        raise ValueError("basis is 'household_car' or 'car_and_licence'.")
    p = persons[(persons["HHGestInkG"].between(1, 10))
                & (persons["HHAuto"] <= 9)
                & persons["HHSam"].isin(_HHSAM)].copy()
    has = p["HHAuto"] > 0
    if basis == "car_and_licence":
        has &= p["OPRijbewijsAu"] == 1
    p["y"] = has.astype(float)
    p["household_type"] = p["HHSam"].map(_HHSAM)
    p["income_class"] = "D" + p["HHGestInkG"].astype(int).astype(str)
    keys = ["household_type", "income_class"]

    def share(g):
        w = g["FactorP"]
        return pd.Series({"n": len(g),
                          "share": float(np.average(g["y"], weights=w))})

    nat = p.groupby(keys).apply(share, include_groups=False)
    loc = (p[p["WoGem"] == municipality] if municipality is not None
           else p.iloc[0:0])
    loc = loc.groupby(keys).apply(share, include_groups=False)
    idx = pd.MultiIndex.from_product(
        [HOUSEHOLD_TYPES, [f"D{i}" for i in range(1, 11)]], names=keys)
    out = pd.DataFrame(index=idx)
    out["n_national"] = nat["n"].reindex(idx).fillna(0)
    out["share_national"] = nat["share"].reindex(idx)
    out["n_local"] = loc["n"].reindex(idx).fillna(0) if len(loc) else 0.0
    out["share_local"] = loc["share"].reindex(idx) if len(loc) else np.nan
    nl = out["n_local"].to_numpy()
    pl = out["share_local"].fillna(0).to_numpy()
    pn = out["share_national"].to_numpy()
    out["share"] = (nl * pl + prior * pn) / (nl + prior)
    return out.reset_index()
