"""Source tables of the envelope (`data/envelope/sources/`, seeded into
`<data root>/inputs/envelope/sources/`). Every table has a `source` column;
this module reads them and drops the provenance columns."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

TABLES = ("nibud_basket", "nibud_households", "nibud_posts", "nibud_type_keys",
          "bijstand_published", "warnaar_anchors", "rents", "equivalence_cbs",
          "cbs_income_percentiles", "price_index", "car_bundles", "car_class",
          "odin_household_types")
PROVENANCE = ("source", "note")


@dataclass(frozen=True)
class Sources:
    """All source tables, by name (see TABLES)."""
    tables: dict

    def __getitem__(self, name: str) -> pd.DataFrame:
        return self.tables[name]

    @property
    def kappa(self) -> float:
        """Price uprating of the 2022 basket to Warnaar's price level."""
        r = self.tables["price_index"].set_index("name").loc["kappa"]
        return float(r["numerator"]) / float(r["denominator"])

    def rent(self, lineage: str, key: str) -> float:
        """A rent (EUR/month) of a rent lineage."""
        r = self.tables["rents"]
        v = r[(r["lineage"] == lineage) & (r["rent_key"] == key)]["eur_month"]
        if len(v) != 1:
            raise KeyError(f"No rent {key!r} in lineage {lineage!r}.")
        return float(v.iloc[0])

    def eqv(self) -> dict:
        """CBS equivalence factor per household type."""
        e = self.tables["equivalence_cbs"]
        return dict(zip(e["hh_type"], e["factor"].astype(float)))


def load_sources(folder: str | Path) -> Sources:
    """Read every source table from `folder`."""
    folder = Path(folder)
    missing = [t for t in TABLES if not (folder / f"{t}.csv").exists()]
    if missing:
        raise FileNotFoundError(f"{folder}: missing source table(s) {missing}.")
    tables = {}
    for t in TABLES:
        df = pd.read_csv(folder / f"{t}.csv")
        tables[t] = df.drop(columns=[c for c in PROVENANCE if c in df.columns])
    return Sources(tables)
