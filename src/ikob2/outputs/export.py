"""Analysis-ready products of a run, for R (or any other tool).

Written next to `accessibility.csv` (tidy long table: one row per origin x
segment x mode):

  segments.csv       segment definitions: household type, income class, the
                     budget interval (low, high, EUR per trip), the atom, the
                     Lime price scale;
  time_margins.csv   the Weibull time margins (mode, job type, shape, scale);
  hubs.csv           shared-bicycle hubs used (name, kind, lat, lon);
  origins.gpkg       for maps (RD New, EPSG:28992): layer `origins` with the
                     buurt polygons and population-weighted mean accessibility
                     per mode, layer `hubs` with the hub points;
  accessibility.parquet  the long table as Parquet, when pyarrow is installed;
  dictionary.csv     every column of these files: description and unit;
  README.md          how to join them.

The join key is `buurtcode` (CBS, e.g. BU03440111); segments are named
`<household_type>_<income_class>`.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

DICTIONARY = [
    ("accessibility.csv", "buurtcode", "origin buurt (CBS code)", "-"),
    ("accessibility.csv", "mode", "mode or shared-bicycle variant (car, bike, pt, pt_v0..pt_v3, bike_v4)", "-"),
    ("accessibility.csv", "segment", "household type x income decile, '<type>_<class>'", "-"),
    ("accessibility.csv", "household_type", "single, couple, single_parent, couple_children", "-"),
    ("accessibility.csv", "income_class", "income decile D1 (lowest) to D10", "-"),
    ("accessibility.csv", "population", "persons of the segment in the origin", "persons"),
    ("accessibility.csv", "accessibility", "acceptable jobs reached by one person of the segment, conditional on having the mode", "jobs"),
    ("accessibility.csv", "atom", "share of the segment for whom no priced trip is acceptable (cost margin atom)", "share"),
    ("accessibility.csv", "accessibility_normalised", "accessibility / (1 - atom)", "jobs"),
    ("accessibility.csv", "availability", "share of the segment that can use the mode (car in household, private bicycle)", "share"),
    ("accessibility.csv", "accessibility_expected", "accessibility x availability", "jobs"),
    ("segments.csv", "low", "lower bound of the per-trip cost threshold (uniform interval)", "EUR"),
    ("segments.csv", "high", "upper bound of the per-trip cost threshold", "EUR"),
    ("segments.csv", "atom", "share with no acceptable priced trip (censored cell: 1)", "share"),
    ("segments.csv", "lime_price_scale", "multiplier on the Lime price of this segment (concessions)", "-"),
    ("time_margins.csv", "shape", "Weibull shape k of the acceptable travel time", "-"),
    ("time_margins.csv", "scale", "Weibull scale eta of the acceptable travel time", "minutes"),
    ("hubs.csv", "kind", "tariff of the hub: lime (municipal hub) or ovfiets", "-"),
    ("origins.gpkg:origins", "acc_<mode>", "population-weighted mean accessibility of the mode", "jobs"),
    ("origins.gpkg:origins", "accx_<mode>", "the same with availability (accessibility_expected)", "jobs"),
    ("origins.gpkg:origins", "population", "persons in the modelled segments", "persons"),
    ("money_gate_curves.csv", "survival", "aggregated survival S_bar(c) of the cost threshold of the origin (population-weighted mixture of the segments)", "share"),
    ("money_gate_curves.csv", "hazard", "-d log S_bar / dc where S_bar is above 5% of S_bar(0+)", "1/EUR"),
    ("money_gate_ttt.csv", "phi", "total-time-on-test transform of the aggregate at u = F(c); diagonal = exponential, above = increasing hazard", "-"),
    ("money_gate_summary.csv", "ttt_shift", "TTT area of the aggregate minus that of the segments alone; negative: aggregation moves the curve toward decreasing hazard", "-"),
    ("money_gate_summary.csv", "cv", "coefficient of variation of the aggregate cost threshold", "-"),
    ("money_gate_summary.csv", "hazard_class", "increasing / mixed / decreasing hazard of the aggregate in the bulk (survival above 5% of S_bar(0+))", "-"),
    ("money_gate_summary.csv", "ttt_class", "shape of the whole aggregate curve from its TTT area: IFR (> +0.05), near-exponential, DFR (< -0.05)", "-"),
    ("money_gate_summary.csv", "ttt_area", "integral of phi(u) - u of the aggregate: 0 exponential, positive increasing hazard, negative decreasing", "-"),
    ("origins.gpkg:origins", "mg_<column>", "money_gate_summary.csv columns joined on the origin polygons", "-"),
    ("interchange_pairs.csv", "R", "gain of intervention A over gain of B, per origin and segment", "-"),
]

README = """# Run products

`accessibility.csv` is the main table (tidy, one row per origin x segment x
mode). Join the others on `buurtcode` and `segment`; `dictionary.csv` lists
every column. `origins.gpkg` (EPSG:28992) holds the origin polygons and the hub
points for maps; `run.json` has the parameters, the command line and the skim
metadata of the run.

R:

    library(sf); library(readr); library(dplyr)
    acc  <- read_csv("accessibility.csv")
    geo  <- st_read("origins.gpkg", layer = "origins")
    geo |> left_join(filter(acc, mode == "pt_v2", income_class == "D3") |>
                     group_by(buurtcode) |> summarise(a = weighted.mean(accessibility, population)),
                     by = "buurtcode")
"""


def _origin_layer(kwb_path: str | Path, codes) -> "gpd.GeoDataFrame":
    import geopandas as gpd
    import pyogrio

    layers = [name for name, _ in pyogrio.list_layers(str(kwb_path))]
    layer = next((n for n in layers if "buurt" in n.lower()), layers[0])
    g = gpd.read_file(kwb_path, layer=layer)
    code_col = next(c for c in ("buurtcode", "bu_code", "buurtcd") if c in g.columns)
    g = g[g[code_col].isin(list(codes))].rename(columns={code_col: "buurtcode"})
    keep = ["buurtcode"] + [c for c in ("buurtnaam", "gemeentecode",
                                         "gemeentenaam") if c in g.columns]
    return g[keep + ["geometry"]].to_crs("EPSG:28992")


def write_products(out_dir: str | Path, table: pd.DataFrame, *, envelope,
                   price_scale: dict | None, time_margins: dict,
                   kwb_path: str | Path | None, hubs: pd.DataFrame | None = None,
                   populations: pd.DataFrame | None = None) -> list[str]:
    """Write the products; returns the file names written."""
    out = Path(out_dir)
    written = []

    seg = envelope.copy()
    seg["segment"] = seg["household_type"] + "_" + seg["income_class"]
    seg["lime_price_scale"] = seg["segment"].map(price_scale or {}).fillna(1.0)
    seg.to_csv(out / "segments.csv", index=False)
    written.append("segments.csv")

    pd.DataFrame([{"mode": m, "job_type": w, "curve": spec.curve,
                   "shape": spec.params[0], "scale": spec.params[1]}
                  for (m, w), spec in sorted(time_margins.items())
                  if spec.curve == "weibull"]).to_csv(
        out / "time_margins.csv", index=False)
    written.append("time_margins.csv")

    if hubs is not None and len(hubs):
        hubs[["hub", "kind", "lat", "lon", "source"]].to_csv(
            out / "hubs.csv", index=False)
        written.append("hubs.csv")

    gate = None
    if populations is not None:
        from ikob2.outputs.diagnostics import money_gate

        gate = money_gate(envelope, populations)
        gate["curves"].to_csv(out / "money_gate_curves.csv", index=False)
        gate["ttt"].to_csv(out / "money_gate_ttt.csv", index=False)
        gate["summary"].to_csv(out / "money_gate_summary.csv", index=False)
        written += ["money_gate_curves.csv", "money_gate_ttt.csv",
                    "money_gate_summary.csv"]

    try:
        table.to_parquet(out / "accessibility.parquet", index=False)
        written.append("accessibility.parquet")
    except ImportError:
        logger.info("pyarrow not installed: no Parquet copy.")

    if kwb_path is not None:
        import geopandas as gpd

        gpkg = out / "origins.gpkg"
        if gpkg.exists():
            gpkg.unlink()
        g = _origin_layer(kwb_path, table["buurtcode"].unique())
        w = table.assign(_w=table["population"])
        for mode, d in w.groupby("mode"):
            for col, prefix in (("accessibility", "acc"),
                                ("accessibility_expected", "accx")):
                num = (d[col] * d["_w"]).groupby(d["buurtcode"]).sum()
                den = d["_w"].groupby(d["buurtcode"]).sum()
                g[f"{prefix}_{mode}"] = g["buurtcode"].map(num / den.where(den > 0))
        pop = table[table["mode"] == table["mode"].iloc[0]].groupby(
            "buurtcode")["population"].sum()
        g["population"] = g["buurtcode"].map(pop)
        if gate is not None:
            s = gate["summary"].set_index("buurtcode")
            for col in ("atom", "mean_threshold", "cv", "ttt_area", "ttt_shift",
                        "share_hazard_increasing", "hazard_class",
                        "ttt_class"):
                g[f"mg_{col}"] = g["buurtcode"].map(s[col])
        g.to_file(gpkg, layer="origins", driver="GPKG")
        if hubs is not None and len(hubs):
            pts = gpd.GeoDataFrame(
                hubs[["hub", "kind", "source"]],
                geometry=gpd.points_from_xy(hubs["lon"], hubs["lat"]),
                crs="EPSG:4326").to_crs("EPSG:28992")
            pts.to_file(gpkg, layer="hubs", driver="GPKG")
        written.append("origins.gpkg")

    pd.DataFrame(DICTIONARY, columns=["file", "column", "description",
                                      "unit"]).to_csv(out / "dictionary.csv",
                                                      index=False)
    (out / "README.md").write_text(README)
    return written + ["dictionary.csv", "README.md"]
