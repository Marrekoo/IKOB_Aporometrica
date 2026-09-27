import pandas as pd
import pytest

from ikob2.domain.filter_config import CurveSpec
from ikob2.outputs.export import DICTIONARY, write_products


def test_products_are_written_and_join_on_buurtcode(tmp_path):
    gpd = pytest.importorskip("geopandas")
    from shapely.geometry import box

    kwb = tmp_path / "kwb.gpkg"
    gpd.GeoDataFrame({"buurtcode": ["BU1", "BU2", "BU3"],
                      "buurtnaam": ["a", "b", "c"]},
                     geometry=[box(i, 0, i + 1, 1) for i in range(3)],
                     crs="EPSG:28992").to_file(kwb, layer="buurten")
    table = pd.DataFrame({
        "buurtcode": ["BU1", "BU1", "BU2", "BU2"],
        "mode": ["pt", "pt", "pt", "pt"],
        "segment": ["single_D2", "single_D3"] * 2,
        "household_type": "single", "income_class": ["D2", "D3"] * 2,
        "population": [10.0, 30.0, 20.0, 20.0],
        "accessibility": [100.0, 200.0, 50.0, 150.0],
        "accessibility_expected": [50.0, 100.0, 25.0, 75.0]})
    env = pd.DataFrame({"household_type": ["single", "single"],
                        "income_class": ["D2", "D3"], "low": [1.0, 2.0],
                        "high": [10.0, 20.0], "atom": [0.0, 0.0]})
    hubs = pd.DataFrame({"hub": ["h"], "kind": ["lime"], "lat": [52.09],
                         "lon": [5.1], "source": ["x.csv"]})
    written = write_products(
        tmp_path, table, envelope=env, price_scale={"single_D2": 0.5},
        time_margins={("pt", "no_wfh"): CurveSpec("weibull", (2.0, 40.0))},
        kwb_path=kwb, hubs=hubs)
    assert {"segments.csv", "time_margins.csv", "hubs.csv", "origins.gpkg",
            "dictionary.csv", "README.md"} <= set(written)
    seg = pd.read_csv(tmp_path / "segments.csv").set_index("segment")
    assert seg.loc["single_D2", "lime_price_scale"] == 0.5
    assert seg.loc["single_D3", "lime_price_scale"] == 1.0
    g = gpd.read_file(tmp_path / "origins.gpkg", layer="origins")
    assert list(g["buurtcode"]) == ["BU1", "BU2"]           # only the origins
    b1 = g.set_index("buurtcode").loc["BU1"]
    assert b1["acc_pt"] == pytest.approx((100 * 10 + 200 * 30) / 40)
    assert b1["accx_pt"] == pytest.approx((50 * 10 + 100 * 30) / 40)
    assert g.crs.to_epsg() == 28992
    assert len(gpd.read_file(tmp_path / "origins.gpkg", layer="hubs")) == 1
    tm = pd.read_csv(tmp_path / "time_margins.csv")
    assert tm.loc[0, "shape"] == 2.0 and tm.loc[0, "scale"] == 40.0


def test_dictionary_documents_the_main_table_columns():
    cols = {c for f, c, *_ in DICTIONARY if f == "accessibility.csv"}
    assert {"buurtcode", "mode", "segment", "population", "accessibility",
            "atom", "availability", "accessibility_expected"} <= cols


def test_money_gate_products_and_map_columns(tmp_path):
    gpd = pytest.importorskip("geopandas")
    from shapely.geometry import box

    kwb = tmp_path / "kwb.gpkg"
    gpd.GeoDataFrame({"buurtcode": ["BU1"], "buurtnaam": ["a"]},
                     geometry=[box(0, 0, 1, 1)], crs="EPSG:28992").to_file(
        kwb, layer="buurten")
    table = pd.DataFrame({
        "buurtcode": ["BU1", "BU1"], "mode": "pt", "segment":
        ["single_D2", "single_D9"], "household_type": "single",
        "income_class": ["D2", "D9"], "population": [50.0, 50.0],
        "accessibility": [1.0, 2.0], "accessibility_expected": [1.0, 2.0]})
    env = pd.DataFrame({"household_type": ["single", "single"],
                        "income_class": ["D2", "D9"], "low": [0.0, 0.0],
                        "high": [10.0, 400.0], "atom": [0.0, 0.0]})
    pop = pd.DataFrame({"single_D2": [50.0], "single_D9": [50.0]},
                       index=["BU1"])
    written = write_products(tmp_path, table, envelope=env, price_scale=None,
                             time_margins={}, kwb_path=kwb, populations=pop)
    assert {"money_gate_curves.csv", "money_gate_ttt.csv",
            "money_gate_summary.csv"} <= set(written)
    g = gpd.read_file(tmp_path / "origins.gpkg", layer="origins")
    assert g.loc[0, "mg_ttt_shift"] < 0 and g.loc[0, "mg_hazard_class"]
