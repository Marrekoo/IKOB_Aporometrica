import numpy as np
import pytest

geopandas = pytest.importorskip("geopandas")
from shapely.geometry import Polygon  # noqa: E402

from ikob2.data.geopackage import load_cbs_buurten  # noqa: E402


def _square(x0, y0, size=100.0):
    return Polygon([(x0, y0), (x0 + size, y0), (x0 + size, y0 + size), (x0, y0 + size)])


def make_buurten_gpkg(path, *, crs="EPSG:28992", with_water=True,
                       secret_income=False, duplicate_code=False):
    codes = ["BU00010001", "BU00010002", "BU00010003"]
    if duplicate_code:
        codes[2] = codes[0]

    data = {
        "buurtcode": codes,
        "buurtnaam": ["Centrum", "Noord", "Zuid"],
        "gemeentecode": ["GM0001", "GM0001", "GM0001"],
        "gemeentenaam": ["Voorbeeldstad", "Voorbeeldstad", "Voorbeeldstad"],
        "water": ["NEE", "NEE", "JA" if with_water else "NEE"],
        "aantal_inwoners": [1200, 3400, -99999999 if secret_income else 500],
        "geometry": [_square(0, 0), _square(200, 0), _square(400, 0)],
    }
    gdf = geopandas.GeoDataFrame(data, crs=crs)
    gdf.to_file(path, driver="GPKG", layer="buurten")
    return gdf


def test_loads_basic_zoneset(tmp_path):
    gpkg = tmp_path / "cbs.gpkg"
    make_buurten_gpkg(gpkg, with_water=False)

    zones, report = load_cbs_buurten(gpkg)

    assert report.ok
    assert zones.n_zones == 3
    assert set(zones.codes) == {"BU00010001", "BU00010002", "BU00010003"}
    assert zones.crs == "EPSG:28992"
    # centroid of a 100x100 square at (0,0) is (50, 50)
    idx = zones.code_to_index["BU00010001"]
    assert zones.centroid_x[idx] == pytest.approx(50.0)
    assert zones.centroid_y[idx] == pytest.approx(50.0)
    assert "aantal_inwoners" in zones.attributes


def test_water_zones_dropped_by_default(tmp_path):
    gpkg = tmp_path / "cbs.gpkg"
    make_buurten_gpkg(gpkg, with_water=True)

    zones, report = load_cbs_buurten(gpkg)

    assert zones.n_zones == 2
    assert "BU00010003" not in zones.codes
    assert any("water zone" in w for w in report.warnings)


def test_water_zones_kept_when_disabled(tmp_path):
    gpkg = tmp_path / "cbs.gpkg"
    make_buurten_gpkg(gpkg, with_water=True)

    zones, _ = load_cbs_buurten(gpkg, skip_water=False)

    assert zones.n_zones == 3


def test_secret_sentinel_becomes_nan(tmp_path):
    gpkg = tmp_path / "cbs.gpkg"
    make_buurten_gpkg(gpkg, with_water=False, secret_income=True)

    zones, report = load_cbs_buurten(gpkg)

    idx = zones.code_to_index["BU00010003"]
    assert np.isnan(zones.attributes["aantal_inwoners"][idx])
    assert any("suppressed CBS value" in w for w in report.warnings)


def test_duplicate_codes_raise(tmp_path):
    gpkg = tmp_path / "cbs.gpkg"
    make_buurten_gpkg(gpkg, with_water=False, duplicate_code=True)

    with pytest.raises(ValueError):
        load_cbs_buurten(gpkg)


def test_reprojects_from_geographic_crs(tmp_path):
    gpkg = tmp_path / "cbs.gpkg"
    # Build directly in RD New, then re-save reprojected to WGS84 to
    # simulate a file distributed in lon/lat.
    gdf = make_buurten_gpkg(tmp_path / "rd.gpkg", with_water=False)
    gdf.to_crs("EPSG:4326").to_file(gpkg, driver="GPKG", layer="buurten")

    zones, _ = load_cbs_buurten(gpkg)

    assert zones.crs == "EPSG:28992"
    idx = zones.code_to_index["BU00010001"]
    # round-trip through WGS84 should still land close to (50, 50)
    assert zones.centroid_x[idx] == pytest.approx(50.0, abs=1.0)
    assert zones.centroid_y[idx] == pytest.approx(50.0, abs=1.0)


def test_missing_code_column_raises_with_available_columns(tmp_path):
    gpkg = tmp_path / "cbs.gpkg"
    gdf = geopandas.GeoDataFrame(
        {"not_a_buurtcode": ["x"], "geometry": [_square(0, 0)]},
        crs="EPSG:28992",
    )
    gdf.to_file(gpkg, driver="GPKG", layer="buurten")

    with pytest.raises(KeyError, match="not_a_buurtcode"):
        load_cbs_buurten(gpkg)


def test_explicit_attribute_columns_subset(tmp_path):
    gpkg = tmp_path / "cbs.gpkg"
    make_buurten_gpkg(gpkg, with_water=False)

    zones, _ = load_cbs_buurten(gpkg, attribute_columns=("aantal_inwoners",))

    assert set(zones.attributes) == {"aantal_inwoners"}


def test_unknown_requested_attribute_column_raises(tmp_path):
    gpkg = tmp_path / "cbs.gpkg"
    make_buurten_gpkg(gpkg, with_water=False)

    with pytest.raises(KeyError):
        load_cbs_buurten(gpkg, attribute_columns=("does_not_exist",))
