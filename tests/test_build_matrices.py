"""Car matrices from a store: routed distances win over the detour model."""

import numpy as np
import pytest

from ikob2.cli.accessibility import build_matrices
from ikob2.domain.zones import ZoneSet
from ikob2.skims.car import CarCostModel, DetourModel
from ikob2.skims.store import SkimStore


def make(tmp_path, with_distance):
    codes = np.array(["Z0", "Z1", "Z2"])
    zones = ZoneSet(codes=codes, names=codes,
                    centroid_x=np.array([0.0, 10000.0, 20000.0]),
                    centroid_y=np.zeros(3), crs="EPSG:28992",
                    municipality_code=np.array(["G", "G", "G"]),
                    attributes={"stedelijkheid_adressen_per_km2":
                                np.array([5.0, 5.0, 5.0])})
    tmp_path.mkdir(parents=True, exist_ok=True)
    store = SkimStore.create(
        tmp_path / "s", ["Z0"], {"near": ["Z0", "Z1", "Z2"], "far": ["G"]},
        cell_of={"far": {"Z0": "G", "Z1": "G", "Z2": "G"}})
    store.write_rows("near", "car", "time", 0, [[0.0, 12.0, 25.0]])
    store.write_rows("far", "car", "time", 0, [[10.0]])
    if with_distance:
        store.write_rows("near", "car", "distance", 0, [[0.0, 11.0, 30.0]])
        store.write_rows("far", "car", "distance", 0, [[15.0]])
    return store, zones


def test_routed_distance_is_used_when_the_store_has_it(tmp_path):
    store, zones = make(tmp_path / "x", True)
    m, _ = build_matrices(store, zones, ["car"], detour=DetourModel.constant(2.0),
                          car_model=CarCostModel(0.10), parking_search=False)
    # cost = 0.10 x routed km (11 and 30), not 0.10 x 2.0 x crow-fly (20, 40)
    np.testing.assert_allclose(m["car"].cost[0], [0.0, 1.1, 3.0], rtol=1e-5)


def test_detour_model_is_the_fallback(tmp_path):
    store, zones = make(tmp_path / "x", False)
    m, _ = build_matrices(store, zones, ["car"], detour=DetourModel.constant(2.0),
                          car_model=CarCostModel(0.10), parking_search=False)
    np.testing.assert_allclose(m["car"].cost[0], [0.0, 2.0, 4.0], rtol=1e-5)
    assert m["car"].time[0, 2] == pytest.approx(25.0)


def test_unsupported_mode_stops(tmp_path):
    store, zones = make(tmp_path / "x", False)
    with pytest.raises(SystemExit, match="no time margin"):
        build_matrices(store, zones, ["walk"], detour=DetourModel.constant(1.3),
                       car_model=CarCostModel(), parking_search=False)


def test_distance_store_supplies_distances_to_a_scenario_store(tmp_path):
    base, zones = make(tmp_path / "base", True)
    peak = SkimStore.create(
        tmp_path / "peak", ["Z0"], {"near": ["Z0", "Z1", "Z2"], "far": ["G"]},
        cell_of={"far": {"Z0": "G", "Z1": "G", "Z2": "G"}})
    peak.write_rows("near", "car", "time", 0, [[0.0, 15.0, 30.0]])
    peak.write_rows("far", "car", "time", 0, [[12.0]])
    m, _ = build_matrices(peak, zones, ["car"], detour=DetourModel.constant(2.0),
                          car_model=CarCostModel(0.10), parking_search=False,
                          distance_store=base)
    assert m["car"].time[0, 2] == pytest.approx(30.0)          # peak times
    np.testing.assert_allclose(m["car"].cost[0], [0.0, 1.1, 3.0], rtol=1e-5)
    other = SkimStore.create(tmp_path / "other", ["Z9"], {"near": ["Z0"],
                                                          "far": ["G"]})
    with pytest.raises(ValueError, match="differ"):
        build_matrices(peak, zones, ["car"], detour=DetourModel.constant(2.0),
                       car_model=CarCostModel(0.10), parking_search=False,
                       distance_store=other)
