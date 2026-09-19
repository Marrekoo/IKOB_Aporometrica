"""Car distances in a skim store: routed nearby, detour model beyond."""

import numpy as np
import pandas as pd
import pytest

from ikob2.skims.car import DetourModel
from ikob2.skims.distance import build_car_distance
from ikob2.skims.store import SkimStore


def frames():
    o = pd.DataFrame({"id": ["O0", "O1"], "x": [0.0, 1000.0],
                      "y": [0.0, 0.0], "lon": 5.0, "lat": 52.0})
    near = pd.DataFrame({"id": ["D0", "D1", "D2"],
                         "x": [3000.0, 10000.0, 80000.0], "y": 0.0,
                         "lon": 5.1, "lat": 52.0})
    far = pd.DataFrame({"id": ["C0"], "x": [50000.0], "y": 0.0,
                        "lon": 5.5, "lat": 52.0})
    return o, near, far


def test_routed_inside_radius_detour_outside(tmp_path):
    o, near, far = frames()
    store = SkimStore.create(tmp_path / "s", list(o["id"]),
                             {"near": list(near["id"]), "far": list(far["id"])})
    detour = DetourModel.constant(1.5)
    calls = []

    def fake(orig, dest):
        calls.append(dest.shape[0])
        km = np.full((len(orig), len(dest)), 4.0)      # the "routed" value
        return km, km * 2

    stats = build_car_distance(store, o, {"near": near, "far": far}, detour,
                               fake, radius_km=20.0, origin_batch=1)
    d = store.block("near", "car", "distance")
    # D0 (3 km) and D1 (10 km) are within 20 km of both origins: routed
    assert d[0, 0] == 4.0 and d[1, 1] == 4.0
    # D2 (80 km) is beyond the radius: crow-fly x detour
    assert d[0, 2] == pytest.approx(80.0 * 1.5)
    assert d[1, 2] == pytest.approx(79.0 * 1.5)
    # the far layer never calls the server
    assert store.block("far", "car", "distance")[0, 0] == pytest.approx(75.0)
    assert stats["routed_pairs"] == 4 and stats["fallback_pairs"] == 0
    assert all(n <= 100 for n in calls)


def test_no_route_falls_back_to_detour_and_run_resumes(tmp_path):
    o, near, far = frames()
    store = SkimStore.create(tmp_path / "s", list(o["id"]),
                             {"near": list(near["id"])})
    detour = DetourModel.constant(1.2)

    def fake(orig, dest):
        km = np.full((len(orig), len(dest)), np.nan)
        return km, km

    stats = build_car_distance(store, o, {"near": near}, detour, fake,
                               radius_km=20.0, origin_batch=1)
    assert stats["routed_pairs"] == 0 and stats["fallback_pairs"] == 4
    assert store.block("near", "car", "distance")[0, 0] == pytest.approx(3.6)
    again = build_car_distance(store, o, {"near": near}, detour,
                               lambda *a: pytest.fail("resumed run routed"),
                               radius_km=20.0)
    assert again == {"routed_pairs": 0, "fallback_pairs": 0}


def test_alignment_errors(tmp_path):
    o, near, far = frames()
    store = SkimStore.create(tmp_path / "s", list(o["id"]),
                             {"near": list(near["id"])})
    detour = DetourModel.constant(1.3)
    with pytest.raises(ValueError, match="origins do not match"):
        build_car_distance(store, o.iloc[::-1], {"near": near}, detour,
                           lambda *a: None)
    with pytest.raises(ValueError, match="does not match"):
        build_car_distance(store, o, {"near": near.iloc[::-1]}, detour,
                           lambda *a: None)
