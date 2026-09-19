"""Skim store, zone points, walking times and the block builder."""

import datetime as dt
import os

import numpy as np
import pandas as pd
import pytest

from ikob2.skims.build import build_time_skims
from ikob2.skims.router import R5Router, TimeRequest, _to_matrix
from ikob2.skims.store import SkimStore
from ikob2.skims.walk import intrazonal_distance_m, walk_time_matrix
from ikob2.skims.zones import coarse_cells, zone_points


# ── Store ────────────────────────────────────────────────────────────

def make_store(tmp_path, n_o=5, n_d=4, layers=None):
    origins = [f"O{i}" for i in range(n_o)]
    layers = layers or {"near": [f"D{j}" for j in range(n_d)]}
    return SkimStore.create(tmp_path / "s", origins, layers)


def test_create_open_and_metadata(tmp_path):
    store = make_store(tmp_path)
    again = SkimStore.open(tmp_path / "s")
    assert again.origins == store.origins
    assert again.layer("near").destinations == ("D0", "D1", "D2", "D3")
    with pytest.raises(FileExistsError):
        make_store(tmp_path)
    with pytest.raises(FileNotFoundError):
        SkimStore.open(tmp_path / "nope")
    with pytest.raises(ValueError, match="unique"):
        SkimStore.create(tmp_path / "x", ["a", "a"], {"l": ["d"]})
    with pytest.raises(KeyError, match="No layer"):
        store.layer("far")


def test_write_read_rows_and_lazy_blocks(tmp_path):
    store = make_store(tmp_path)
    vals = np.arange(20, dtype=np.float32).reshape(5, 4)
    store.write_rows("near", "car", "time", 0, vals[:3])
    store.write_rows("near", "car", "time", 3, vals[3:])
    full = store.block("near", "car", "time")
    np.testing.assert_array_equal(full, vals)
    sub = store.block("near", "car", "time", origins=["O3", "O1"],
                      destinations=["D2", "D0"])
    np.testing.assert_array_equal(sub, vals[[3, 1]][:, [2, 0]])
    mm = store.array("near", "car", "time")
    assert isinstance(mm, np.memmap) and not mm.flags.writeable
    assert store.arrays() == [("near", "car", "time")]


def test_unwritten_cells_are_nan_and_fill_replaces_them(tmp_path):
    store = make_store(tmp_path)
    store.write_rows("near", "car", "time", 0, np.ones((2, 4)))
    block = store.block("near", "car", "time")
    assert np.isnan(block[2:]).all() and (block[:2] == 1).all()
    filled = store.block("near", "car", "time", fill=9999.0)
    assert (filled[2:] == 9999.0).all()


def test_progress_survives_reopen_and_resume_blocks(tmp_path):
    store = make_store(tmp_path, n_o=10)
    store.write_rows("near", "car", "time", 0, np.zeros((4, 4)))
    store.write_rows("near", "car", "time", 4, np.zeros((2, 4)))
    reopened = SkimStore.open(tmp_path / "s")
    assert reopened.done_rows("near", "car", "time") == [[0, 6]]
    assert reopened.pending_blocks("near", "car", "time", 3) == [(6, 9), (9, 10)]


def test_write_rows_validation(tmp_path):
    store = make_store(tmp_path)
    with pytest.raises(ValueError, match="shape"):
        store.write_rows("near", "car", "time", 0, np.ones((2, 3)))
    with pytest.raises(ValueError, match="outside"):
        store.write_rows("near", "car", "time", 4, np.ones((3, 4)))
    with pytest.raises(FileNotFoundError):
        store.array("near", "bike", "time")
    store.allocate("near", "car", "time")
    with pytest.raises(KeyError, match="unknown origin"):
        store.block("near", "car", "time", origins=["nope"])


def test_near_far_combination(tmp_path):
    # far layer = two cells; buurten D1..D5, D0/D1 also routed near
    origins = ["O0", "O1"]
    near = ["D0", "D1"]
    far = ["C_a", "C_b"]
    cell_of = {"far": {"D0": "C_a", "D1": "C_a", "D2": "C_a", "D3": "C_b",
                       "D4": "C_b"}}
    store = SkimStore.create(tmp_path / "s", origins,
                             {"near": near, "far": far}, cell_of=cell_of)
    store.write_rows("near", "car", "time", 0, [[1, 2], [3, 4]])
    store.write_rows("far", "car", "time", 0, [[10, 20], [30, 40]])
    out = store.combined("car", "time", ["D3", "D0", "D2", "D1"],
                         near="near", far="far")
    np.testing.assert_array_equal(out, [[20, 1, 10, 2], [40, 3, 30, 4]])
    only_o1 = store.combined("car", "time", ["D4"], near="near", far="far",
                             origins=["O1"])
    np.testing.assert_array_equal(only_o1, [[40]])
    with pytest.raises(KeyError, match="neither"):
        store.combined("car", "time", ["D9"], near="near", far="far")
    with pytest.raises(ValueError, match="no cell_of"):
        store.combined("car", "time", ["D0"], near="far", far="near")


def test_cell_of_must_point_into_the_layer(tmp_path):
    with pytest.raises(ValueError, match="not in the layer"):
        SkimStore.create(tmp_path / "s", ["O"], {"far": ["C"]},
                         cell_of={"far": {"D": "Z"}})


# ── Zones ────────────────────────────────────────────────────────────

def test_zone_points_rd_to_wgs84():
    # The RD New origin (Amersfoort tower) is 52.155172 N, 5.387206 E
    z = zone_points(["A"], [155000.0], [463000.0], area_m2=[1e6])
    assert z.loc[0, "lon"] == pytest.approx(5.387206, abs=2e-5)
    assert z.loc[0, "lat"] == pytest.approx(52.155172, abs=2e-5)
    assert z.loc[0, "area_m2"] == 1e6
    with pytest.raises(ValueError, match="unique"):
        zone_points(["A", "A"], [1, 2], [3, 4])


def test_coarse_cells_weighted_mean_and_mapping():
    cells, cell_of = coarse_cells(
        ["b1", "b2", "b3"], [0.0, 100.0, 500.0], [0.0, 0.0, 0.0],
        ["G1", "G1", "G2"], weights=[1.0, 3.0, 0.0])
    g1 = cells[cells["id"] == "G1"].iloc[0]
    assert g1["x"] == pytest.approx(75.0)               # weighted
    assert cells[cells["id"] == "G2"].iloc[0]["x"] == pytest.approx(500.0)
    assert cell_of == {"b1": "G1", "b2": "G1", "b3": "G2"}
    unweighted, _ = coarse_cells(["a", "b"], [0.0, 100.0], [0, 0], ["G", "G"])
    assert unweighted.iloc[0]["x"] == pytest.approx(50.0)
    zeros, _ = coarse_cells(["a", "b"], [0.0, 100.0], [0, 0], ["G", "G"],
                            weights=[0, 0])
    assert zeros.iloc[0]["x"] == pytest.approx(50.0)


# ── Walking ──────────────────────────────────────────────────────────

def test_walk_time_from_distance_detour_and_speed():
    t = walk_time_matrix([[0.0, 0.0]], [[0.0, 3000.0], [4000.0, 0.0]],
                         speed_kmh=6.0, detour=1.25, max_minutes=None)
    # 3000 m * 1.25 = 3750 m at 100 m/min = 37.5 min
    np.testing.assert_allclose(t, [[37.5, 50.0]], rtol=1e-6)
    limited = walk_time_matrix([[0.0, 0.0]], [[0.0, 3000.0], [4000.0, 0.0]],
                               speed_kmh=6.0, detour=1.25, max_minutes=40)
    assert limited[0, 0] == pytest.approx(37.5) and np.isnan(limited[0, 1])


def test_intrazonal_walk_uses_zone_area():
    area = 1_000_000.0                          # 1 km2 -> 520 m mean
    assert intrazonal_distance_m(area) == pytest.approx(520.0)
    t = walk_time_matrix([[0.0, 0.0], [5000.0, 0.0]],
                         [[0.0, 0.0], [5000.0, 0.0]],
                         origin_codes=["a", "b"], dest_codes=["a", "b"],
                         origin_area_m2=[area, area], speed_kmh=4.8,
                         detour=1.0, max_minutes=None)
    assert t[0, 0] == pytest.approx(520.0 / 80.0)       # 6.5 min
    assert t[1, 1] == pytest.approx(6.5) and t[0, 1] > 60
    with pytest.raises(ValueError, match="origin_area_m2"):
        walk_time_matrix([[0.0, 0.0]], [[0.0, 0.0]], origin_codes=["a"],
                         dest_codes=["a"])
    with pytest.raises(ValueError, match="positive"):
        walk_time_matrix([[0.0, 0.0]], [[1.0, 1.0]], speed_kmh=0)


# ── Requests and matrix assembly ─────────────────────────────────────

def test_time_request_validation():
    assert TimeRequest("car").max_minutes == 120
    with pytest.raises(ValueError, match="Unknown mode"):
        TimeRequest("boat")
    with pytest.raises(ValueError, match="departure"):
        TimeRequest("pt")
    with pytest.raises(ValueError, match="percentile"):
        TimeRequest("car", percentile=0)
    TimeRequest("pt", departure=dt.datetime(2026, 9, 1, 8, 0))


def test_to_matrix_places_pairs_and_leaves_missing_as_nan():
    ttm = pd.DataFrame({"from_id": ["a", "a", "b", "z"],
                        "to_id": ["x", "y", "y", "x"],
                        "travel_time": [5.0, 7.0, np.nan, 1.0]})
    m = _to_matrix(ttm, "travel_time", ["a", "b"], ["x", "y", "w"])
    np.testing.assert_array_equal(np.isnan(m), [[0, 0, 1], [1, 1, 1]])
    assert m[0, 0] == 5 and m[0, 1] == 7


# ── Builder ──────────────────────────────────────────────────────────

class FakeRouter:
    """time = 10 * (origin index + 1) + destination index, per mode."""
    def __init__(self):
        self.calls = []

    def time_matrix(self, origins, destinations, request):
        self.calls.append((request.mode, len(origins), len(destinations)))
        offset = {"car": 0.0, "bike": 1000.0}[request.mode]
        oi = origins["id"].str[1:].astype(int).to_numpy()
        di = destinations["id"].str[-1:].astype(int).to_numpy()
        return offset + 10.0 * (oi[:, None] + 1) + di[None, :]


def _frames(n_o, dests):
    o = pd.DataFrame({"id": [f"O{i}" for i in range(n_o)],
                      "lon": 5.0, "lat": 52.0})
    d = pd.DataFrame({"id": dests, "lon": 5.1, "lat": 52.1})
    return o, d


def test_builder_fills_all_modes_in_blocks_and_resumes(tmp_path):
    o, d = _frames(7, ["D0", "D1", "D2"])
    store = SkimStore.create(tmp_path / "s", list(o["id"]),
                             {"near": list(d["id"])})
    reqs = {"car": TimeRequest("car"), "bike": TimeRequest("bike")}
    router = FakeRouter()
    build_time_skims(router, store, o, {"near": d}, reqs, block_size=3)
    assert len(router.calls) == 2 * 3                 # 3 blocks per mode
    car = store.block("near", "car", "time")
    assert car[2, 1] == 31 and car[6, 0] == 70
    assert store.block("near", "bike", "time")[0, 0] == 1010

    again = FakeRouter()
    build_time_skims(again, store, o, {"near": d}, reqs, block_size=3)
    assert again.calls == []                          # nothing left to do


def test_builder_resumes_after_interruption(tmp_path):
    o, d = _frames(6, ["D0", "D1"])
    store = SkimStore.create(tmp_path / "s", list(o["id"]),
                             {"near": list(d["id"])})

    class Flaky(FakeRouter):
        def time_matrix(self, origins, destinations, request):
            if len(self.calls) == 2:
                raise RuntimeError("router died")
            return super().time_matrix(origins, destinations, request)

    with pytest.raises(RuntimeError):
        build_time_skims(Flaky(), store, o, {"near": d},
                         {"car": TimeRequest("car")}, block_size=2)
    reopened = SkimStore.open(tmp_path / "s")
    assert reopened.done_rows("near", "car", "time") == [[0, 4]]
    resume = FakeRouter()
    build_time_skims(resume, reopened, o, {"near": d},
                     {"car": TimeRequest("car")}, block_size=2)
    assert resume.calls == [("car", 2, 2)]
    assert not np.isnan(reopened.block("near", "car", "time")).any()


def test_builder_checks_alignment(tmp_path):
    o, d = _frames(3, ["D0", "D1"])
    store = SkimStore.create(tmp_path / "s", list(o["id"]),
                             {"near": list(d["id"])})
    with pytest.raises(ValueError, match="origins do not match"):
        build_time_skims(FakeRouter(), store, o.iloc[::-1], {"near": d},
                         {"car": TimeRequest("car")})
    with pytest.raises(ValueError, match="destinations do not match"):
        build_time_skims(FakeRouter(), store, o, {"near": d.iloc[::-1]},
                         {"car": TimeRequest("car")})
    with pytest.raises(ValueError, match="has mode"):
        build_time_skims(FakeRouter(), store, o, {"near": d},
                         {"car": TimeRequest("bike")})


# ── Real routing on the development extract (optional) ───────────────

OSM = os.environ.get("IKOB_SKIM_DEV_OSM", "")


@pytest.mark.skipif(not os.path.exists(OSM),
                    reason="set IKOB_SKIM_DEV_OSM to an OSM .pbf (Java 21 "
                           "and r5py required)")
def test_r5_car_times_are_plausible():
    # Utrecht Dom area to Utrecht Centraal and to Amersfoort
    pts = pd.DataFrame({"id": ["dom", "cs", "amf"],
                        "lon": [5.1214, 5.1104, 5.3878],
                        "lat": [52.0907, 52.0894, 52.1552]})
    router = R5Router(OSM)
    car = router.time_matrix(pts.iloc[:1], pts, TimeRequest("car"))
    bike = router.time_matrix(pts.iloc[:1], pts, TimeRequest("bike"))
    assert car[0, 0] <= 3                       # same point
    assert 2 <= car[0, 1] <= 12                 # ~1 km through the city
    assert 15 <= car[0, 2] <= 45                # ~25 km to Amersfoort
    assert bike[0, 2] > car[0, 2]               # cycling is slower
