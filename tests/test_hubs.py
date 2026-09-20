"""Hub locations from files, and the router's bicycle egress through them."""

import json

import numpy as np
import pytest

from ikob2.skims.gtfs_pt import LegSpec, PtRouter, load_peak_timetable
from ikob2.skims.hubs import hub_xy, load_hubs, resolve_path
from tests.test_gtfs_pt import gtfs_zip, xy


def test_csv_and_ovfiets_json_are_read(tmp_path):
    (tmp_path / "h.csv").write_text("hub,lat,lon\nA,52.0,5.0\nB,52.1,5.1\n")
    (tmp_path / "o.json").write_text(json.dumps({"locaties": {
        "x1": {"name": "OV X", "lat": 52.2, "lng": 5.2}}}))
    h = load_hubs([tmp_path / "h.csv", tmp_path / "o.json"])
    assert list(h["hub"]) == ["A", "B", "OV X"]
    assert list(h["source"]) == ["h.csv", "h.csv", "o.json"]
    assert hub_xy(h).shape == (3, 2)
    assert 100_000 < h["x"].iloc[0] < 200_000          # RD New metres


def test_hubs_without_coordinates_are_an_error(tmp_path):
    (tmp_path / "h.csv").write_text("hub,lat,lon\nA,52.0,5.0\nB,,\n")
    with pytest.raises(ValueError, match="without coordinates"):
        load_hubs([tmp_path / "h.csv"])
    (tmp_path / "n.csv").write_text("a,b\n1,2\n")
    with pytest.raises(ValueError, match="lat and lon"):
        load_hubs([tmp_path / "n.csv"])


def test_relative_paths_fall_back_to_the_inputs_folder(tmp_path):
    (tmp_path / "hubs").mkdir()
    (tmp_path / "hubs" / "h.csv").write_text("lat,lon\n52,5\n")
    assert resolve_path("hubs/h.csv", tmp_path) == tmp_path / "hubs" / "h.csv"
    with pytest.raises(FileNotFoundError):
        resolve_path("nope.csv", tmp_path)


def egress(tmp_path, hubs, dest, **kw):
    tt = load_peak_timetable(gtfs_zip(tmp_path), "2026-09-15")
    router = PtRouter(tt, hubs_xy=hubs, **kw)
    return router.journeys(np.array([xy(0, 200)]), np.array(dest),
                           max_minutes=180,
                           egress=LegSpec(hubs_only=True, fixed_minutes=1.0))


def test_egress_uses_the_hubs_of_the_file_not_the_rail_stops(tmp_path):
    far_from_rail = [xy(15000, 2500)]          # only the bus stop D is near
    # no hubs near the destination: not reachable, although rail stop C
    # would be a hub under the default rule
    assert np.isnan(egress(tmp_path, np.array([xy(0, 5000)]),
                           far_from_rail)["time"]).all()
    # a hub 100 m from bus stop D makes it reachable
    j = egress(tmp_path, np.array([xy(15000, 100)]), far_from_rail)
    assert np.isfinite(j["time"]).all()
    ride = 2400 * 1.3 / (16000 / 60)
    assert j["egress_min"][0, 0] == pytest.approx(ride, abs=0.1)
    # time: alight at D, walk 100 m to the hub, ride, 1 minute fixed
    direct = egress(tmp_path, np.array([xy(15000, 0)]), far_from_rail)
    walk = 100 * 1.3 / (4000 / 60)
    assert j["time"][0, 0] - direct["time"][0, 0] == pytest.approx(
        walk + ride + 1 - (2500 * 1.3 / (16000 / 60) + 1), abs=0.2)


def test_a_hub_out_of_walking_reach_of_any_stop_is_unused(tmp_path):
    far = [xy(15000, 2500)]
    assert np.isnan(egress(tmp_path, np.array([xy(15000, 1200)]),
                           far)["time"]).all()       # 1.2 km from any stop
    j = egress(tmp_path, np.array([xy(15000, 1200)]), far,
               hub_walk_radius_m=1500.0)
    assert np.isfinite(j["time"]).all()


def test_the_fastest_hub_wins_per_stop(tmp_path):
    far = [xy(15000, 2500)]
    h1, h2 = xy(15000, 100), xy(15000, 250)      # both within 300 m of stop D
    both = egress(tmp_path, np.array([h1, h2]), far)["time"][0, 0]
    t1 = egress(tmp_path, np.array([h1]), far)["time"][0, 0]
    t2 = egress(tmp_path, np.array([h2]), far)["time"][0, 0]
    assert t1 != pytest.approx(t2, abs=0.05)
    assert both == pytest.approx(min(t1, t2), abs=0.01)
