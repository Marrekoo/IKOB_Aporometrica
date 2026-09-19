"""Regional GTFS subset for the local OTP server."""

import zipfile

import pandas as pd
import pytest

from ikob2.skims.gtfs_subset import subset_gtfs


def make(tmp_path):
    z = tmp_path / "full.zip"
    stops = pd.DataFrame({
        "stop_id": ["in1", "in2", "out1", "st"],
        "stop_lat": [52.0, 52.05, 50.0, 52.0],
        "stop_lon": [5.0, 5.1, 6.0, 5.0],
        "parent_station": [None, None, None, None]})
    routes = pd.DataFrame({"route_id": ["r1", "r2"], "agency_id": ["a", "b"],
                           "route_type": [2, 3]})
    trips = pd.DataFrame({
        "route_id": ["r1", "r1", "r2", "r1"],
        "service_id": ["S", "S", "S", "OTHER"],
        "trip_id": ["t_in", "t_cross", "t_out", "t_wrongday"]})
    st = pd.DataFrame([
        ("t_in", 1, "in1"), ("t_in", 2, "in2"),
        ("t_cross", 1, "in1"), ("t_cross", 2, "out1"),
        ("t_out", 1, "out1"), ("t_out", 2, "out1"),
        ("t_wrongday", 1, "in1"), ("t_wrongday", 2, "in2")],
        columns=["trip_id", "stop_sequence", "stop_id"])
    st["arrival_time"] = st["departure_time"] = "08:00:00"
    with zipfile.ZipFile(z, "w") as zf:
        for name, df in (("stops.txt", stops), ("routes.txt", routes),
                         ("trips.txt", trips), ("stop_times.txt", st),
                         ("agency.txt", pd.DataFrame(
                             {"agency_id": ["a", "b"], "agency_name": ["A", "B"]})),
                         ("calendar_dates.txt", pd.DataFrame(
                             {"service_id": ["S", "OTHER"],
                              "date": ["20260915", "20260916"],
                              "exception_type": [1, 1]}))):
            zf.writestr(name, df.to_csv(index=False))
    return z


def test_subset_keeps_region_trips_and_dates(tmp_path):
    src = make(tmp_path)
    dst = tmp_path / "sub.zip"
    counts = subset_gtfs(src, dst, (4.5, 51.5, 5.5, 52.5), ["2026-09-15"])
    with zipfile.ZipFile(dst) as z:
        stops = pd.read_csv(z.open("stops.txt"), dtype=str)
        trips = pd.read_csv(z.open("trips.txt"), dtype=str)
        st = pd.read_csv(z.open("stop_times.txt"), dtype=str)
        routes = pd.read_csv(z.open("routes.txt"), dtype=str)
        cd = pd.read_csv(z.open("calendar_dates.txt"), dtype=str)
    # t_cross has only one stop inside the box, t_out none, t_wrongday runs
    # on another date: only t_in survives
    assert list(trips["trip_id"]) == ["t_in"]
    assert set(stops["stop_id"]) == {"in1", "in2"}
    assert len(st) == 2 and list(routes["route_id"]) == ["r1"]
    assert list(cd["service_id"]) == ["S"]
    assert counts == {"stops": 2, "trips": 1, "stop_times": 2, "routes": 1}


def test_no_service_on_the_dates(tmp_path):
    with pytest.raises(ValueError, match="No service"):
        subset_gtfs(make(tmp_path), tmp_path / "x.zip",
                    (4.5, 51.5, 5.5, 52.5), ["2026-09-20"])
