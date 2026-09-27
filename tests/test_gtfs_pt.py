"""Frequency-model public transport skims from GTFS."""

import zipfile

import numpy as np
import pandas as pd
import pytest
from pyproj import Transformer

from ikob2.skims.gtfs_pt import (
    PtRouter,
    boarding_wait,
    load_peak_timetable,
)

X0, Y0 = 155000.0, 463000.0
TO_WGS = Transformer.from_crs("EPSG:28992", "EPSG:4326", always_xy=True)


def lonlat(dx, dy):
    lon, lat = TO_WGS.transform(X0 + dx, Y0 + dy)
    return lon, lat


def hms(seconds):
    return f"{int(seconds // 3600):02d}:{int(seconds % 3600 // 60):02d}:00"


def gtfs_zip(tmp_path, *, l1_headway_min=10, l2_headway_min=30,
             date="20260915", pickup_b=0):
    """Line R1: A - B - C every l1 minutes (6 min per hop); line R2: C' - D
    every l2 minutes (10 min); C' is 100 m from C. 07:00-09:00 plus a lone
    12:00 trip that must not count; service S1 runs on `date` only."""
    stops = {"A": (0, 0), "B": (5000, 0), "C": (10000, 0),
             "Cp": (10000, 100), "D": (15000, 0)}
    stop_rows = [(k, *lonlat(*v)[::-1]) for k, v in stops.items()]
    trips, times = [], []

    def add_line(route, stop_seq, hops_min, headway_min, tag):
        t0 = 7 * 3600
        n = 0
        starts = list(range(t0, 9 * 3600, int(headway_min * 60))) + [12 * 3600]
        for s in starts:
            tid = f"{tag}{n}"
            n += 1
            trips.append((route, "S1", tid, "0"))
            t = s
            for i, stop in enumerate(stop_seq):
                pick = pickup_b if (route == "R1" and stop == "B") else 0
                times.append((tid, i + 1, stop, hms(t), hms(t), pick, 0))
                if i < len(hops_min):
                    t += hops_min[i] * 60

    add_line("R1", ["A", "B", "C"], [6, 6], l1_headway_min, "a")
    add_line("R2", ["Cp", "D"], [10], l2_headway_min, "b")
    trips.append(("R1", "OTHER", "x0", "0"))
    times.append(("x0", 1, "A", "07:30:00", "07:30:00", 0, 0))
    times.append(("x0", 2, "B", "07:36:00", "07:36:00", 0, 0))

    z = tmp_path / "gtfs.zip"
    with zipfile.ZipFile(z, "w") as zf:
        def put(name, df):
            zf.writestr(name, df.to_csv(index=False))
        put("stops.txt", pd.DataFrame(stop_rows,
                                      columns=["stop_id", "stop_lat", "stop_lon"]))
        put("routes.txt", pd.DataFrame([("R1", 2), ("R2", 3)],
                                       columns=["route_id", "route_type"]))
        put("trips.txt", pd.DataFrame(
            trips, columns=["route_id", "service_id", "trip_id",
                            "direction_id"]))
        put("stop_times.txt", pd.DataFrame(
            times, columns=["trip_id", "stop_sequence", "stop_id",
                            "arrival_time", "departure_time", "pickup_type",
                            "drop_off_type"]))
        put("calendar_dates.txt", pd.DataFrame(
            [("S1", date, 1), ("OTHER", "20260916", 1)],
            columns=["service_id", "date", "exception_type"]))
    return z


def xy(dx, dy):
    return [X0 + dx, Y0 + dy]


ORIGIN = np.array([xy(0, 200)])
DESTS = np.array([xy(5000, 100), xy(15000, 150), xy(30000, 0), xy(0, 300)])


# ── waiting rule ─────────────────────────────────────────────────────

def test_boarding_wait_is_half_the_headway_capped_at_7_5():
    h = [5, 10, 14.9, 15, 30, 120]
    np.testing.assert_allclose(boarding_wait(h), [2.5, 5, 7.45, 7.5, 7.5, 7.5])
    np.testing.assert_allclose(boarding_wait([30], cap_min=10), [10])


# ── reading GTFS ─────────────────────────────────────────────────────

def test_peak_timetable_headways_rides_and_service_filter(tmp_path):
    tt = load_peak_timetable(gtfs_zip(tmp_path), "2026-09-15")
    line = tt.lines.set_index("route_id")
    assert set(line.index) == {"R1", "R2"}                 # OTHER not running
    assert line.loc["R1", "route_type"] == "2"
    r1 = int(line.loc["R1", "line_id"])
    ls = tt.line_stops
    h = ls[(ls.line_id == r1) & (ls.stop_id == "A")].iloc[0]
    assert h["headway_min"] == pytest.approx(10.0)         # 12 departures / 2 h
    r2 = int(line.loc["R2", "line_id"])
    assert ls[(ls.line_id == r2) & (ls.stop_id == "Cp")].iloc[0][
        "headway_min"] == pytest.approx(30.0)              # 4 departures
    rides = tt.rides
    ab = rides[(rides.line_id == r1) & (rides.from_stop == "A")
               & (rides.to_stop == "B")]
    assert ab["minutes"].iloc[0] == pytest.approx(6.0)
    # the 12:00 trip is outside the window: not counted in headways
    assert (ls[ls.line_id == r1].groupby("stop_id").size() == 1).all()


def test_no_service_on_the_date_and_calendar_txt(tmp_path):
    with pytest.raises(ValueError, match="No service"):
        load_peak_timetable(gtfs_zip(tmp_path), "2026-09-20")
    # calendar.txt weekdays and ranges are honoured, exceptions applied
    src = gtfs_zip(tmp_path)
    z = tmp_path / "cal.zip"
    with zipfile.ZipFile(src) as a, zipfile.ZipFile(z, "w") as b:
        for name in a.namelist():
            if name != "calendar_dates.txt":
                b.writestr(name, a.read(name))
        b.writestr("calendar.txt", pd.DataFrame(
            [("S1", 1, 1, 1, 1, 1, 0, 0, "20260901", "20261031")],
            columns=["service_id", "monday", "tuesday", "wednesday",
                     "thursday", "friday", "saturday", "sunday",
                     "start_date", "end_date"]).to_csv(index=False))
        b.writestr("calendar_dates.txt", pd.DataFrame(
            [("S1", "20260922", 2)],
            columns=["service_id", "date", "exception_type"]
        ).to_csv(index=False))
    assert load_peak_timetable(z, "2026-09-15").lines.shape[0] == 2   # Tuesday
    with pytest.raises(ValueError, match="No service"):
        load_peak_timetable(z, "2026-09-19")          # Saturday
    with pytest.raises(ValueError, match="No service"):
        load_peak_timetable(z, "2026-09-22")          # removed by exception


def test_pickup_restriction_blocks_boarding(tmp_path):
    tt = load_peak_timetable(gtfs_zip(tmp_path, pickup_b=1), "2026-09-15")
    ls = tt.line_stops
    r1 = int(tt.lines.set_index("route_id").loc["R1", "line_id"])
    b = ls[(ls.line_id == r1) & (ls.stop_id == "B")].iloc[0]
    assert not b["can_board"] and b["can_alight"]


# ── routing ──────────────────────────────────────────────────────────

def route(tmp_path, **kw):
    tt = load_peak_timetable(gtfs_zip(tmp_path,
                                      **{k: v for k, v in kw.items()
                                         if k in ("l1_headway_min",
                                                  "l2_headway_min")}),
                             "2026-09-15")
    opts = {k: v for k, v in kw.items()
            if k not in ("l1_headway_min", "l2_headway_min")}
    return PtRouter(tt, **opts).time_matrix(ORIGIN, DESTS, max_minutes=180)[0]


def walk(m, kmh=4.0, detour=1.3):
    return m * detour / (kmh * 1000 / 60)


def test_door_to_door_time_adds_walk_wait_ride_transfer_and_egress(tmp_path):
    t = route(tmp_path)
    # A -> B: access 200 m, wait 5 (10-min line), ride 6, egress 100 m
    assert t[0] == pytest.approx(walk(200) + 5 + 6 + walk(100), abs=1e-4)
    # to D: ride 12 to C, walk 100 m to C', wait 7.5 (30-min line, capped),
    # ride 10, egress 150 m; no transfer penalty
    assert t[1] == pytest.approx(walk(200) + 5 + 12 + walk(100) + 7.5 + 10
                                 + walk(150), abs=1e-4)
    assert np.isnan(t[2])                   # 30 km away: not served
    assert np.isnan(t[3])                   # 300 m walk: never a walk-only trip


def test_infrequent_lines_wait_the_capped_average(tmp_path):
    t60 = route(tmp_path, l1_headway_min=60)
    t20 = route(tmp_path, l1_headway_min=20)
    assert t60[0] == pytest.approx(walk(200) + 7.5 + 6 + walk(100), abs=1e-4)
    assert t20[0] == pytest.approx(t60[0], abs=1e-4)          # both capped
    t4 = route(tmp_path, l1_headway_min=4)                    # 2 min wait
    assert t4[0] == pytest.approx(walk(200) + 2 + 6 + walk(100), abs=1e-4)


def test_walking_speed_is_adjustable_and_default_4_kmh(tmp_path):
    slow = route(tmp_path)                                    # 4 km/h default
    fast = route(tmp_path, walk_kmh=6.0)
    assert slow[0] - fast[0] == pytest.approx(
        walk(200) + walk(100) - walk(200, 6) - walk(100, 6), abs=1e-4)
    with pytest.raises(ValueError, match="positive"):
        PtRouter(load_peak_timetable(gtfs_zip(tmp_path), "2026-09-15"),
                 walk_kmh=0)


def test_access_radius_follows_walking_speed_and_boarding_penalty(tmp_path):
    tt = load_peak_timetable(gtfs_zip(tmp_path), "2026-09-15")
    far_origin = np.array([xy(0, 1500)])          # 1.5 km from stop A
    slow = PtRouter(tt, walk_kmh=4.0).time_matrix(far_origin, DESTS[:1])
    fast = PtRouter(tt, walk_kmh=6.0).time_matrix(far_origin, DESTS[:1])
    assert np.isnan(slow[0, 0]) and np.isfinite(fast[0, 0])   # 20 min limit
    base = PtRouter(tt).time_matrix(ORIGIN, DESTS)[0]
    pen = PtRouter(tt, boarding_penalty_min=3.0).time_matrix(ORIGIN, DESTS)[0]
    assert pen[0] - base[0] == pytest.approx(3.0)             # one boarding
    assert pen[1] - base[1] == pytest.approx(6.0)             # two boardings


def test_max_minutes_limits_the_search(tmp_path):
    tt = load_peak_timetable(gtfs_zip(tmp_path), "2026-09-15")
    t = PtRouter(tt).time_matrix(ORIGIN, DESTS, max_minutes=30)[0]
    assert np.isfinite(t[0]) and np.isnan(t[1])               # 43 min > 30


# ── the store layer ──────────────────────────────────────────────────

from ikob2.skims.pt_build import build_pt_layer  # noqa: E402
from ikob2.skims.store import SkimStore  # noqa: E402


def test_pt_layer_in_a_store_resumes_and_matches_the_router(tmp_path):
    tt = load_peak_timetable(gtfs_zip(tmp_path), "2026-09-15")
    router = PtRouter(tt)
    origins = np.array([xy(0, 200), xy(0, 250), xy(5000, 50)])
    store = SkimStore.create(tmp_path / "s", ["O0", "O1", "O2"],
                             {"near": ["Z0"]})
    codes = ["Z0", "Z1", "Z2", "Z3"]
    build_pt_layer(store, router, origins, codes, DESTS, block_size=2)
    got = store.block("all", "pt", "time")
    exp = router.time_matrix(origins, DESTS)
    np.testing.assert_allclose(got, exp, equal_nan=True)
    assert store.combined("pt", "time", ["Z1", "Z0"], near="all")[0, 0] \
        == pytest.approx(exp[0, 1])
    assert store.done_rows("all", "pt", "time") == [[0, 3]]
    # a second call has nothing to do
    build_pt_layer(store, router, origins, codes, DESTS, block_size=2)
    with pytest.raises(ValueError, match="different destinations"):
        build_pt_layer(store, router, origins, codes[::-1], DESTS[::-1])
    with pytest.raises(ValueError, match="origin_xy"):
        build_pt_layer(store, router, origins[:2], codes, DESTS,
                       layer="other")
    with pytest.raises(ValueError, match="already exists"):
        store.add_layer("all", codes)


# ── journeys: distances and boardings for fares ──────────────────────

from ikob2.skims.pt_fare import PtFareModel  # noqa: E402


def journeys(tmp_path, **kw):
    """R1 (rail, A-B-C every 10 min) and R2 (bus, C'-D every 30 min)."""
    tt = load_peak_timetable(gtfs_zip(tmp_path), "2026-09-15")
    return PtRouter(tt, **kw).journeys(ORIGIN, DESTS, max_minutes=180)


def test_journeys_match_time_matrix_and_track_kilometres(tmp_path):
    j = journeys(tmp_path)
    t = route(tmp_path)
    np.testing.assert_allclose(j["time"][0], t, equal_nan=True, rtol=1e-6)
    # to B: 5 km of rail, no regional boarding
    assert j["rail_km"][0, 0] == pytest.approx(5.0 * 1.15, rel=1e-4)
    assert j["other_km"][0, 0] == 0 and j["other_boardings"][0, 0] == 0
    # to D: rail A-C (10 km) then the bus C'-D (5 km) with one boarding
    assert j["rail_km"][0, 1] == pytest.approx(10.0 * 1.15, rel=1e-3)
    assert j["other_km"][0, 1] == pytest.approx(np.hypot(5000, 100) / 1000
                                                * 1.25, rel=1e-3)
    assert j["other_boardings"][0, 1] == 1
    assert np.isnan(j["rail_km"][0, 2]) and np.isnan(j["other_km"][0, 3])


def test_detour_factors_scale_the_kilometres(tmp_path):
    a = journeys(tmp_path)
    b = journeys(tmp_path, rail_detour=1.0, other_detour=1.5)
    assert b["rail_km"][0, 0] == pytest.approx(5.0, rel=1e-4)
    assert b["other_km"][0, 1] == pytest.approx(
        a["other_km"][0, 1] / 1.25 * 1.5, rel=1e-4)
    with pytest.raises(ValueError, match=">= 1"):
        PtRouter(load_peak_timetable(gtfs_zip(tmp_path), "2026-09-15"),
                 rail_detour=0.9)


def test_journeys_over_several_origins_and_chunks(tmp_path):
    tt = load_peak_timetable(gtfs_zip(tmp_path), "2026-09-15")
    origins = np.array([xy(0, 200), xy(0, 250), xy(5000, 50), xy(9900, 0)])
    r = PtRouter(tt)
    a = r.journeys(origins, DESTS, chunk=1)
    b = r.journeys(origins, DESTS, chunk=4)
    for k in a:
        np.testing.assert_allclose(a[k], b[k], equal_nan=True)
    # an origin at C: only the bus leg to D, no rail
    assert a["rail_km"][3, 1] == 0 and a["other_boardings"][3, 1] == 1


# ── fare model ───────────────────────────────────────────────────────

def test_official_ns_2026_table_is_the_default():
    from ikob2.skims.pt_fare import NS_RAIL_TABLE, parse_ns_tariff  # noqa: F401

    m = PtFareModel()
    assert len(NS_RAIL_TABLE) == 193 and NS_RAIL_TABLE[0] == (8.0, 3.00)
    km = [0.0, 0.5, 3.0, 8.0, 15.0, 30.0, 50.0, 80.0, 100.0, 150.0, 200.0,
          260.0, 1000.0]
    np.testing.assert_allclose(
        m.rail_fare(km),
        [0.0, 3.0, 3.0, 3.0, 4.60, 8.00, 12.40, 19.10, 22.70, 28.80, 33.30,
         33.30, 33.30])
    # linear between whole tariff units: 11.5 is halfway 3.70 and 4.00
    assert m.rail_fare([11.5])[0] == pytest.approx(3.85)
    assert np.isnan(m.rail_fare([np.nan])[0])
    assert (np.diff([e for _, e in NS_RAIL_TABLE]) >= 0).all()


def test_parse_ns_tariff_from_price_list_text():
    from ikob2.skims.pt_fare import parse_ns_tariff

    text = """
 0 t/m 8           € 3,00             € 2,40             € 1,80             € 2,75
 9                 € 3,30             € 2,64             € 1,98             € 3,03
 10                € 3,50             € 2,80             € 2,10             € 3,21
 Abonnement        Dal Voordeel   per jaar    € 76,20   € 76,20
 200               € 1.234,50         € 1,00
"""
    rows = parse_ns_tariff(text)
    assert rows[:3] == ((8.0, 3.0), (9.0, 3.3), (10.0, 3.5))
    assert rows[-1] == (200.0, 1234.5) and len(rows) == 4   # thousands separator


def test_rail_discount_scales_the_rail_fare_only():
    m = PtFareModel(rail_discount=0.2)
    assert m.rail_fare([50.0])[0] == pytest.approx(12.40 * 0.8)
    assert m.fare(50.0, 10.0, 1) == pytest.approx(12.40 * 0.8 + 1.08 + 1.8)
    assert "-0.2" in m.matrix_id and "-0.2" not in PtFareModel().matrix_id
    with pytest.raises(ValueError, match="rail_discount"):
        PtFareModel(rail_discount=1.0)


def test_rail_anchor_power_law_when_no_table():
    m = PtFareModel(rail_table=None)
    f = m.rail_fare([0.0, 0.4, 1.0, 10.0, 100.0, 200.0])
    assert f[0] == 0.0 and f[1] == pytest.approx(2.60) and f[2] == pytest.approx(2.60)
    assert f[4] == pytest.approx(20.0, rel=1e-9)
    per_km = m.rail_fare([1.0, 10.0, 100.0]) / np.array([1.0, 10.0, 100.0])
    assert per_km[0] > per_km[1] > per_km[2]


def test_rail_table_overrides_and_can_extrapolate():
    m = PtFareModel(rail_table=((1, 3.0), (10, 5.0), (50, 12.0)),
                    rail_beyond_table="linear")
    np.testing.assert_allclose(m.rail_fare([0, 1, 5.5, 50, 60]),
                               [0, 3.0, 4.0, 12.0, 12.0 + 0.175 * 10])
    capped = PtFareModel(rail_table=((1, 3.0), (10, 5.0), (50, 12.0)))
    assert capped.rail_fare([60.0])[0] == 12.0
    assert m.rail_fare([0.2])[0] == pytest.approx(3.0)      # floor at first fare


def test_regional_fare_single_versus_count_and_total():
    single = PtFareModel()
    count = PtFareModel(boardings="count")
    assert single.fare(0, 10.0, 0)[()] == 0.0           # no regional boarding
    assert single.fare(0, 10.0, 1) == pytest.approx(1.08 + 0.18 * 10)
    assert single.fare(0, 10.0, 3) == pytest.approx(1.08 + 0.18 * 10)
    assert count.fare(0, 10.0, 3) == pytest.approx(3 * 1.08 + 0.18 * 10)
    both = single.fare(10.0, 5.0, 1)
    assert both == pytest.approx(single.rail_fare([10.0])[0] + 1.08 + 0.9)
    assert single.rail_fare([10.0])[0] == pytest.approx(3.50)
    assert np.isnan(single.fare(np.nan, 5.0, 1))


def test_fare_model_validation_and_id():
    with pytest.raises(ValueError, match="taper"):
        PtFareModel(rail_eur_per_km_at_1km=0.2, rail_eur_per_km_at_100km=0.2)
    with pytest.raises(ValueError, match="cap"):
        PtFareModel(rail_beyond_table="wrap")
    with pytest.raises(ValueError, match="boardings"):
        PtFareModel(boardings="many")
    with pytest.raises(ValueError, match="increasing km"):
        PtFareModel(rail_table=((5, 1.0), (1, 2.0)))
    assert PtFareModel().matrix_id != PtFareModel(boardings="count").matrix_id
    assert "rail=ns" in PtFareModel().matrix_id
    assert "rail=2.6-0.2" in PtFareModel(rail_table=None).matrix_id


def test_fare_of_a_routed_journey(tmp_path):
    j = journeys(tmp_path)
    fare = PtFareModel().fare(j["rail_km"], j["other_km"],
                              j["other_boardings"])
    assert fare[0, 0] == pytest.approx(PtFareModel().rail_fare([5.75])[0],
                                       rel=1e-3)
    assert fare[0, 1] > fare[0, 0] and np.isnan(fare[0, 2])


# ── bicycle access and egress ────────────────────────────────────────

from ikob2.skims.gtfs_pt import LegSpec  # noqa: E402


def leg_journeys(tmp_path, o, d, **kw):
    tt = load_peak_timetable(gtfs_zip(tmp_path), "2026-09-15")
    return PtRouter(tt).journeys(np.array([o]), np.array(d), max_minutes=180,
                                 **kw)


def test_bicycle_access_reaches_where_walking_does_not(tmp_path):
    o, d = xy(0, 3000), [xy(5000, 100)]
    walk = leg_journeys(tmp_path, o, d)
    assert np.isnan(walk["time"]).all()                    # 3 km: too far
    bike = leg_journeys(tmp_path, o, d, access=LegSpec(kmh=16.0, detour=1.3,
                                                       fixed_minutes=1.0))
    ride = 3000 * 1.3 / (16000 / 60)                        # minutes
    assert bike["access_min"][0, 0] == pytest.approx(ride, abs=0.05)
    assert np.isfinite(bike["time"][0, 0])
    # access ride + 1 min unlock + wait 5 + 6 min in the train + egress walk
    egress = 100 * 1.3 / (4000 / 60)
    assert bike["time"][0, 0] == pytest.approx(ride + 1 + 5 + 6 + egress,
                                               abs=0.2)
    assert walk["access_min"].shape == (1, 1)


def test_walking_access_reports_zero_bicycle_minutes(tmp_path):
    j = leg_journeys(tmp_path, xy(0, 200), [xy(5000, 100)])
    assert j["access_min"][0, 0] == 0.0


def test_bicycle_egress_only_from_rail_hubs(tmp_path):
    o = xy(0, 200)
    far_from_rail = [xy(15000, 2500)]          # 2.5 km from bus stop D
    near_rail = [xy(10000, 2500)]              # 2.5 km from rail stop C
    hub = LegSpec(hubs_only=True)
    anyy = LegSpec(hubs_only=False)
    assert np.isnan(leg_journeys(tmp_path, o, far_from_rail,
                                 egress=hub)["time"]).all()
    assert np.isfinite(leg_journeys(tmp_path, o, far_from_rail,
                                    egress=anyy)["time"]).all()
    assert np.isfinite(leg_journeys(tmp_path, o, near_rail,
                                    egress=hub)["time"]).all()


def test_bicycle_egress_is_faster_than_walking_where_both_work(tmp_path):
    o, d = xy(0, 200), [xy(10000, 900)]        # 0.9 km from rail stop C
    walk = leg_journeys(tmp_path, o, d)
    bike = leg_journeys(tmp_path, o, d, egress=LegSpec(hubs_only=True,
                                                       fixed_minutes=1.0))
    # 0.9 km: walking 17.6 min; bicycle 4.4 min ride + 1 min fixed
    assert walk["time"][0, 0] - bike["time"][0, 0] > 5


def test_bicycle_egress_minutes_are_reported(tmp_path):
    o, d = xy(0, 200), [xy(10000, 900), xy(5000, 100)]
    j = leg_journeys(tmp_path, o, d, egress=LegSpec(hubs_only=True,
                                                    fixed_minutes=1.0))
    ride = 900 * 1.3 / (16000 / 60)                # 0.9 km from rail stop C
    assert j["egress_min"][0, 0] == pytest.approx(ride, abs=0.1)
    assert np.isfinite(j["time"][0, 1]) and np.isfinite(j["egress_min"][0, 1])
    walk_only = leg_journeys(tmp_path, o, d)
    assert np.isnan(walk_only["egress_min"]).all()     # no bicycle egress
