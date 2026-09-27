"""
Cut a GTFS feed down to a region and a date range.

A national graph does not fit in memory on a 15 GB machine (OpenTripPlanner
was killed at 10.5 GB while still building the street graph), so the
local OTP server works on a region: stops inside a bounding box, the trips
that visit at least two of them, and only the services that run on one of
the given dates.
"""

from __future__ import annotations

import logging
import zipfile

import pandas as pd

from ikob2.skims.gtfs_pt import active_services

logger = logging.getLogger(__name__)


def subset_gtfs(src, dst, bbox, dates, *, chunksize: int = 2_000_000) -> dict:
    """Write `dst` from `src`. bbox = (lon_min, lat_min, lon_max, lat_max);
    dates: ISO dates whose services are kept. Returns counts."""
    lon0, lat0, lon1, lat1 = bbox
    zin = zipfile.ZipFile(src)
    services: set[str] = set()
    for d in dates:
        services |= active_services(zin, d)
    if not services:
        raise ValueError("No service runs on any of the dates.")

    stops = pd.read_csv(zin.open("stops.txt"), dtype=str)
    lat = pd.to_numeric(stops["stop_lat"], errors="coerce")
    lon = pd.to_numeric(stops["stop_lon"], errors="coerce")
    inside = lat.between(lat0, lat1) & lon.between(lon0, lon1)
    keep_stops = set(stops.loc[inside, "stop_id"])
    # parent stations of kept stops stay, so the hierarchy is intact
    if "parent_station" in stops.columns:
        parents = set(stops.loc[inside, "parent_station"].dropna())
        keep_stops |= parents
    trips = pd.read_csv(zin.open("trips.txt"), dtype=str)
    trips = trips[trips["service_id"].isin(services)]
    trip_ids = set(trips["trip_id"])

    parts = []
    for chunk in pd.read_csv(zin.open("stop_times.txt"), dtype=str,
                             chunksize=chunksize):
        chunk = chunk[chunk["trip_id"].isin(trip_ids)
                      & chunk["stop_id"].isin(keep_stops)]
        if len(chunk):
            parts.append(chunk)
    st = pd.concat(parts, ignore_index=True)
    n = st.groupby("trip_id").size()
    st = st[st["trip_id"].isin(n[n >= 2].index)]
    trips = trips[trips["trip_id"].isin(set(st["trip_id"]))]
    routes = pd.read_csv(zin.open("routes.txt"), dtype=str)
    routes = routes[routes["route_id"].isin(set(trips["route_id"]))]
    agency = pd.read_csv(zin.open("agency.txt"), dtype=str)
    if "agency_id" in routes.columns and "agency_id" in agency.columns:
        agency = agency[agency["agency_id"].isin(set(routes["agency_id"]))]
    used_stops = set(st["stop_id"])
    stops_out = stops[stops["stop_id"].isin(
        used_stops | (parents if "parent_station" in stops.columns
                      else set()))]

    names = set(zin.namelist())
    with zipfile.ZipFile(dst, "w", zipfile.ZIP_DEFLATED) as zout:
        def put(name, df):
            zout.writestr(name, df.to_csv(index=False))
        put("agency.txt", agency)
        put("stops.txt", stops_out)
        put("routes.txt", routes)
        put("trips.txt", trips)
        put("stop_times.txt", st)
        if "calendar_dates.txt" in names:
            cd = pd.read_csv(zin.open("calendar_dates.txt"), dtype=str)
            put("calendar_dates.txt",
                cd[cd["service_id"].isin(set(trips["service_id"]))])
        if "calendar.txt" in names:
            cal = pd.read_csv(zin.open("calendar.txt"), dtype=str)
            put("calendar.txt",
                cal[cal["service_id"].isin(set(trips["service_id"]))])
        if "feed_info.txt" in names:
            zout.writestr("feed_info.txt", zin.read("feed_info.txt"))
    return {"stops": len(stops_out), "trips": len(trips),
            "stop_times": len(st), "routes": len(routes)}
