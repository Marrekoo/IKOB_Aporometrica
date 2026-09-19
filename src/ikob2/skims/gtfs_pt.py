"""
Public transport skims from GTFS with a frequency (headway) model.

No timetable search and no averaging over departure times. Each line
(route x direction) gets, from the GTFS trips of one weekday inside a
peak window, a headway per stop and a median in-vehicle time between
consecutive stops. A boarding costs the expected wait

    wait = min(headway / 2, 7.5 minutes)

i.e. half the headway below 15 minutes and a cap of 7.5 minutes above it
(passengers time their arrival to infrequent services). Transfers have no
penalty besides the boarding wait and the walk between stops
(`boarding_penalty_min` is 0 and adjustable). Walking (access, egress,
transfers) is the crow-fly distance times a detour factor at an
adjustable speed (default 4 km/h).

Graph: every stop has a "before boarding" and an "after alighting" node;
origins walk to the first kind, which only connect to the line-stop nodes
(board, wait), then ride to the next line-stop and alight (0) onto the
second kind, from which one may walk to a nearby stop, board again, or
leave to the destination. This forces at least one boarding: a trip is
never just a walk. Shortest paths (scipy Dijkstra) give times at the
after-alighting nodes; a destination's time is the best of those plus the
egress walk.

Limits: the headway is the number of trips at a stop in the window (lines
that share a corridor are not combined into a higher frequency), all
route types are treated alike, and a line's in-vehicle times are medians
over its trips.
"""

from __future__ import annotations

import io
import logging
import zipfile
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse import csgraph
from scipy.spatial import cKDTree

logger = logging.getLogger(__name__)

WAIT_CAP_MIN = 7.5


def boarding_wait(headway_min, cap_min: float = WAIT_CAP_MIN):
    """Expected wait: half the headway, capped (so 7.5 minutes for every
    headway of 15 minutes or more)."""
    h = np.asarray(headway_min, dtype=float)
    return np.minimum(h / 2.0, cap_min)


@dataclass
class PeakTimetable:
    """The peak-window view of a GTFS feed.

    stops:      DataFrame indexed by stop_id with lat, lon
    line_stops: line_id, stop_id, headway_min, can_board, can_alight
    rides:      line_id, from_stop, to_stop, minutes (consecutive stops)
    lines:      line_id -> route_id, direction_id, route_type
    """
    stops: pd.DataFrame
    line_stops: pd.DataFrame
    rides: pd.DataFrame
    lines: pd.DataFrame
    meta: dict = field(default_factory=dict)


# ── GTFS reading ─────────────────────────────────────────────────────

def _seconds(col: pd.Series) -> pd.Series:
    """'HH:MM:SS' (hours may exceed 24) -> seconds; NaN where empty."""
    parts = col.str.extract(r"^\s*(\d+):(\d+):(\d+)\s*$").astype(float)
    return parts[0] * 3600 + parts[1] * 60 + parts[2]


def active_services(zf: zipfile.ZipFile, date: str) -> set[str]:
    """service_ids running on `date` (YYYYMMDD): calendar.txt weekdays and
    range, then the calendar_dates.txt additions and removals."""
    ymd = pd.Timestamp(date)
    day = ymd.strftime("%A").lower()
    active: set[str] = set()
    names = set(zf.namelist())
    stamp = ymd.strftime("%Y%m%d")
    if "calendar.txt" in names:
        cal = pd.read_csv(zf.open("calendar.txt"), dtype=str)
        ok = ((cal["start_date"] <= stamp) & (cal["end_date"] >= stamp)
              & (cal[day] == "1"))
        active |= set(cal.loc[ok, "service_id"])
    if "calendar_dates.txt" in names:
        cd = pd.read_csv(zf.open("calendar_dates.txt"), dtype=str)
        cd = cd[cd["date"] == stamp]
        active |= set(cd.loc[cd["exception_type"] == "1", "service_id"])
        active -= set(cd.loc[cd["exception_type"] == "2", "service_id"])
    return active


def load_peak_timetable(gtfs_zip, date: str, *, window_h=(7.0, 9.0),
                        chunksize: int = 2_000_000) -> PeakTimetable:
    """Read one weekday of a GTFS feed and reduce it to line headways
    and in-vehicle times in the window (hours, local time)."""
    w0, w1 = window_h[0] * 3600, window_h[1] * 3600
    zf = zipfile.ZipFile(gtfs_zip)
    services = active_services(zf, date)
    if not services:
        raise ValueError(f"No service runs on {date} in {gtfs_zip}.")
    trips = pd.read_csv(
        zf.open("trips.txt"), dtype=str,
        usecols=["route_id", "service_id", "trip_id", "direction_id"])
    trips = trips[trips["service_id"].isin(services)].copy()
    trips["direction_id"] = trips["direction_id"].fillna("0")
    trip_line = trips.set_index("trip_id")[["route_id", "direction_id"]]
    trip_ids = set(trip_line.index)
    routes = pd.read_csv(zf.open("routes.txt"), dtype=str,
                         usecols=["route_id", "route_type"])
    logger.info("%d trips run on %s", len(trip_ids), date)

    kept = []
    cols = ["trip_id", "stop_sequence", "stop_id", "arrival_time",
            "departure_time", "pickup_type", "drop_off_type"]
    reader = pd.read_csv(zf.open("stop_times.txt"), dtype=str,
                         usecols=lambda c: c in cols, chunksize=chunksize)
    for chunk in reader:
        chunk = chunk[chunk["trip_id"].isin(trip_ids)]
        if chunk.empty:
            continue
        dep = _seconds(chunk["departure_time"].fillna(chunk["arrival_time"]))
        arr = _seconds(chunk["arrival_time"].fillna(chunk["departure_time"]))
        keep = (dep >= w0 - 3600) & (dep <= w1 + 7200)
        chunk = chunk.loc[keep].assign(dep=dep[keep], arr=arr[keep])
        if len(chunk):
            kept.append(chunk[["trip_id", "stop_sequence", "stop_id", "arr",
                               "dep"]
                              + [c for c in ("pickup_type", "drop_off_type")
                                 if c in chunk.columns]])
    if not kept:
        raise ValueError("No stop times in the window.")
    st = pd.concat(kept, ignore_index=True)
    st["stop_sequence"] = st["stop_sequence"].astype(int)
    for c in ("pickup_type", "drop_off_type"):
        st[c] = st[c].fillna("0") if c in st.columns else "0"
    st = st.join(trip_line, on="trip_id")
    stops = pd.read_csv(zf.open("stops.txt"), dtype={"stop_id": str},
                        usecols=["stop_id", "stop_lat", "stop_lon"]
                        ).rename(columns={"stop_lat": "lat", "stop_lon": "lon"}
                                 ).set_index("stop_id")
    return reduce_stop_times(st, stops, routes, w0, w1,
                             meta={"gtfs": str(gtfs_zip), "date": date,
                                   "window_h": list(window_h)})


def reduce_stop_times(st: pd.DataFrame, stops: pd.DataFrame,
                      routes: pd.DataFrame, w0: float, w1: float,
                      meta: dict | None = None) -> PeakTimetable:
    """Stop times (trip_id, stop_sequence, stop_id, arr, dep, pickup_type,
    drop_off_type, route_id, direction_id; seconds) -> PeakTimetable."""
    st = st.sort_values(["trip_id", "stop_sequence"]).copy()
    st["line"] = st["route_id"] + "|" + st["direction_id"]
    line_ids = {k: i for i, k in enumerate(sorted(st["line"].unique()))}
    st["line_id"] = st["line"].map(line_ids)

    # headway per line and stop: departures inside the window
    win = st[(st["dep"] >= w0) & (st["dep"] < w1)]
    n = win.groupby(["line_id", "stop_id"]).size().rename("n").reset_index()
    n["headway_min"] = (w1 - w0) / 60.0 / n["n"]
    flags = st.groupby(["line_id", "stop_id"]).agg(
        can_board=("pickup_type", lambda s: bool((s != "1").any())),
        can_alight=("drop_off_type", lambda s: bool((s != "1").any()))
    ).reset_index()
    line_stops = n.merge(flags, on=["line_id", "stop_id"])

    # in-vehicle minutes between consecutive stops of a trip
    nxt = st.groupby("trip_id").shift(-1)
    ok = nxt["stop_id"].notna()
    rides = pd.DataFrame({
        "line_id": st.loc[ok, "line_id"], "from_stop": st.loc[ok, "stop_id"],
        "to_stop": nxt.loc[ok, "stop_id"],
        "minutes": (nxt.loc[ok, "arr"] - st.loc[ok, "arr"]) / 60.0})
    rides = rides[(rides["minutes"] >= 0)
                  & rides["from_stop"].ne(rides["to_stop"])]
    rides = rides.groupby(["line_id", "from_stop", "to_stop"],
                          as_index=False)["minutes"].median()
    # keep only rides between stops that have a headway (run in the window)
    have = set(zip(line_stops["line_id"], line_stops["stop_id"]))
    mask = [(l, a) in have and (l, b) in have for l, a, b in zip(
        rides["line_id"], rides["from_stop"], rides["to_stop"])]
    rides = rides[mask]

    inv = {v: k for k, v in line_ids.items()}
    lines = pd.DataFrame({"line_id": list(inv),
                          "line": [inv[i] for i in inv]})
    lines[["route_id", "direction_id"]] = lines["line"].str.split(
        "|", expand=True)
    lines = lines.merge(routes[["route_id", "route_type"]], on="route_id",
                        how="left").drop(columns="line")
    return PeakTimetable(stops, line_stops, rides, lines, meta or {})


# ── Router ───────────────────────────────────────────────────────────

class PtRouter:
    """Frequency-model public transport router over a PeakTimetable."""

    def __init__(self, tt: PeakTimetable, *, walk_kmh: float = 4.0,
                 walk_detour: float = 1.3, max_access_min: float = 20.0,
                 transfer_radius_m: float = 300.0,
                 wait_cap_min: float = WAIT_CAP_MIN,
                 boarding_penalty_min: float = 0.0):
        if walk_kmh <= 0 or walk_detour <= 0:
            raise ValueError("walk_kmh and walk_detour must be positive.")
        self.tt = tt
        self.walk_kmh = walk_kmh
        self.detour = walk_detour
        self.max_access_min = max_access_min
        self.transfer_radius_m = transfer_radius_m
        self.wait_cap = wait_cap_min
        self.board_penalty = boarding_penalty_min
        self._build()

    # walking minutes for a crow-fly distance in metres
    def walk_min(self, crow_m):
        return np.asarray(crow_m, dtype=float) * self.detour \
            / (self.walk_kmh * 1000.0 / 60.0)

    @property
    def access_radius_m(self) -> float:
        return self.max_access_min * (self.walk_kmh * 1000.0 / 60.0) \
            / self.detour

    def _build(self) -> None:
        from pyproj import Transformer

        tt = self.tt
        used = set(tt.line_stops["stop_id"])
        stops = tt.stops.loc[[s for s in tt.stops.index if s in used]]
        self.stop_ids = list(stops.index)
        self.stop_index = {s: i for i, s in enumerate(self.stop_ids)}
        x, y = Transformer.from_crs("EPSG:4326", "EPSG:28992",
                                    always_xy=True).transform(
            stops["lon"].to_numpy(), stops["lat"].to_numpy())
        self.stop_xy = np.column_stack([x, y])
        self.stop_tree = cKDTree(self.stop_xy)
        n_stops = len(self.stop_ids)          # layer 0: before boarding
        # layer 1 (after alighting) is n_stops + i; line-stops follow

        ls = tt.line_stops
        ls = ls[ls["stop_id"].isin(self.stop_index)].reset_index(drop=True)
        # two nodes per line-stop: "boarded here" (b) and "on board after at
        # least one ride" (r); only r can alight, so a boarding is never
        # cancelled at the same stop
        n_ls = len(ls)
        ls_index = {(l, s): i for i, (l, s) in enumerate(
            zip(ls["line_id"], ls["stop_id"]))}
        self.n_stops, self.n_line_stops = n_stops, 2 * n_ls
        b0 = 2 * n_stops                       # first "boarded" node
        r0 = 2 * n_stops + n_ls                # first "riding" node
        u, v, w = [], [], []

        sidx = ls["stop_id"].map(self.stop_index).to_numpy()
        lsn = np.arange(n_ls) + b0
        wait = boarding_wait(ls["headway_min"].to_numpy(), self.wait_cap) \
            + self.board_penalty
        b = ls["can_board"].to_numpy()
        for layer in (0, 1):                                        # board
            u.append(sidx[b] + layer * n_stops)
            v.append(lsn[b]); w.append(wait[b])
        a = ls["can_alight"].to_numpy()
        u.append(np.arange(n_ls)[a] + r0)                            # alight
        v.append(sidx[a] + n_stops)
        w.append(np.zeros(a.sum()))

        r = tt.rides
        ru = [ls_index.get((l, s)) for l, s in zip(r["line_id"], r["from_stop"])]
        rv = [ls_index.get((l, s)) for l, s in zip(r["line_id"], r["to_stop"])]
        ok = np.array([a_ is not None and b_ is not None
                       for a_, b_ in zip(ru, rv)], dtype=bool)
        ru_i = np.array(ru, dtype=object)[ok].astype(int)
        rv_i = np.array(rv, dtype=object)[ok].astype(int)
        rm = r["minutes"].to_numpy(dtype=float)[ok]
        for start in (b0, r0):                     # boarded->riding, riding->riding
            u.append(ru_i + start)
            v.append(rv_i + r0)
            w.append(rm)

        # walking transfers between nearby stops
        pairs = self.stop_tree.query_pairs(self.transfer_radius_m,
                                           output_type="ndarray")
        if len(pairs):
            d = np.hypot(*(self.stop_xy[pairs[:, 0]]
                           - self.stop_xy[pairs[:, 1]]).T)
            tw = self.walk_min(d)
            # transfers only between after-alighting nodes
            u.append(pairs[:, 0] + n_stops); v.append(pairs[:, 1] + n_stops)
            w.append(tw)
            u.append(pairs[:, 1] + n_stops); v.append(pairs[:, 0] + n_stops)
            w.append(tw)
        self._edges = (np.concatenate(u), np.concatenate(v),
                       np.concatenate(w))
        logger.info("PT graph: %d stops, %d line-stops, %d edges",
                    n_stops, len(ls), len(self._edges[0]))

    # ── queries ──────────────────────────────────────────────────────

    def _links(self, xy: np.ndarray):
        """CSR of point -> stops within the access radius: indptr,
        stop indices, walking minutes."""
        near = self.stop_tree.query_ball_point(xy, self.access_radius_m)
        counts = np.array([len(n) for n in near])
        indptr = np.concatenate([[0], np.cumsum(counts)])
        idx = np.concatenate([np.array(n, dtype=int) for n in near]) \
            if counts.sum() else np.zeros(0, dtype=int)
        rep = np.repeat(np.arange(len(xy)), counts)
        d = np.hypot(*(xy[rep] - self.stop_xy[idx]).T) if len(idx) else \
            np.zeros(0)
        return indptr, idx, self.walk_min(d)

    def time_matrix(self, origin_xy, dest_xy, *, max_minutes: float = 180.0,
                    chunk: int = 8) -> np.ndarray:
        """Door-to-door PT minutes (origins x destinations, float32, NaN
        where not reachable within `max_minutes`). Coordinates: RD New
        metres. A trip uses at least one boarding."""
        o = np.asarray(origin_xy, dtype=float)
        d = np.asarray(dest_xy, dtype=float)
        n_nodes = 2 * self.n_stops + self.n_line_stops
        o_ptr, o_idx, o_w = self._links(o)
        u, v, w = self._edges
        origin_nodes = n_nodes + np.arange(len(o))
        rep = np.repeat(np.arange(len(o)), np.diff(o_ptr))
        graph = sparse.csr_matrix(
            (np.concatenate([w, o_w]),
             (np.concatenate([u, origin_nodes[rep]]),
              np.concatenate([v, o_idx]))),
            shape=(n_nodes + len(o), n_nodes + len(o)))
        e_ptr, e_idx, e_w = self._links(d)
        out = np.full((len(o), len(d)), np.nan, dtype=np.float32)
        has = np.diff(e_ptr) > 0
        starts = e_ptr[:-1][has]
        for s in range(0, len(o), chunk):
            src = origin_nodes[s:s + chunk]
            dist = csgraph.dijkstra(graph, directed=True, indices=src,
                                    limit=max_minutes)
            stop_t = dist[:, self.n_stops:2 * self.n_stops]   # after alighting
            vals = stop_t[:, e_idx] + e_w[None, :]
            best = np.minimum.reduceat(vals, starts, axis=1) \
                if len(starts) else np.zeros((len(src), 0))
            block = np.full((len(src), len(d)), np.inf)
            block[:, has] = best
            block[block > max_minutes] = np.nan
            out[s:s + chunk] = block
        return out
