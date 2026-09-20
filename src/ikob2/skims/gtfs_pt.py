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

Fares need the distance travelled by rail and by other modes (bus, tram,
metro, ferry) and the number of boardings on the latter. `journeys()`
returns them next to the time: they are accumulated along the shortest-path
tree (pointer doubling), taken from the crow-fly distance between
consecutive stops times a detour factor per class (`rail_detour`,
`other_detour`). `skims.pt_fare` turns them into a fare.

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

from ikob2.params import DEFAULTS

logger = logging.getLogger(__name__)

_PT = DEFAULTS.pt
_BIKE = DEFAULTS.bike_leg
WAIT_CAP_MIN = _PT.wait_cap_min


def is_rail(route_type) -> bool:
    """GTFS route type 2 or an extended railway type (100-199)."""
    t = str(route_type)
    return t == "2" or (t.isdigit() and 100 <= int(t) < 200)


def boarding_wait(headway_min, cap_min: float = WAIT_CAP_MIN):
    """Expected wait: half the headway, capped (so 7.5 minutes for every
    headway of 15 minutes or more)."""
    h = np.asarray(headway_min, dtype=float)
    return np.minimum(h / 2.0, cap_min)


@dataclass(frozen=True)
class LegSpec:
    """An access or egress leg by another mode than walking (a bicycle).

    kmh, detour : speed and crow-fly detour factor of the leg;
    max_minutes : longest leg (ride time), with `fixed_minutes` (unlocking,
        parking, returning) added on top of the ride;
    hubs_only   : the leg can only start/end at hub stops (rail stops:
        where OV-fiets is available); otherwise at any stop.
    """
    kmh: float = _BIKE.kmh
    detour: float = _BIKE.detour
    max_minutes: float = _BIKE.max_minutes
    fixed_minutes: float = _BIKE.fixed_minutes
    hubs_only: bool = False

    def __post_init__(self):
        if self.kmh <= 0 or self.detour <= 0 or self.max_minutes <= 0:
            raise ValueError("kmh, detour and max_minutes must be positive.")
        if self.fixed_minutes < 0:
            raise ValueError("fixed_minutes must be >= 0.")


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


def load_peak_timetable(gtfs_zip, date: str, *,
                        window_h=tuple(_PT.window_h),
                        chunksize: int = _PT.gtfs_chunksize) -> PeakTimetable:
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

    def __init__(self, tt: PeakTimetable, *, walk_kmh: float = _PT.walk_kmh,
                 walk_detour: float = _PT.walk_detour,
                 max_access_min: float = _PT.max_access_min,
                 transfer_radius_m: float = _PT.transfer_radius_m,
                 wait_cap_min: float = WAIT_CAP_MIN,
                 boarding_penalty_min: float = _PT.boarding_penalty_min,
                 rail_detour: float = _PT.rail_detour,
                 other_detour: float = _PT.other_detour,
                 hubs_xy: np.ndarray | None = None,
                 hub_walk_radius_m: float = _PT.hub_walk_radius_m):
        if walk_kmh <= 0 or walk_detour <= 0:
            raise ValueError("walk_kmh and walk_detour must be positive.")
        if rail_detour < 1 or other_detour < 1:
            raise ValueError("ride detour factors must be >= 1.")
        self.rail_detour = rail_detour
        self.other_detour = other_detour
        self.tt = tt
        self.walk_kmh = walk_kmh
        self.detour = walk_detour
        self.max_access_min = max_access_min
        self.transfer_radius_m = transfer_radius_m
        self.wait_cap = wait_cap_min
        self.board_penalty = boarding_penalty_min
        # hub locations from a file (RD New metres); None: the rail stops
        self.hubs_xy = None if hubs_xy is None else np.asarray(
            hubs_xy, dtype=float).reshape(-1, 2)
        self.hub_walk_radius_m = hub_walk_radius_m
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
        rail_lines = set(tt.lines.loc[[is_rail(t) for t in
                                       tt.lines["route_type"]], "line_id"])
        rail_stops = set(tt.line_stops.loc[
            tt.line_stops["line_id"].isin(rail_lines), "stop_id"])
        self.hub_idx = np.array([i for i, sid in enumerate(self.stop_ids)
                                 if sid in rail_stops], dtype=int)
        self.hub_tree = cKDTree(self.stop_xy[self.hub_idx]) \
            if len(self.hub_idx) else None
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
        k_rail, k_other, b_other = [], [], []      # per-edge fare attributes

        def attrs(n, rail=None, other=None, board=None):
            z = np.zeros(n)
            k_rail.append(z if rail is None else rail)
            k_other.append(z if other is None else other)
            b_other.append(z if board is None else board)

        sidx = ls["stop_id"].map(self.stop_index).to_numpy()
        lsn = np.arange(n_ls) + b0
        wait = boarding_wait(ls["headway_min"].to_numpy(), self.wait_cap) \
            + self.board_penalty
        b = ls["can_board"].to_numpy()
        line_rail = ls["line_id"].map(
            dict(zip(tt.lines["line_id"],
                     [is_rail(t) for t in tt.lines["route_type"]]))
        ).fillna(False).to_numpy(dtype=bool)
        for layer in (0, 1):                                        # board
            u.append(sidx[b] + layer * n_stops)
            v.append(lsn[b]); w.append(wait[b])
            attrs(int(b.sum()), board=(~line_rail[b]).astype(float))
        a = ls["can_alight"].to_numpy()
        u.append(np.arange(n_ls)[a] + r0)                            # alight
        v.append(sidx[a] + n_stops)
        w.append(np.zeros(a.sum()))
        attrs(int(a.sum()))

        r = tt.rides
        ru = [ls_index.get((l, s)) for l, s in zip(r["line_id"], r["from_stop"])]
        rv = [ls_index.get((l, s)) for l, s in zip(r["line_id"], r["to_stop"])]
        ok = np.array([a_ is not None and b_ is not None
                       for a_, b_ in zip(ru, rv)], dtype=bool)
        ru_i = np.array(ru, dtype=object)[ok].astype(int)
        rv_i = np.array(rv, dtype=object)[ok].astype(int)
        rm = r["minutes"].to_numpy(dtype=float)[ok]
        # ride length: crow-fly between the stops x a detour per class
        a_xy = self.stop_xy[ls["stop_id"].map(self.stop_index).to_numpy()
                            [ru_i]]
        b_xy = self.stop_xy[ls["stop_id"].map(self.stop_index).to_numpy()
                            [rv_i]]
        crow_km = np.hypot(*(a_xy - b_xy).T) / 1000.0
        rail_ride = line_rail[ru_i]
        km_r = np.where(rail_ride, crow_km * self.rail_detour, 0.0)
        km_o = np.where(rail_ride, 0.0, crow_km * self.other_detour)
        for start in (b0, r0):                     # boarded->riding, riding->riding
            u.append(ru_i + start)
            v.append(rv_i + r0)
            w.append(rm)
            attrs(len(rm), rail=km_r, other=km_o)

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
            attrs(2 * len(pairs))
        self._edges = (np.concatenate(u), np.concatenate(v),
                       np.concatenate(w))
        self._attr = np.column_stack([np.concatenate(k_rail),
                                      np.concatenate(k_other),
                                      np.concatenate(b_other)])
        n_nodes = 2 * n_stops + self.n_line_stops
        self._keys = self._edges[0].astype(np.int64) * n_nodes \
            + self._edges[1].astype(np.int64)
        order = np.argsort(self._keys)
        self._key_sorted = self._keys[order]
        self._key_to_edge = order
        logger.info("PT graph: %d stops, %d line-stops, %d edges",
                    n_stops, len(ls), len(self._edges[0]))

    # ── queries ──────────────────────────────────────────────────────

    def _links(self, xy: np.ndarray, leg: LegSpec | None = None):
        """CSR of point -> stops within reach: indptr, stop indices,
        minutes (ride plus fixed minutes), ride minutes. `leg` None is
        walking (max_access_min, walk speed); a LegSpec is a bicycle leg,
        optionally only to hub stops."""
        if leg is not None and leg.hubs_only and self.hubs_xy is not None:
            return self._hub_links(xy, leg)
        if leg is None:
            radius, tree, ids = self.access_radius_m, self.stop_tree, None
            fixed = 0.0
        else:
            radius = leg.max_minutes * (leg.kmh * 1000.0 / 60.0) / leg.detour
            tree, ids = ((self.hub_tree, self.hub_idx) if leg.hubs_only
                         else (self.stop_tree, None))
            fixed = leg.fixed_minutes
        if tree is None:                              # no hubs at all
            z = np.zeros(0)
            return np.zeros(len(xy) + 1, dtype=int), np.zeros(0, dtype=int), z, z
        near = tree.query_ball_point(xy, radius)
        counts = np.array([len(n) for n in near])
        indptr = np.concatenate([[0], np.cumsum(counts)])
        idx = np.concatenate([np.array(n, dtype=int) for n in near]) \
            if counts.sum() else np.zeros(0, dtype=int)
        rep = np.repeat(np.arange(len(xy)), counts)
        d = np.hypot(*(xy[rep] - tree.data[idx]).T) if len(idx) \
            else np.zeros(0)
        stop_idx = ids[idx] if ids is not None else idx
        if leg is None:
            ride = self.walk_min(d)
        else:
            ride = d * leg.detour / (leg.kmh * 1000.0 / 60.0)
        return indptr, stop_idx, ride + fixed, ride

    def _hub_links(self, xy, leg: LegSpec):
        """Egress links through hubs from a file: a stop within
        `hub_walk_radius_m` of a hub, a walk to the hub, then the bicycle
        from the hub to the point. Same CSR layout as `_links`: per point the
        stops it can be reached from, with minutes (walk to the hub, ride,
        fixed minutes) and the ride minutes; the fastest hub per stop."""
        xy = np.asarray(xy, dtype=float)
        empty = (np.zeros(len(xy) + 1, dtype=int), np.zeros(0, dtype=int),
                 np.zeros(0), np.zeros(0))
        if not len(self.hubs_xy):
            return empty
        hub_stops = self.stop_tree.query_ball_point(self.hubs_xy,
                                                    self.hub_walk_radius_m)
        h_cnt = np.array([len(s) for s in hub_stops])
        h_ptr = np.concatenate([[0], np.cumsum(h_cnt)])
        h_stop = np.concatenate([np.array(s, dtype=int) for s in hub_stops]) \
            if h_cnt.sum() else np.zeros(0, dtype=int)
        h_rep = np.repeat(np.arange(len(self.hubs_xy)), h_cnt)
        h_walk = self.walk_min(np.hypot(*(self.hubs_xy[h_rep]
                                          - self.stop_xy[h_stop]).T)) \
            if len(h_stop) else np.zeros(0)
        radius = leg.max_minutes * (leg.kmh * 1000.0 / 60.0) / leg.detour
        near = cKDTree(self.hubs_xy).query_ball_point(xy, radius)
        n_near = np.array([len(n) for n in near])
        if not n_near.sum():
            return empty
        pt = np.repeat(np.arange(len(xy)), n_near)
        hub = np.concatenate([np.array(n, dtype=int) for n in near])
        ride = np.hypot(*(xy[pt] - self.hubs_xy[hub]).T) * leg.detour \
            / (leg.kmh * 1000.0 / 60.0)
        k = h_cnt[hub]                       # stops next to each hub
        rep = np.repeat(np.arange(len(pt)), k)
        first = np.repeat(np.cumsum(k) - k, k)
        pos = h_ptr[hub][rep] + (np.arange(k.sum()) - first)
        p_pt, p_stop = pt[rep], h_stop[pos]
        p_ride = ride[rep]
        p_min = p_ride + leg.fixed_minutes + h_walk[pos]
        order = np.lexsort((p_min, p_stop, p_pt))      # fastest hub per stop
        p_pt, p_stop, p_min, p_ride = (a[order] for a in
                                       (p_pt, p_stop, p_min, p_ride))
        keep = np.ones(len(p_pt), dtype=bool)
        keep[1:] = (p_pt[1:] != p_pt[:-1]) | (p_stop[1:] != p_stop[:-1])
        p_pt, p_stop, p_min, p_ride = (a[keep] for a in
                                       (p_pt, p_stop, p_min, p_ride))
        indptr = np.concatenate([[0], np.cumsum(np.bincount(
            p_pt, minlength=len(xy)))])
        return indptr, p_stop, p_min, p_ride

    def time_matrix(self, origin_xy, dest_xy, *, max_minutes: float = _PT.max_minutes,
                    chunk: int = 8) -> np.ndarray:
        """Door-to-door PT minutes (origins x destinations, float32, NaN
        where not reachable within `max_minutes`). Coordinates: RD New
        metres. A trip uses at least one boarding."""
        return self.journeys(origin_xy, dest_xy, max_minutes=max_minutes,
                             chunk=chunk, track=False)["time"]

    def journeys(self, origin_xy, dest_xy, *, max_minutes: float = _PT.max_minutes,
                 chunk: int = 8, track: bool = True,
                 access: LegSpec | None = None,
                 egress: LegSpec | None = None) -> dict:
        """Time and, if `track`, the fare inputs of the fastest journey:
        {'time', 'rail_km', 'other_km', 'other_boardings', 'access_min',
        'egress_min'},
        each (origins x destinations) float32 (NaN where not reachable).
        access_min / egress_min: minutes of the bicycle ride of the access
        / egress leg (excluding fixed minutes; zero on foot), for metered
        tariffs and leg-wise time gates.

        access / egress: None walks; a LegSpec uses a bicycle for that leg
        (egress with hubs_only: OV-fiets at rail stops).

        rail_km / other_km: in-vehicle kilometres on rail and on other
        lines (bus, tram, metro, ferry); other_boardings: boardings onto
        other lines. They come from the time-optimal path, not from a
        fare-optimal one."""
        o = np.asarray(origin_xy, dtype=float)
        d = np.asarray(dest_xy, dtype=float)
        n_nodes = 2 * self.n_stops + self.n_line_stops
        o_ptr, o_idx, o_w, o_ride = self._links(o, access)
        u, v, w = self._edges
        origin_nodes = n_nodes + np.arange(len(o))
        rep = np.repeat(np.arange(len(o)), np.diff(o_ptr))
        total = n_nodes + len(o)
        graph = sparse.csr_matrix(
            (np.concatenate([w, o_w]),
             (np.concatenate([u, origin_nodes[rep]]),
              np.concatenate([v, o_idx]))), shape=(total, total))
        e_ptr, e_idx, e_w, e_ride = self._links(d, egress)
        names = ["time", "rail_km", "other_km", "other_boardings",
                 "access_min", "egress_min"] if track else ["time"]
        out = {k: np.full((len(o), len(d)), np.nan, dtype=np.float32)
               for k in names}
        has = np.diff(e_ptr) > 0
        starts = e_ptr[:-1][has]
        seg_len = np.diff(e_ptr)[has]
        seg_id = np.repeat(np.arange(len(starts)), seg_len)
        for s0 in range(0, len(o), chunk):
            src = origin_nodes[s0:s0 + chunk]
            res = csgraph.dijkstra(graph, directed=True, indices=src,
                                   limit=max_minutes,
                                   return_predecessors=track)
            dist, pred = res if track else (res, None)
            stop_t = dist[:, self.n_stops:2 * self.n_stops]
            vals = stop_t[:, e_idx] + e_w[None, :]
            if len(starts) == 0:
                continue
            best = np.minimum.reduceat(vals, starts, axis=1)
            time_block = np.full((len(src), len(d)), np.inf)
            time_block[:, has] = best
            time_block[time_block > max_minutes] = np.nan
            out["time"][s0:s0 + chunk] = time_block
            if not track:
                continue
            link_ride = np.zeros((len(src), total))
            for r_, o_ in enumerate(range(s0, s0 + len(src))):
                if access is None:                  # walking: no bicycle ride
                    break
                sl = slice(o_ptr[o_], o_ptr[o_ + 1])
                link_ride[r_, o_idx[sl]] = o_ride[sl]
            attrs = self._path_attributes(pred, total, link_ride)  # (c,total,4)
            zone_idx = np.flatnonzero(has)
            for r in range(len(src)):
                hit = vals[r] == np.repeat(best[r], seg_len)
                pos = np.flatnonzero(hit)
                first_seg, first = np.unique(seg_id[pos], return_index=True)
                stop_at = e_idx[pos[first]] + self.n_stops   # S1 node
                z = zone_idx[first_seg]
                good = np.isfinite(best[r][first_seg]) \
                    & (best[r][first_seg] <= max_minutes)
                for j, name in enumerate(names[1:5]):
                    out[name][s0 + r, z[good]] = attrs[r, stop_at[good], j]
                if egress is not None:                # bicycle egress ride
                    out["egress_min"][s0 + r, z[good]] = \
                        e_ride[pos[first]][good]
        return out

    def _path_attributes(self, pred: np.ndarray, total: int,
                         link_ride: np.ndarray | None = None) -> np.ndarray:
        """Cumulative (rail_km, other_km, other_boardings, access ride
        minutes) from the origin
        to every node along the shortest-path tree. pred: (c, total)
        predecessor matrix (-9999 for none). Pointer doubling: log(depth)
        vectorised passes instead of a walk per node."""
        c = pred.shape[0]
        n_nodes = 2 * self.n_stops + self.n_line_stops
        cum = np.zeros((c, total + 1, 4))
        anc = np.full((c, total + 1), total, dtype=np.int64)  # total = sink
        rows, nodes = np.nonzero(pred >= 0)
        p = pred[rows, nodes].astype(np.int64)
        keys = p * n_nodes + nodes
        in_graph = (p < n_nodes) & (nodes < n_nodes)         # not origin links
        idx = np.searchsorted(self._key_sorted, keys[in_graph])
        idx = np.minimum(idx, len(self._key_sorted) - 1)
        match = self._key_sorted[idx] == keys[in_graph]
        edge = self._key_to_edge[idx]
        r_ok, n_ok = rows[in_graph][match], nodes[in_graph][match]
        cum[r_ok, n_ok, :3] = self._attr[edge[match]]
        if link_ride is not None:               # origin -> first stop link
            first = ~in_graph
            cum[rows[first], nodes[first], 3] = link_ride[rows[first],
                                                        nodes[first]]
        anc[rows, nodes] = p
        for _ in range(64):
            live = anc[:, :total] != total
            if not live.any():
                break
            ri = np.arange(c)[:, None]
            cum[:, :total] += cum[ri, anc[:, :total]] * live[..., None]
            anc[:, :total] = anc[ri, anc[:, :total]]
        return cum[:, :total]
