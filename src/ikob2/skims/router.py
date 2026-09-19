"""Routing interface and its r5py implementation."""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, Sequence

import numpy as np
import pandas as pd

MODES = ("car", "bike", "walk", "pt")


@dataclass(frozen=True)
class TimeRequest:
    """One travel-time matrix request (times in minutes)."""
    mode: str
    max_minutes: int = 120
    departure: dt.datetime | None = None      # public transport only
    window_minutes: int = 60                  # departures sampled after it
    percentile: int = 50
    walk_kmh: float = 4.8
    cycle_kmh: float = 16.0
    max_rides: int = 4

    def __post_init__(self):
        if self.mode not in MODES:
            raise ValueError(f"Unknown mode {self.mode!r}; use {MODES}.")
        if self.mode == "pt" and self.departure is None:
            raise ValueError("A pt request needs a departure datetime.")
        if not (1 <= self.percentile <= 99):
            raise ValueError("percentile must be in 1..99.")


class Router(Protocol):
    def time_matrix(self, origins: pd.DataFrame, destinations: pd.DataFrame,
                    request: TimeRequest) -> np.ndarray:
        """(len(origins), len(destinations)) minutes, NaN where not
        reachable. Frames have columns id, lon, lat."""


class R5Router:
    """Travel times with r5py (R5). The network (OSM, optionally GTFS) is
    built lazily on first use; building a national one takes many
    minutes and memory, pass `max_memory` (e.g. '11G') to bound it.

    Needs a Java 21 runtime and the optional `routing` extra (r5py).
    """

    def __init__(self, osm_pbf: str | Path, gtfs: Sequence[str | Path] = (),
                 max_memory: str | None = None):
        self.osm_pbf = Path(osm_pbf)
        self.gtfs = [Path(g) for g in gtfs]
        self.max_memory = max_memory
        self._network = None

    @property
    def network(self):
        if self._network is None:
            import sys
            if self.max_memory and "--max-memory" not in sys.argv:
                sys.argv.extend(["--max-memory", self.max_memory])
            from r5py import TransportNetwork
            self._network = TransportNetwork(
                str(self.osm_pbf), [str(g) for g in self.gtfs])
        return self._network

    def time_matrix(self, origins, destinations, request):
        import geopandas as gpd
        from r5py import TransportMode, TravelTimeMatrix

        def points(df):
            return gpd.GeoDataFrame(
                {"id": df["id"].astype(str).to_numpy()},
                geometry=gpd.points_from_xy(df["lon"], df["lat"]),
                crs="EPSG:4326")

        modes = {"car": [TransportMode.CAR],
                 "bike": [TransportMode.BICYCLE],
                 "walk": [TransportMode.WALK],
                 "pt": [TransportMode.TRANSIT]}[request.mode]
        kwargs = dict(
            transport_modes=modes,
            max_time=dt.timedelta(minutes=request.max_minutes),
            speed_walking=request.walk_kmh,
            speed_cycling=request.cycle_kmh,
            snap_to_network=True,
        )
        if request.mode == "pt":
            kwargs.update(
                departure=request.departure,
                departure_time_window=dt.timedelta(
                    minutes=request.window_minutes),
                percentiles=[request.percentile],
                max_public_transport_rides=request.max_rides,
                access_modes=[TransportMode.WALK],
                egress_modes=[TransportMode.WALK])
        o, d = points(origins), points(destinations)
        ttm = TravelTimeMatrix(self.network, origins=o, destinations=d,
                               **kwargs)
        col = [c for c in ttm.columns if c.startswith("travel_time")][0]
        return _to_matrix(ttm, col, o["id"].tolist(), d["id"].tolist())


def _to_matrix(ttm: pd.DataFrame, col: str, origin_ids, dest_ids) -> np.ndarray:
    oi = {c: i for i, c in enumerate(origin_ids)}
    di = {c: i for i, c in enumerate(dest_ids)}
    out = np.full((len(origin_ids), len(dest_ids)), np.nan, dtype=np.float32)
    rows = ttm["from_id"].map(oi).to_numpy()
    cols = ttm["to_id"].map(di).to_numpy()
    vals = ttm[col].to_numpy(dtype=float)
    ok = ~(np.isnan(rows.astype(float)) | np.isnan(cols.astype(float)))
    out[rows[ok].astype(int), cols[ok].astype(int)] = vals[ok]
    return out
