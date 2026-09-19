"""
Route distances from an OSRM server, for calibrating car distances.

r5py's matrices carry no distance, OSRM's table service does
(`annotations=distance`). The public demo server (router.project-osrm.org)
is meant for light use: keep to a few dozen requests, one per second (the
defaults here), and prefer a self-hosted OSRM (osrm-backend on the
Netherlands extract) for anything larger. Its default table limit is 100
coordinates per request.
"""

from __future__ import annotations

import json
import logging
import time
import urllib.parse
import urllib.request

import numpy as np
import pandas as pd

from ikob2.skims.car import DetourModel, calibrate_detour, crowfly_km

logger = logging.getLogger(__name__)

DEMO_URL = "https://router.project-osrm.org"
MAX_TABLE = 100


def osrm_table(base_url: str, origins: np.ndarray, dests: np.ndarray,
               *, profile: str = "driving", timeout: float = 60.0,
               retries: int = 3, pause: float = 1.0):
    """Route distance (km) and duration (min) matrices, NaN where OSRM
    finds no route. origins / dests are (n, 2) arrays of lon, lat; the
    request must respect the server's table limit (len(o) + len(d) <=
    100 on the demo server)."""
    o = np.asarray(origins, dtype=float)
    d = np.asarray(dests, dtype=float)
    if len(o) + len(d) > MAX_TABLE:
        raise ValueError(f"{len(o)} + {len(d)} coordinates exceed the "
                         f"table limit of {MAX_TABLE}.")
    coords = np.vstack([o, d])
    path = ";".join(f"{lon:.6f},{lat:.6f}" for lon, lat in coords)
    query = urllib.parse.urlencode({
        "annotations": "distance,duration",
        "sources": ";".join(str(i) for i in range(len(o))),
        "destinations": ";".join(str(len(o) + j) for j in range(len(d)))},
        safe=";,")
    url = f"{base_url.rstrip('/')}/table/v1/{profile}/{path}?{query}"
    last = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as resp:
                payload = json.load(resp)
            if payload.get("code") != "Ok":
                raise RuntimeError(f"OSRM: {payload.get('code')} "
                                   f"{payload.get('message', '')}")
            dist = np.array(payload["distances"], dtype=float,
                            ) / 1000.0
            dur = np.array(payload["durations"], dtype=float) / 60.0
            time.sleep(pause)
            return dist, dur
        except Exception as exc:               # noqa: BLE001
            last = exc
            time.sleep(pause * (attempt + 1))
    raise RuntimeError(f"OSRM request failed after {retries} attempts: "
                       f"{last}")


def routed_pairs(base_url: str, origins: pd.DataFrame, dests: pd.DataFrame,
                 *, origin_batch: int = 20, dest_batch: int = 80,
                 **kwargs) -> pd.DataFrame:
    """All origin x destination pairs, in batches within the table limit.
    Frames need id, lon, lat. Returns from_id, to_id, route_km, minutes
    (unreachable pairs dropped)."""
    rows = []
    for i in range(0, len(origins), origin_batch):
        o = origins.iloc[i:i + origin_batch]
        for j in range(0, len(dests), dest_batch):
            d = dests.iloc[j:j + dest_batch]
            dist, dur = osrm_table(base_url, o[["lon", "lat"]].to_numpy(),
                                   d[["lon", "lat"]].to_numpy(), **kwargs)
            ii, jj = np.meshgrid(np.arange(len(o)), np.arange(len(d)),
                                 indexing="ij")
            ok = np.isfinite(dist)
            rows.append(pd.DataFrame({
                "from_id": o["id"].to_numpy()[ii[ok]],
                "to_id": d["id"].to_numpy()[jj[ok]],
                "route_km": dist[ok], "minutes": dur[ok]}))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(
        columns=["from_id", "to_id", "route_km", "minutes"])


def calibrate_from_routes(points_xy: pd.DataFrame, routes: pd.DataFrame,
                          **kwargs) -> DetourModel:
    """DetourModel from routed pairs; points_xy has id, x, y (RD metres)."""
    xy = points_xy.set_index("id")[["x", "y"]]
    a = xy.loc[routes["from_id"]].to_numpy()
    b = xy.loc[routes["to_id"]].to_numpy()
    crow = np.hypot(a[:, 0] - b[:, 0], a[:, 1] - b[:, 1]) / 1000.0
    return calibrate_detour(crow, routes["route_km"].to_numpy(), **kwargs)


def sample_pairs(points: pd.DataFrame, origin_ids, *, n_origins: int = 20,
                 n_far: int = 80, n_near: int = 80, near_km: float = 15.0,
                 seed: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Origins from the study area and a destination sample that covers
    all distances: uniform over the country plus a set within `near_km`
    of the origins' centre (uniform sampling alone yields few short
    trips)."""
    rng = np.random.default_rng(seed)
    o = points[points["id"].isin(list(origin_ids))]
    o = o.iloc[rng.choice(len(o), min(n_origins, len(o)), replace=False)]
    cx, cy = o["x"].mean(), o["y"].mean()
    d_km = crowfly_km([[cx, cy]], points[["x", "y"]].to_numpy())[0]
    near_pool = points[(d_km <= near_km) & (d_km > 0.1)]
    near = near_pool.iloc[rng.choice(len(near_pool),
                                     min(n_near, len(near_pool)),
                                     replace=False)]
    far = points.iloc[rng.choice(len(points), n_far, replace=False)]
    dests = pd.concat([near, far]).drop_duplicates("id")
    dests = dests[~dests["id"].isin(o["id"])]
    return o.reset_index(drop=True), dests.reset_index(drop=True)
