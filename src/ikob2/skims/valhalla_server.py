"""
A local Valhalla routing server on the OSM extract (no Docker, no sudo).

`pyvalhalla` ships the Valhalla executables, so the whole chain runs from
the project environment: build the admin database and the routing tiles
from the OSM extract once, then run `valhalla_service` as an HTTP server
on localhost. Its matrix service (`/sources_to_targets`) returns route
DISTANCE and time for car, bicycle and pedestrian costings, which r5py's
matrices lack; the service limits are raised because it is local.

Public transport is not served here: Valhalla's multimodal matrix is not
supported. PT stays with R5 (r5py) or OpenTripPlanner.
"""

from __future__ import annotations

import json
import logging
import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np

from ikob2.utils.paths import DataLayout

logger = logging.getLogger(__name__)

PORT = 8002
COSTINGS = {"car": "auto", "bike": "bicycle", "walk": "pedestrian"}
# Local limits. The default matrix limits are small; raising them is
# right for a local server, but a matrix over long distances needs a lot
# of memory (a 25 x 1000 national matrix grew the server to 10 GB and it
# was killed; even 10 x 200 pairs within 150 km did the same), so the
# matrix distance stays bounded: pairs further apart than 80 km crow-fly
# are rejected instead of run, and long distances come from the detour
# model. Keep batches small (about 5 x 50) and concurrency low.
_LIMITS = {"max_distance": 5_000_000.0, "max_locations": 20_000,
           "max_matrix_distance": 80_000.0,
           "max_matrix_location_pairs": 20_000}


def make_config(layout: DataLayout, *, concurrency: int = 6,
                port: int = PORT) -> dict:
    import valhalla

    vdir = layout.valhalla_dir()
    (vdir / "tiles").mkdir(parents=True, exist_ok=True)
    cfg = valhalla.get_config(tile_extract="", tile_dir=str(vdir / "tiles"),
                              verbose=True)
    cfg["mjolnir"].update({
        "admin": str(vdir / "admin.sqlite"),
        "timezone": str(vdir / "tz_world.sqlite"),
        "transit_dir": str(vdir / "transit"),
        "transit_feeds_dir": str(vdir / "transit_feeds"),
        "concurrency": concurrency,
        "max_cache_size": 4_000_000_000,
        "use_lru_mem_cache": True,
    })
    cfg["mjolnir"]["data_processing"]["use_admin_db"] = True
    cfg["httpd"]["service"]["listen"] = f"tcp://*:{port}"
    cfg["httpd"]["service"]["loopback"] = f"ipc://{vdir}/loopback"
    cfg["httpd"]["service"]["interrupt"] = f"ipc://{vdir}/interrupt"
    cfg["loki"]["service"] = {"proxy": f"ipc://{vdir}/loki"}
    cfg["thor"]["service"] = {"proxy": f"ipc://{vdir}/thor"}
    cfg["odin"]["service"] = {"proxy": f"ipc://{vdir}/odin"}
    for costing in COSTINGS.values():
        cfg["service_limits"][costing].update(_LIMITS)
    cfg["service_limits"]["skadi"] = cfg["service_limits"].get("skadi", {})
    return cfg


def config_path(layout: DataLayout) -> Path:
    return layout.valhalla_dir() / "valhalla.json"


def write_config(layout: DataLayout, **kwargs) -> Path:
    cfg = make_config(layout, **kwargs)
    path = config_path(layout)
    path.write_text(json.dumps(cfg, indent=1))
    return path


def _valhalla(*args: str, log: Path | None = None, wait: bool = True):
    cmd = [sys.executable, "-m", "valhalla", *args]
    out = open(log, "ab") if log else subprocess.DEVNULL
    proc = subprocess.Popen(cmd, stdout=out, stderr=subprocess.STDOUT)
    if wait:
        rc = proc.wait()
        if rc != 0:
            raise RuntimeError(f"{' '.join(args[:2])} failed (exit {rc}); "
                               f"see {log}")
    return proc


def build(layout: DataLayout, osm_pbf: str | Path, *, concurrency: int = 6):
    """Admin database and routing tiles from an OSM extract (long: the
    national extract takes tens of minutes and a few GB of memory)."""
    cfg = write_config(layout, concurrency=concurrency)
    log = layout.valhalla_dir() / "build.log"
    logger.info("Building admin database ...")
    _valhalla("valhalla_build_admins", "-c", str(cfg), str(osm_pbf), log=log)
    logger.info("Building tiles ...")
    _valhalla("valhalla_build_tiles", "-c", str(cfg), str(osm_pbf), log=log)


def pid_file(layout: DataLayout) -> Path:
    return layout.valhalla_dir() / "server.pid"


def start(layout: DataLayout, *, concurrency: int = 4,
          wait_seconds: float = 120.0) -> int:
    """Start valhalla_service in the background; returns its pid."""
    if status(layout, port=_port(layout)):
        raise RuntimeError("A Valhalla server already answers on the port.")
    cfg = config_path(layout)
    if not cfg.exists():
        raise FileNotFoundError(f"{cfg} missing; run build first.")
    log = layout.valhalla_dir() / "server.log"
    # run the executable itself (not the `python -m valhalla` wrapper) so
    # that the recorded pid is the server's and stop() ends it
    import valhalla as _v
    exe = Path(_v.__file__).parent / "bin" / "valhalla_service"
    proc = subprocess.Popen([str(exe), str(cfg), str(concurrency)],
                            stdout=open(log, "ab"),
                            stderr=subprocess.STDOUT)
    pid_file(layout).write_text(str(proc.pid))
    deadline = time.time() + wait_seconds
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"valhalla_service exited early; see {log}")
        if status(layout, port=_port(layout)):
            return proc.pid
        time.sleep(1.0)
    raise TimeoutError(f"Server did not answer within {wait_seconds}s.")


def _port(layout: DataLayout) -> int:
    cfg = json.loads(config_path(layout).read_text())
    return int(cfg["httpd"]["service"]["listen"].rsplit(":", 1)[1])


def stop(layout: DataLayout) -> bool:
    pf = pid_file(layout)
    if not pf.exists():
        return False
    pid = int(pf.read_text())
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    pf.unlink()
    return True


def status(layout: DataLayout | None = None, *, port: int = PORT,
           url: str | None = None) -> bool:
    base = url or f"http://localhost:{port}"
    try:
        with urllib.request.urlopen(f"{base}/status", timeout=3) as r:
            return r.status == 200
    except Exception:                                    # noqa: BLE001
        return False


# ── client ───────────────────────────────────────────────────────────

def matrix(origins: np.ndarray, dests: np.ndarray, *, mode: str = "car",
           url: str = f"http://localhost:{PORT}", timeout: float = 600.0):
    """Route distance (km) and time (minutes) matrices via the local
    server; NaN where no route exists. origins/dests: (n, 2) lon, lat."""
    o = np.asarray(origins, dtype=float)
    d = np.asarray(dests, dtype=float)
    body = {
        "sources": [{"lon": float(x), "lat": float(y)} for x, y in o],
        "targets": [{"lon": float(x), "lat": float(y)} for x, y in d],
        "costing": COSTINGS[mode], "units": "kilometers"}
    req = urllib.request.Request(
        f"{url.rstrip('/')}/sources_to_targets",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            payload = json.load(resp)
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"Valhalla {exc.code}: "
                           f"{exc.read().decode()[:200]}") from None
    dist = np.full((len(o), len(d)), np.nan, dtype=np.float64)
    tim = np.full_like(dist, np.nan)
    for row in payload["sources_to_targets"]:
        for cell in row:
            i, j = cell["from_index"], cell["to_index"]
            if cell.get("distance") is not None:
                dist[i, j] = cell["distance"]
                tim[i, j] = cell["time"] / 60.0
    return dist, tim
