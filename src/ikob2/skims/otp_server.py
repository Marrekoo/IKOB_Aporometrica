"""
A local OpenTripPlanner (OTP 2) server on OSM + GTFS.

OTP is a jar: with Java 21 it needs no Docker and no sudo. The graph is
built once from the OSM extract and the GTFS feed (the transit service
window is limited to keep memory down), then served on localhost.
OTP answers itinerary requests (`/otp/gtfs/v1` GraphQL and the
Transmodel API), which is what public transport fares need: legs with
modes, distances and times. It has no origin-destination matrix service;
bulk PT travel-time matrices stay with R5 (r5py).

Layout under `<data>/intermediate/otp/`:
    otp-<version>-shaded.jar     the program
    graph/                       inputs (links to the pbf and gtfs zip),
                                 build-config.json, router-config.json,
                                 and, after the build, graph.obj
    build.log, server.log, server.pid
"""

from __future__ import annotations

import json
import logging
import os
import signal
import subprocess
import time
import urllib.request
from pathlib import Path

from ikob2.params import DEFAULTS
from ikob2.utils.paths import DataLayout

logger = logging.getLogger(__name__)

PORT = DEFAULTS.servers.otp_port
# 2.9 and later are compiled for Java 25; 2.8.x runs on Java 21
JAR_VERSION = DEFAULTS.servers.otp_version
JAR_URL = ("https://repo1.maven.org/maven2/org/opentripplanner/otp-shaded/"
           "{v}/otp-shaded-{v}.jar")


def jar_path(layout: DataLayout, version: str = JAR_VERSION) -> Path:
    return layout.otp_dir() / f"otp-{version}-shaded.jar"


def prepare(layout: DataLayout, osm_pbf, gtfs_zip: str | Path,
            *, service_start: str, service_end: str) -> Path:
    """Graph directory with links to the inputs and the two config files.
    osm_pbf: one path or several (OTP merges them). service_start /
    service_end (ISO dates) bound the GTFS service days loaded into the
    graph. Links and archives from a previous prepare are replaced; a
    saved graph.obj is removed because it no longer matches."""
    graph = layout.otp_dir() / "graph"
    graph.mkdir(parents=True, exist_ok=True)
    for old in list(graph.glob("*.pbf")) + list(graph.glob("*.zip")) \
            + list(graph.glob("graph.obj")):
        old.unlink()
    pbfs = [osm_pbf] if isinstance(osm_pbf, (str, Path)) else list(osm_pbf)
    for src in (*pbfs, gtfs_zip):
        src = Path(src).resolve()
        dest = graph / src.name
        if dest.is_symlink() or dest.exists():
            dest.unlink()
        dest.symlink_to(src)
    (graph / "build-config.json").write_text(json.dumps({
        "transitServiceStart": service_start,
        "transitServiceEnd": service_end,
        "osmDefaults": {"timeZone": "Europe/Amsterdam"},
    }, indent=1))
    (graph / "router-config.json").write_text(json.dumps({
        "routingDefaults": {"walk": {"speed": 1.33}, "bicycle": {"speed": 4.44}}
    }, indent=1))
    return graph


def build(layout: DataLayout, *, heap: str = "11G",
          version: str = JAR_VERSION) -> None:
    jar = jar_path(layout, version)
    if not jar.exists():
        raise FileNotFoundError(f"{jar} missing; download it from "
                                f"{JAR_URL.format(v=version)}")
    log = layout.otp_dir() / "build.log"
    with open(log, "ab") as fh:
        rc = subprocess.call(
            ["java", f"-Xmx{heap}", "-jar", str(jar), "--build", "--save",
             str(layout.otp_dir() / "graph")], stdout=fh,
            stderr=subprocess.STDOUT)
    if rc != 0:
        raise RuntimeError(f"OTP graph build failed (exit {rc}); see {log}")


def pid_file(layout: DataLayout) -> Path:
    return layout.otp_dir() / "server.pid"


def status(*, port: int = PORT) -> bool:
    try:
        with urllib.request.urlopen(
                f"http://localhost:{port}/otp", timeout=3) as r:
            return r.status == 200
    except Exception:                                    # noqa: BLE001
        return False


def start(layout: DataLayout, *, heap: str = DEFAULTS.servers.otp_heap, port: int = PORT,
          version: str = JAR_VERSION, wait_seconds: float = 600.0) -> int:
    if status(port=port):
        raise RuntimeError("An OTP server already answers on the port.")
    graph = layout.otp_dir() / "graph"
    if not (graph / "graph.obj").exists():
        raise FileNotFoundError("graph.obj missing; run build first.")
    log = layout.otp_dir() / "server.log"
    fh = open(log, "ab")
    proc = subprocess.Popen(
        ["java", f"-Xmx{heap}", "-jar", str(jar_path(layout, version)),
         "--load", str(graph), "--port", str(port)],
        stdout=fh, stderr=subprocess.STDOUT)
    pid_file(layout).write_text(str(proc.pid))
    deadline = time.time() + wait_seconds
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"OTP exited early; see {log}")
        if status(port=port):
            return proc.pid
        time.sleep(2.0)
    raise TimeoutError(f"OTP did not answer within {wait_seconds}s.")


def stop(layout: DataLayout) -> bool:
    pf = pid_file(layout)
    if not pf.exists():
        return False
    try:
        os.kill(int(pf.read_text()), signal.SIGTERM)
    except ProcessLookupError:
        pass
    pf.unlink()
    return True


# ── client ───────────────────────────────────────────────────────────

_PLAN_QUERY = """
{ plan(date: "%(date)s", time: "%(time)s",
       from: {lat: %(flat)s, lon: %(flon)s}, to: {lat: %(tlat)s, lon: %(tlon)s},
       transportModes: [{mode: TRANSIT}, {mode: WALK}], numItineraries: 1) {
    itineraries { duration
      legs { mode distance duration route { type } } } } }
"""
RAIL_MODES = {"RAIL"}


def plan(from_lonlat, to_lonlat, date: str, time_hhmm: str, *,
         url: str = f"http://localhost:{PORT}", timeout: float = 120.0):
    """Best itinerary between two points at a departure time, or None:
    {'minutes': ..., 'legs': [{'mode', 'km', 'minutes'}, ...]}."""
    import urllib.request

    body = json.dumps({"query": _PLAN_QUERY % {
        "date": date, "time": time_hhmm, "flon": from_lonlat[0],
        "flat": from_lonlat[1], "tlon": to_lonlat[0],
        "tlat": to_lonlat[1]}}).encode()
    req = urllib.request.Request(f"{url.rstrip('/')}/otp/gtfs/v1", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        payload = json.load(resp)
    if payload.get("errors"):
        raise RuntimeError(f"OTP: {payload['errors'][0].get('message')}")
    its = payload["data"]["plan"]["itineraries"]
    if not its:
        return None
    it = its[0]
    return {"minutes": it["duration"] / 60.0,
            "legs": [{"mode": leg["mode"], "km": leg["distance"] / 1000.0,
                      "minutes": leg["duration"] / 60.0}
                     for leg in it["legs"]]}


def journey_summary(itinerary: dict) -> dict:
    """Fare inputs of an itinerary, comparable with PtRouter.journeys:
    rail km, km on other transit (bus, tram, metro, ferry), the number of
    boardings onto them, and the walking km."""
    rail = other = walk = 0.0
    boardings = 0
    for leg in itinerary["legs"]:
        mode = leg["mode"]
        if mode == "WALK":
            walk += leg["km"]
        elif mode in RAIL_MODES:
            rail += leg["km"]
        else:
            other += leg["km"]
            boardings += 1
    return {"minutes": itinerary["minutes"], "rail_km": rail,
            "other_km": other, "other_boardings": boardings, "walk_km": walk}
