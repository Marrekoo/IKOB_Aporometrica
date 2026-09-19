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

from ikob2.utils.paths import DataLayout

logger = logging.getLogger(__name__)

PORT = 8080
JAR_VERSION = "2.10.0"
JAR_URL = ("https://repo1.maven.org/maven2/org/opentripplanner/otp-shaded/"
           "{v}/otp-shaded-{v}.jar")


def jar_path(layout: DataLayout, version: str = JAR_VERSION) -> Path:
    return layout.otp_dir() / f"otp-{version}-shaded.jar"


def prepare(layout: DataLayout, osm_pbf: str | Path, gtfs_zip: str | Path,
            *, service_start: str, service_end: str) -> Path:
    """Graph directory with links to the inputs and the two config files.
    service_start / service_end (ISO dates) bound the GTFS service days
    loaded into the graph."""
    graph = layout.otp_dir() / "graph"
    graph.mkdir(parents=True, exist_ok=True)
    for src in (osm_pbf, gtfs_zip):
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


def start(layout: DataLayout, *, heap: str = "8G", port: int = PORT,
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
