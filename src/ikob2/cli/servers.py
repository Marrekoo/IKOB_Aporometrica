"""
Local routing servers on the OSM and GTFS data.

    python -m ikob2.cli.servers valhalla build      # tiles from the OSM extract
    python -m ikob2.cli.servers valhalla start|stop|status
    python -m ikob2.cli.servers otp prepare         # links + config
    python -m ikob2.cli.servers otp build           # graph (OSM + GTFS)
    python -m ikob2.cli.servers otp start|stop|status

Files live in <data-root>/intermediate/{valhalla,otp}. See docs/servers.md.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from ikob2.skims import otp_server, valhalla_server
from ikob2.utils.paths import DEFAULT_ROOT, DataLayout


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--data-root", default=str(DEFAULT_ROOT))
    p.add_argument("--log-level", default="INFO")
    sub = p.add_subparsers(dest="server", required=True)

    v = sub.add_parser("valhalla")
    v.add_argument("action", choices=["build", "start", "stop", "status"])
    v.add_argument("--osm", default=None)
    v.add_argument("--concurrency", type=int, default=5)

    o = sub.add_parser("otp")
    o.add_argument("action", choices=["prepare", "build", "start", "stop",
                                      "status"])
    o.add_argument("--osm", default=None)
    o.add_argument("--gtfs", default=None)
    o.add_argument("--service-start", default="2026-09-01")
    o.add_argument("--service-end", default="2026-09-30")
    o.add_argument("--build-heap", default="11G")
    o.add_argument("--heap", default="8G")

    args = p.parse_args(argv)
    logging.basicConfig(level=args.log_level,
                        format="%(levelname)s %(name)s: %(message)s")
    lay = DataLayout(Path(args.data_root))
    osm = args.osm or str(next((lay.inputs / "osm").glob("*.pbf")))

    if args.server == "valhalla":
        if args.action == "build":
            valhalla_server.build(lay, osm, concurrency=args.concurrency)
        elif args.action == "start":
            print("pid", valhalla_server.start(lay, concurrency=args.concurrency))
        elif args.action == "stop":
            print("stopped" if valhalla_server.stop(lay) else "not running")
        else:
            print("up" if valhalla_server.status(
                port=valhalla_server._port(lay)) else "down")
    else:
        gtfs = args.gtfs or str(next((lay.inputs / "gtfs").glob("*.zip")))
        if args.action == "prepare":
            print(otp_server.prepare(lay, osm, gtfs,
                                     service_start=args.service_start,
                                     service_end=args.service_end))
        elif args.action == "build":
            otp_server.build(lay, heap=args.build_heap)
        elif args.action == "start":
            print("pid", otp_server.start(lay, heap=args.heap))
        elif args.action == "stop":
            print("stopped" if otp_server.stop(lay) else "not running")
        else:
            print("up" if otp_server.status() else "down")


if __name__ == "__main__":
    main()
