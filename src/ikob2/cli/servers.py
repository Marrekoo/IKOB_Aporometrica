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

from ikob2.skims import otp_server, valhalla_server
from ikob2 import params as params_mod
from ikob2.utils.paths import DataLayout

FLAGS = {"concurrency": "servers.valhalla_concurrency",
         "service_start": "servers.otp_service_start",
         "service_end": "servers.otp_service_end",
         "build_heap": "servers.otp_build_heap", "heap": "servers.otp_heap"}


def main(argv=None) -> None:
    """Command line entry point (`python -m ikob2.cli.servers`): build, start,
    stop or query the local Valhalla and OpenTripPlanner servers."""
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)
    params_mod.add_arguments(p)
    p.add_argument("--data-root", default=None)
    p.add_argument("--log-level", default="INFO")
    sub = p.add_subparsers(dest="server", required=True)

    v = sub.add_parser("valhalla")
    v.add_argument("action", choices=["build", "start", "stop", "status"])
    v.add_argument("--osm", default=None)
    v.add_argument("--concurrency", type=int, default=None,
                   help="default: servers.valhalla_concurrency")

    o = sub.add_parser("otp")
    o.add_argument("action", choices=["prepare", "build", "start", "stop",
                                      "status"])
    o.add_argument("--osm", nargs="*", default=None,
                   help="one or more .pbf (default: the four central "
                        "provinces in inputs/osm)")
    o.add_argument("--gtfs", default=None)
    o.add_argument("--bbox", nargs=4, type=float, default=None,
                   metavar=("LON_MIN", "LAT_MIN", "LON_MAX", "LAT_MAX"),
                   help="cut the GTFS to this box before linking it")
    o.add_argument("--service-start", default=None)
    o.add_argument("--service-end", default=None)
    o.add_argument("--build-heap", default=None)
    o.add_argument("--heap", default=None)

    args = p.parse_args(argv)
    logging.basicConfig(level=args.log_level,
                        format="%(levelname)s %(name)s: %(message)s")
    prm = params_mod.from_args(args, FLAGS)
    for name, key in FLAGS.items():       # flags default to the parameters
        if hasattr(args, name):
            setattr(args, name, prm.get(key))
    lay = DataLayout(params_mod.data_root(args.data_root, prm))
    osm = args.osm[0] if (args.osm and args.server == "valhalla") else (
        args.osm or None)
    if osm is None and args.server == "valhalla":
        osm = str(lay.inputs / "osm" / prm.paths.osm_national)

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
            osm = osm or [str(lay.inputs / "osm" / f"{r}.osm.pbf")
                          for r in prm.paths.osm_regions]
            if args.bbox:
                from datetime import date, timedelta

                from ikob2.skims.gtfs_subset import subset_gtfs
                d0 = date.fromisoformat(args.service_start)
                d1 = date.fromisoformat(args.service_end)
                days = [(d0 + timedelta(n)).isoformat()
                        for n in range((d1 - d0).days + 1)]
                out = lay.otp_dir() / "gtfs-region.zip"
                print("GTFS subset:", subset_gtfs(gtfs, out, args.bbox, days))
                gtfs = str(out)
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
