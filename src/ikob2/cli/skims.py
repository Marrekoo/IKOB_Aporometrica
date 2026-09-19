"""
CLI for skims.

    python -m ikob2.cli.skims build --kwb <wijkenbuurten.gpkg> \
        --study GM0344 --osm netherlands.osm.pbf --gtfs gtfs-nl.zip \
        --modes car bike walk pt --out data/skims/utrecht

Origins are the buurten of the study municipalities; destinations are
all buurten in the KWB file. Destinations within --near-km of any
origin are routed individually (layer 'near'); the rest are routed as
municipality points (layer 'far'), and `SkimStore.combined` assembles a
full-resolution matrix on demand. Walking times come from zone
geometry (`--walk-model zone`) or the router.
"""

from __future__ import annotations

import argparse
import datetime as dt
import logging
from pathlib import Path

import numpy as np

from ikob2.skims.build import build_time_skims
from ikob2.skims.router import MODES, R5Router, TimeRequest
from ikob2.skims.store import SkimStore
from ikob2.skims.walk import walk_time_matrix
from ikob2.skims.zones import coarse_cells, zone_points

logger = logging.getLogger("ikob2.cli.skims")

DEFAULT_MAX_MINUTES = {"car": 120, "bike": 90, "walk": 30, "pt": 180}


def cmd_build(args) -> None:
    from ikob2.data.geopackage import load_cbs_buurten

    logging.getLogger("ikob2.data.geopackage").setLevel(logging.ERROR)
    zones, _ = load_cbs_buurten(args.kwb)
    muni = np.asarray(zones.municipality_code, dtype=str)
    is_origin = np.isin(muni, args.study)
    if not is_origin.any():
        raise SystemExit(f"No buurten in study municipalities {args.study}.")

    codes = np.asarray(zones.codes, dtype=str)
    area_m2 = np.nan_to_num(
        zones.attributes.get("oppervlakte_land_in_ha",
                             np.full(zones.n_zones, np.nan)), nan=0.0) * 1e4
    pts = zone_points(codes, zones.centroid_x, zones.centroid_y, zones.crs,
                      area_m2=area_m2)
    origins = pts[is_origin].reset_index(drop=True)

    if args.far_cells == "none":
        near_mask = np.ones(zones.n_zones, dtype=bool)
    else:
        d = np.hypot(pts["x"].to_numpy()[:, None]
                     - origins["x"].to_numpy()[None, :],
                     pts["y"].to_numpy()[:, None]
                     - origins["y"].to_numpy()[None, :]).min(axis=1)
        near_mask = (d <= args.near_km * 1000.0) | is_origin
    near = pts[near_mask].reset_index(drop=True)

    layers = {"near": near}
    cell_of = {}
    if args.far_cells != "none":
        cells, mapping = coarse_cells(codes, pts["x"], pts["y"], muni)
        layers["far"] = cells
        cell_of = {"far": mapping}
    logger.info("origins %d, near destinations %d, far cells %s",
                len(origins), len(near),
                len(layers["far"]) if "far" in layers else 0)

    modes = list(args.modes)
    departure = dt.datetime.fromisoformat(args.departure)
    requests = {
        m: TimeRequest(
            m, max_minutes=args.max_minutes or DEFAULT_MAX_MINUTES[m],
            departure=departure if m == "pt" else None,
            window_minutes=args.window)
        for m in modes}
    meta = {"kwb": str(args.kwb), "study": list(args.study),
            "osm": str(args.osm), "gtfs": [str(g) for g in args.gtfs],
            "near_km": args.near_km, "far_cells": args.far_cells,
            "walk_model": args.walk_model, "departure": args.departure,
            "created": dt.datetime.now().isoformat(timespec="seconds"),
            "variable_units": {"time": "minutes"}}
    store = SkimStore.create(args.out, list(origins["id"]),
                             {k: list(v["id"]) for k, v in layers.items()},
                             cell_of=cell_of, meta=meta)

    routed = {m: r for m, r in requests.items()
              if not (m == "walk" and args.walk_model == "zone")}
    if "walk" in requests and args.walk_model == "zone":
        for name, dests in layers.items():
            store.allocate(name, "walk", "time")
            for start in range(0, len(origins), args.block_size):
                sub = origins.iloc[start:start + args.block_size]
                t = walk_time_matrix(
                    sub[["x", "y"]].to_numpy(), dests[["x", "y"]].to_numpy(),
                    origin_codes=sub["id"], dest_codes=dests["id"],
                    origin_area_m2=sub["area_m2"],
                    max_minutes=requests["walk"].max_minutes)
                store.write_rows(name, "walk", "time", start, t)
    if routed:
        router = R5Router(args.osm, args.gtfs, max_memory=args.max_memory)
        build_time_skims(router, store, origins, layers, routed,
                         block_size=args.block_size)
    print(f"Skim store written to {args.out}: {len(origins)} origins, "
          f"layers {store.layer_names}, matrices {store.arrays()}")


def cmd_inspect(args) -> None:
    store = SkimStore.open(args.store)
    print(f"origins: {len(store.origins)}")
    for name in store.layer_names:
        info = store.layer(name)
        print(f"layer {name}: {len(info.destinations)} destinations"
              f"{' (coarse)' if info.cell_of else ''}")
    for layer, mode, var in store.arrays():
        a = store.array(layer, mode, var)
        done = store.done_rows(layer, mode, var)
        print(f"  {layer}/{mode}/{var}: {a.shape}, reachable "
              f"{100 * np.isfinite(a).mean():.1f}%, rows done {done}")
    print("meta:", store.meta)


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--log-level", default="INFO")
    sub = p.add_subparsers(dest="command", required=True)

    b = sub.add_parser("build", help="compute a skim store")
    b.add_argument("--kwb", required=True, help="CBS wijkenbuurten .gpkg")
    b.add_argument("--study", nargs="+", required=True, metavar="GMxxxx",
                   help="study municipality codes (origins)")
    b.add_argument("--osm", required=True, help="OSM .pbf")
    b.add_argument("--gtfs", nargs="*", default=[], help="GTFS zip(s)")
    b.add_argument("--modes", nargs="+", default=["car", "bike", "walk"],
                   choices=MODES)
    b.add_argument("--out", required=True)
    b.add_argument("--near-km", type=float, default=50.0)
    b.add_argument("--far-cells", choices=["municipality", "none"],
                   default="municipality")
    b.add_argument("--walk-model", choices=["zone", "router"], default="zone")
    b.add_argument("--departure", default="2026-09-01T08:00:00")
    b.add_argument("--window", type=int, default=60,
                   help="PT departure window in minutes")
    b.add_argument("--max-minutes", type=int, default=None)
    b.add_argument("--max-memory", default=None, help="JVM heap, e.g. 11G")
    b.add_argument("--block-size", type=int, default=25)
    b.set_defaults(func=cmd_build)

    i = sub.add_parser("inspect", help="describe a skim store")
    i.add_argument("store")
    i.set_defaults(func=cmd_inspect)

    args = p.parse_args(argv)
    logging.basicConfig(level=args.log_level,
                        format="%(levelname)s %(name)s: %(message)s")
    args.func(args)


if __name__ == "__main__":
    main()
