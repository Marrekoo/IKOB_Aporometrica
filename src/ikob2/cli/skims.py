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
import os
from pathlib import Path

import numpy as np

from ikob2 import params as params_mod
from ikob2.skims.build import build_time_skims
from ikob2.skims.router import MODES, R5Router, TimeRequest
from ikob2.skims.store import SkimStore
from ikob2.skims.walk import walk_time_matrix
from ikob2.skims.zones import coarse_cells, zone_points
from ikob2.utils.paths import DataLayout

logger = logging.getLogger("ikob2.cli.skims")

# command-line flag -> parameter it overrides (flags default to None)
BUILD_FLAGS = {"near_km": "skims.near_km", "far_cells": "skims.far_cells",
               "walk_model": "skims.walk_model", "departure": "skims.departure",
               "window": "skims.window_minutes",
               "block_size": "skims.block_size"}
CALIBRATE_FLAGS = {"osrm_url": "distance.osrm_url",
                   "origins": "distance.calibration_origins",
                   "far": "distance.calibration_far",
                   "near": "distance.calibration_near",
                   "seed": "distance.calibration_seed"}
DISTANCE_FLAGS = {"port": "servers.valhalla_port",
                  "radius_km": "distance.radius_km",
                  "origin_batch": "distance.origin_batch"}
PT_FLAGS = {"date": "pt.date", "walk_kmh": "pt.walk_kmh",
            "walk_detour": "pt.walk_detour",
            "max_access_min": "pt.max_access_min",
            "transfer_radius_m": "pt.transfer_radius_m",
            "wait_cap_min": "pt.wait_cap_min",
            "boarding_penalty_min": "pt.boarding_penalty_min",
            "rail_detour": "pt.rail_detour", "other_detour": "pt.other_detour",
            "max_minutes": "pt.max_minutes", "egress_hubs": "pt.egress_hubs",
            "bike_kmh": "bike_leg.kmh", "bike_detour": "bike_leg.detour",
            "bike_max_min": "bike_leg.max_minutes",
            "bike_fixed_min": "bike_leg.fixed_minutes"}


def resolve(args, flags):
    """Parameters of a command: file, --set and flags; the flag attributes
    on `args` are filled with the resolved values."""
    prm = params_mod.from_args(args, flags)
    for name, key in flags.items():
        setattr(args, name, prm.get(key))
    return prm


def cmd_build(args) -> None:
    """Create a skim store for the study municipalities and fill it with R5
    (car, bicycle) and zone-geometry (walking) travel times, block by block."""
    from ikob2.data.geopackage import load_cbs_buurten

    prm = resolve(args, BUILD_FLAGS)

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
            m, max_minutes=args.max_minutes or prm.skims.max_minutes[m],
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
                    speed_kmh=prm.skims.walk_kmh,
                    detour=prm.skims.walk_detour,
                    intrazonal_factor=prm.skims.intrazonal_factor,
                    max_minutes=requests["walk"].max_minutes)
                store.write_rows(name, "walk", "time", start, t)
    if routed:
        router = R5Router(args.osm, args.gtfs, max_memory=args.max_memory)
        build_time_skims(router, store, origins, layers, routed,
                         block_size=args.block_size)
    print(f"Skim store written to {args.out}: {len(origins)} origins, "
          f"layers {store.layer_names}, matrices {store.arrays()}")


def cmd_calibrate_detour(args) -> None:
    """Fit the route / crow-fly detour factor per distance band from OSRM
    routes of sampled pairs and save the detour model (JSON, plus the routed
    pairs as CSV)."""
    from ikob2.data.geopackage import load_cbs_buurten
    from ikob2.skims import osrm

    prm = resolve(args, CALIBRATE_FLAGS)
    if not args.out:
        root = args.data_root or os.environ.get(params_mod.ENV_ROOT) \
            or prm.paths.data_root
        if not root:
            raise SystemExit("Give --out or --data-root.")
        args.out = str(DataLayout(Path(root)).detour_model())
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    logging.getLogger("ikob2.data.geopackage").setLevel(logging.ERROR)
    zones, _ = load_cbs_buurten(args.kwb)
    codes = np.asarray(zones.codes, dtype=str)
    pts = zone_points(codes, zones.centroid_x, zones.centroid_y, zones.crs)
    study = np.isin(np.asarray(zones.municipality_code, dtype=str),
                    args.study)
    o, d = osrm.sample_pairs(pts, codes[study], n_origins=args.origins,
                             n_far=args.far, n_near=args.near,
                             seed=args.seed)
    logger.info("routing %d origins x %d destinations on %s", len(o), len(d),
                args.osrm_url)
    routes = osrm.routed_pairs(args.osrm_url, o, d)
    model = osrm.calibrate_from_routes(pts, routes)
    model.save(args.out)
    routes.to_csv(Path(args.out).with_suffix(".routes.csv"), index=False)
    print(f"{len(routes)} routed pairs -> {args.out}")
    print(f"bands (crow-fly km -> detour): "
          f"{list(zip(np.round(model.km, 1), np.round(model.factor, 3)))}")


def cmd_build_distance(args) -> None:
    """Fill the store's car `distance`: Valhalla routes near the origins, the
    detour model beyond and where no route is found."""
    from ikob2.data.geopackage import load_cbs_buurten
    from ikob2.skims import distance as dist_mod
    from ikob2.skims import valhalla_server
    from ikob2.skims.car import DetourModel

    resolve(args, DISTANCE_FLAGS)
    logging.getLogger("ikob2.data.geopackage").setLevel(logging.ERROR)
    store = SkimStore.open(args.store)
    zones, _ = load_cbs_buurten(args.kwb)
    codes = np.asarray(zones.codes, dtype=str)
    pts = zone_points(codes, zones.centroid_x, zones.centroid_y, zones.crs
                      ).set_index("id")
    origins = pts.loc[store.origins].reset_index()
    cells, _ = coarse_cells(codes, pts.loc[codes, "x"], pts.loc[codes, "y"],
                            np.asarray(zones.municipality_code, dtype=str))
    layers = {}
    for name in store.layer_names:
        ids = list(store.layer(name).destinations)
        layers[name] = (cells.set_index("id").loc[ids].reset_index()
                        if name == "far" else pts.loc[ids].reset_index())
    detour = DetourModel.load(args.detour)
    url = f"http://localhost:{args.port}"
    if not valhalla_server.status(port=args.port):
        raise SystemExit(f"No Valhalla server on port {args.port}: "
                         f"python -m ikob2.cli.servers valhalla start")
    stats = dist_mod.build_car_distance(
        store, origins, layers, detour,
        lambda o, d: valhalla_server.matrix(o, d, mode="car", url=url),
        radius_km=args.radius_km, origin_batch=args.origin_batch)
    print(f"distance layers written to {args.store}: {stats}")


def cmd_make_peak(args) -> None:
    """Write a peak-load copy of an OSM extract (maxspeed divided by a
    congestion factor per road class)."""
    from ikob2.skims import peak

    stats = peak.make_peak_extract(args.osm, args.out)
    print(f"Peak-load extract written to {args.out}")
    for cls, v in peak.coverage(stats).items():
        print(f"  {cls:15s} x{peak.PEAK_FACTORS[cls]:.2f}  ways {v['ways']:8d}"
              f"  rescaled {100 * v['share_rescaled']:5.1f}%")


def cmd_build_pt(args) -> None:
    """Fill a public transport mode of the store with the GTFS frequency model:
    time, rail and other km, boardings, and the riding minutes of bicycle
    legs when access or egress is by bicycle."""
    from ikob2.data.geopackage import load_cbs_buurten
    from ikob2.skims.gtfs_pt import LegSpec, PtRouter, load_peak_timetable
    from ikob2.skims.pt_build import build_pt_layer

    from ikob2.skims import hubs as hubs_mod

    prm = resolve(args, PT_FLAGS)
    if args.hub_file:
        try:
            pairs = [hubs_mod.parse_hub_file_arg(
                v, prm.shared_bike.hub_tariffs) for v in args.hub_file]
        except ValueError as exc:
            raise SystemExit(str(exc)) from None
        prm = prm.with_values({"pt.hub_files": [f for f, _ in pairs],
                               "pt.hub_kinds": [k for _, k in pairs]})
    args.window = args.window or prm.pt.window_h
    logging.getLogger("ikob2.data.geopackage").setLevel(logging.ERROR)
    store = SkimStore.open(args.store)
    zones, _ = load_cbs_buurten(args.kwb)
    codes = [str(c) for c in zones.codes]
    xy = np.column_stack([zones.centroid_x, zones.centroid_y])
    idx = {c: i for i, c in enumerate(codes)}
    o_xy = xy[[idx[o] for o in store.origins]]
    tt = load_peak_timetable(args.gtfs, args.date,
                             window_h=(args.window[0], args.window[1]))
    router = PtRouter(tt, walk_kmh=args.walk_kmh, walk_detour=args.walk_detour,
                      max_access_min=args.max_access_min,
                      transfer_radius_m=args.transfer_radius_m,
                      wait_cap_min=args.wait_cap_min,
                      boarding_penalty_min=args.boarding_penalty_min,
                      rail_detour=args.rail_detour,
                      other_detour=args.other_detour,
                      hub_walk_radius_m=prm.pt.hub_walk_radius_m)
    leg = LegSpec(kmh=args.bike_kmh, detour=args.bike_detour,
                  max_minutes=args.bike_max_min,
                  fixed_minutes=args.bike_fixed_min)
    access = leg if args.access == "bike" else None
    hub_files = []
    if args.egress == "bike" and args.egress_hubs == "file":
        root = args.data_root or os.environ.get(params_mod.ENV_ROOT) \
            or prm.paths.data_root
        hubs = hubs_mod.load_hubs(prm.pt.hub_files,
                                  hubs_mod.search_dirs(root),
                                  kinds=prm.pt.hub_kinds,
                                  tariffs=prm.shared_bike.hub_tariffs)
        if args.hub_kind:            # one tariff class per skim mode
            hubs = hubs[hubs["kind"] == args.hub_kind].reset_index(drop=True)
            if hubs.empty:
                raise SystemExit(f"No hubs of kind '{args.hub_kind}' in "
                                 f"{prm.pt.hub_files} ({prm.pt.hub_kinds}).")
        router.hubs_xy = hubs_mod.hub_xy(hubs)
        hub_files = sorted(set(hubs["source"]))
        logger.info("egress from %d hubs (%s)", len(hubs), ", ".join(hub_files))
    egress = (LegSpec(**{**leg.__dict__, "hubs_only": args.egress_hubs
                         in ("rail", "file")})
              if args.egress == "bike" else None)
    previous = store.meta.get(args.mode_name)
    if previous and previous.get("bike", {}).get("egress_hubs") not in (
            None, args.egress_hubs):
        raise SystemExit(
            f"Mode '{args.mode_name}' in {args.store} was built with egress "
            f"hubs '{previous['bike']['egress_hubs']}', not "
            f"'{args.egress_hubs}': give a new --mode-name (finished blocks "
            f"are not recomputed).")
    build_pt_layer(store, router, o_xy, codes, xy, mode=args.mode_name,
                   max_minutes=args.max_minutes, access=access,
                   egress=egress)
    store.set_meta(args.mode_name, {
        "access": args.access, "egress": args.egress,
        "bike": {"kmh": args.bike_kmh, "detour": args.bike_detour,
                 "max_min": args.bike_max_min,
                 "fixed_min": args.bike_fixed_min,
                 "egress_hubs": args.egress_hubs,
                 "hub_files": hub_files,
                 "hub_kind": args.hub_kind,
                 "hub_walk_radius_m": prm.pt.hub_walk_radius_m},
        "gtfs": str(args.gtfs), "date": args.date, "window_h": args.window,
        "walk_kmh": args.walk_kmh, "walk_detour": args.walk_detour,
        "max_access_min": args.max_access_min,
        "wait": "min(headway/2, %g) min per boarding" % args.wait_cap_min,
        "boarding_penalty_min": args.boarding_penalty_min,
        "rail_detour": args.rail_detour, "other_detour": args.other_detour})
    print(f"PT time layer 'all' written to {args.store}")


def cmd_inspect(args) -> None:
    """Print the origins, layers, arrays and metadata of a skim store."""
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
    """Command line entry point (`python -m ikob2.cli.skims`)."""
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--log-level", default="INFO")
    params_mod.add_arguments(p)
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
    b.add_argument("--near-km", type=float, default=None)
    b.add_argument("--far-cells", choices=["municipality", "none"],
                   default=None)
    b.add_argument("--walk-model", choices=["zone", "router"], default=None)
    b.add_argument("--departure", default=None)
    b.add_argument("--window", type=int, default=None,
                   help="PT departure window in minutes")
    b.add_argument("--max-minutes", type=int, default=None)
    b.add_argument("--max-memory", default=None, help="JVM heap, e.g. 11G")
    b.add_argument("--block-size", type=int, default=None)
    b.set_defaults(func=cmd_build)

    c = sub.add_parser("calibrate-detour",
                       help="fit crow-fly -> route distance factors via OSRM")
    c.add_argument("--kwb", required=True)
    c.add_argument("--study", nargs="+", required=True, metavar="GMxxxx")
    c.add_argument("--out", default=None,
                   help="detour model JSON (default: <root>/intermediate/"
                        "calibration/car_detour.json)")
    c.add_argument("--data-root", default=None)
    c.add_argument("--osrm-url", default=None,
                   help="demo server: light use only; self-host for more")
    c.add_argument("--origins", type=int, default=None)
    c.add_argument("--far", type=int, default=None)
    c.add_argument("--near", type=int, default=None)
    c.add_argument("--seed", type=int, default=None)
    c.set_defaults(func=cmd_calibrate_detour)

    d = sub.add_parser("build-distance",
                       help="car route distances via the local Valhalla")
    d.add_argument("store")
    d.add_argument("--kwb", required=True)
    d.add_argument("--detour", required=True, help="DetourModel JSON")
    d.add_argument("--port", type=int, default=None)
    d.add_argument("--radius-km", type=float, default=None)
    d.add_argument("--origin-batch", type=int, default=None)
    d.set_defaults(func=cmd_build_distance)

    m = sub.add_parser("make-peak",
                       help="OSM extract with congestion factors by road class")
    m.add_argument("--osm", required=True)
    m.add_argument("--out", required=True)
    m.set_defaults(func=cmd_make_peak)

    t = sub.add_parser("build-pt",
                       help="public transport times from GTFS (frequency model)")
    t.add_argument("store")
    t.add_argument("--kwb", required=True)
    t.add_argument("--gtfs", required=True)
    t.add_argument("--date", default=None,
                   help="a weekday inside the feed's validity")
    t.add_argument("--window", nargs=2, type=float, default=None,
                   metavar=("FROM_H", "TO_H"))
    t.add_argument("--walk-kmh", type=float, default=None)
    t.add_argument("--walk-detour", type=float, default=None)
    t.add_argument("--max-access-min", type=float, default=None)
    t.add_argument("--transfer-radius-m", type=float, default=None)
    t.add_argument("--wait-cap-min", type=float, default=None)
    t.add_argument("--boarding-penalty-min", type=float, default=None)
    t.add_argument("--rail-detour", type=float, default=None,
                   help="rail km = crow-fly between stops x this")
    t.add_argument("--other-detour", type=float, default=None,
                   help="bus/tram/metro km = crow-fly between stops x this")
    t.add_argument("--max-minutes", type=float, default=None)
    t.add_argument("--mode-name", default="pt",
                   help="name of the mode in the store: 'pt' for the plain "
                        "walk-walk journey, e.g. pt_bw (bicycle access, walk "
                        "egress), pt_wb, pt_bb")
    t.add_argument("--access", choices=["walk", "bike"], default="walk")
    t.add_argument("--egress", choices=["walk", "bike"], default="walk")
    t.add_argument("--egress-hubs", choices=["rail", "all", "file"],
                   default=None,
                   help="where a bicycle egress can start: hub locations "
                        "from files (pt.hub_files or --hub-file), rail stops, "
                        "or every stop")
    t.add_argument("--hub-file", action="append", default=[],
                   metavar="FILE:KIND",
                   help="hub locations (CSV with lat, lon, or the OV-fiets "
                        "JSON) and their tariff kind, e.g. "
                        "hubs/utrecht_hubs.csv:lime; relative paths under "
                        "<data root>/inputs or /intermediate; repeatable; "
                        "replaces pt.hub_files and pt.hub_kinds")
    t.add_argument("--hub-kind", default=None,
                   help="use only the hub files of this kind (pt.hub_kinds: "
                        "lime, ovfiets); the tariff differs by kind, so each "
                        "kind is its own skim mode (pt_wb_<kind>)")
    t.add_argument("--data-root", default=None)
    t.add_argument("--bike-kmh", type=float, default=None)
    t.add_argument("--bike-detour", type=float, default=None)
    t.add_argument("--bike-max-min", type=float, default=None,
                   help="longest bicycle leg (ride minutes)")
    t.add_argument("--bike-fixed-min", type=float, default=None,
                   help="unlock/park/return minutes added to a bicycle leg")
    t.set_defaults(func=cmd_build_pt)

    i = sub.add_parser("inspect", help="describe a skim store")
    i.add_argument("store")
    i.set_defaults(func=cmd_inspect)

    args = p.parse_args(argv)
    logging.basicConfig(level=args.log_level,
                        format="%(levelname)s %(name)s: %(message)s")
    args.func(args)


if __name__ == "__main__":
    main()
