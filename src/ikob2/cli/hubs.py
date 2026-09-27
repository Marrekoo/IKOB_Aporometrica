"""Extra hubs for scenario S2.

    python -m ikob2.cli.hubs propose --data-root <root> \
        --base-accessibility outputs/runs/s0/accessibility.csv

Ranks the origin buurten by low baseline accessibility and low bicycle
ownership and places `siting.hub_density_factor - 1` times the existing number
of hubs of kind `siting.kind` there (spacing `siting.min_spacing_m`). Writes
`intermediate/hubs/utrecht_hubs_s2.csv`: hub, lat, lon, precision, source and
the buurt, its accessibility, bicycle share and score. Build the S2 skims with
the existing and the new hubs together (docs/scenarios.md).
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from ikob2 import params as params_mod
from ikob2.skims import hubs as hubs_mod
from ikob2.skims.hub_siting import propose_hubs
from ikob2.utils.paths import DataLayout

FLAGS = {"factor": "siting.hub_density_factor",
         "spacing": "siting.min_spacing_m",
         "access_mode": "siting.access_mode"}


def cmd_propose(args) -> None:
    from pyproj import Transformer

    from ikob2.data.geopackage import load_cbs_buurten
    from ikob2.segments.ownership import load_bike_ownership
    from ikob2.skims.store import SkimStore

    prm = params_mod.from_args(args, FLAGS)
    root = params_mod.data_root(args.data_root, prm)
    lay = DataLayout(root)
    logging.getLogger("ikob2.data.geopackage").setLevel(logging.ERROR)
    kwb = args.kwb or str(lay.kwb(prm.accessibility.kwb_year,
                                  prm.paths.kwb_version))
    zones, _ = load_cbs_buurten(kwb)
    store = SkimStore.open(args.skims or str(lay.skim_dir(args.study)))
    idx = {str(c): i for i, c in enumerate(zones.codes)}
    rows = [idx[o] for o in store.origins]

    acc = pd.read_csv(args.base_accessibility)
    mode = prm.siting.access_mode
    acc = acc[acc["mode"] == mode]
    if acc.empty:
        raise SystemExit(f"No mode '{mode}' in {args.base_accessibility}.")
    a = (acc["accessibility"] * acc["population"]).groupby(
        acc["buurtcode"]).sum() / acc["population"].groupby(
        acc["buurtcode"]).sum().where(lambda s: s > 0)
    bike = load_bike_ownership(args.bike_ownership or str(lay.bike_ownership()),
                               store.origins)
    cand = pd.DataFrame({
        "code": list(store.origins),
        "x": np.asarray(zones.centroid_x)[rows],
        "y": np.asarray(zones.centroid_y)[rows],
        "access": a.reindex(store.origins).to_numpy(),
        "bike_share": bike.reindex(store.origins).to_numpy()})

    files, kinds = list(prm.pt.hub_files), list(prm.pt.hub_kinds)
    hubs = hubs_mod.load_hubs(files, hubs_mod.search_dirs(root), kinds=kinds,
                              tariffs=prm.shared_bike.hub_tariffs)
    existing = hubs[hubs["kind"] == prm.siting.kind]
    n_new = int(round(len(existing) * (prm.siting.hub_density_factor - 1.0)))
    chosen = propose_hubs(cand, hubs_mod.hub_xy(hubs), n_new,
                          min_spacing_m=prm.siting.min_spacing_m,
                          access_weight=prm.siting.access_weight)
    lon, lat = Transformer.from_crs("EPSG:28992", "EPSG:4326",
                                    always_xy=True).transform(
        chosen["x"].to_numpy(), chosen["y"].to_numpy())
    out = pd.DataFrame({
        "hub": [f"S2 hub {c}" for c in chosen["code"]],
        "lat": np.round(lat, 6), "lon": np.round(lon, 6),
        "precision": "buurt_centroid",
        "source": f"S2 siting: low '{mode}' accessibility "
                  f"(weight {prm.siting.access_weight:g}) and low bicycle "
                  f"ownership, spacing {prm.siting.min_spacing_m:g} m",
        "buurtcode": chosen["code"], "access": chosen["access"].round(1),
        "bike_share": chosen["bike_share"].round(3),
        "score": chosen["score"].round(4)})
    target = Path(args.out) if args.out else lay.s2_hubs()
    target.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(target, index=False)
    print(f"{len(out)} new hubs (existing {len(existing)}) written to {target}")
    print(out[["hub", "access", "bike_share", "score"]].to_string(index=False))


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--log-level", default="INFO")
    params_mod.add_arguments(p)
    sub = p.add_subparsers(dest="command", required=True)
    s = sub.add_parser("propose", help="place the extra hubs of S2")
    s.add_argument("--data-root", default=None)
    s.add_argument("--study", default="utrecht_nl")
    s.add_argument("--kwb", default=None)
    s.add_argument("--skims", default=None)
    s.add_argument("--bike-ownership", default=None)
    s.add_argument("--base-accessibility", required=True,
                   help="accessibility.csv of the baseline run")
    s.add_argument("--out", default=None)
    s.add_argument("--factor", type=float, default=None)
    s.add_argument("--spacing", type=float, default=None)
    s.add_argument("--access-mode", default=None)
    s.set_defaults(func=cmd_propose)
    args = p.parse_args(argv)
    logging.basicConfig(level=args.log_level,
                        format="%(levelname)s %(name)s: %(message)s")
    args.func(args)


if __name__ == "__main__":
    main()
