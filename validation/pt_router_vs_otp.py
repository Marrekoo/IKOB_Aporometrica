"""
Validation of the frequency-model PT router against OpenTripPlanner
(docs/servers.md, "Validation of the frequency-model PT router").

    # needs a running OTP server (python -m ikob2.cli.servers otp start)
    python validation/pt_router_vs_otp.py sample --data-root <root> \
        --out validation/results/otp_vs_freq_national.csv
    python validation/pt_router_vs_otp.py summary \
        validation/results/otp_vs_freq_national.csv

`sample` draws random origins of the skim store and, per origin, random
destinations that the frequency model reaches in more than 15 minutes; for
each pair it plans OTP itineraries at several departure times and records
the mean, minimum and maximum OTP minutes and the mean rail km, other-transit
km and boardings, next to the store's `time`, `rail_km`, `other_km` and
`other_boardings`. `--bbox` restricts destinations to a lon/lat box (the
regional graph). `summary` prints the statistics of the documentation from a
sample file.

The samples behind the documented numbers are in validation/results/:
otp_vs_freq_regional.csv (14 origins x 10 destinations, destinations in
3.6-5.85 E, 51.75-52.95 N, regional graph, seed 3) and
otp_vs_freq_national.csv (20 x 12, national graph, seed 3); both with
departures 07:00-09:00 every 30 minutes on 2026-09-15. Pairs for which OTP
found no itinerary are left out.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd

DEPARTURES = ("07:00", "07:30", "08:00", "08:30", "09:00")


def sample(args) -> None:
    from ikob2.data.geopackage import load_cbs_buurten
    from ikob2.skims import otp_server as otp
    from ikob2.skims.store import SkimStore
    from ikob2.skims.zones import zone_points
    from ikob2.utils.paths import DataLayout

    lay = DataLayout(Path(args.data_root))
    z, _ = load_cbs_buurten(lay.kwb(2022))
    pts = zone_points(z.codes, z.centroid_x, z.centroid_y, z.crs).set_index("id")
    s = SkimStore.open(lay.skim_dir(args.study))
    t = s.block("all", "pt", "time")
    rk = s.block("all", "pt", "rail_km")
    ok = s.block("all", "pt", "other_km")
    ob = s.block("all", "pt", "other_boardings")
    dest = list(s.layer("all").destinations)
    lon, lat = pts.loc[dest, "lon"].to_numpy(), pts.loc[dest, "lat"].to_numpy()
    inside = np.ones(len(dest), bool)
    if args.bbox:
        w, e, so, n = args.bbox
        inside = (lon > w) & (lon < e) & (lat > so) & (lat < n)
    rng = np.random.default_rng(args.seed)
    rows = []
    for i in rng.choice(len(s.origins), args.origins, replace=False):
        cand = np.flatnonzero(inside & np.isfinite(t[i]) & (t[i] > 15))
        for j in rng.choice(cand, args.per_origin, replace=False):
            o, d = pts.loc[s.origins[i]], pts.loc[dest[j]]
            found = []
            for h in args.departures:
                it = otp.plan((o.lon, o.lat), (d.lon, d.lat), args.date, h)
                if it:
                    found.append(otp.journey_summary(it))
            if not found:
                continue
            rows.append(dict(
                o=s.origins[i], d=dest[j], freq_min=float(t[i, j]),
                otp_mean=np.mean([f["minutes"] for f in found]),
                otp_min=np.min([f["minutes"] for f in found]),
                otp_max=np.max([f["minutes"] for f in found]),
                freq_rail=float(rk[i, j]),
                otp_rail=np.mean([f["rail_km"] for f in found]),
                freq_other=float(ok[i, j]),
                otp_other=np.mean([f["other_km"] for f in found]),
                freq_b=float(ob[i, j]),
                otp_b=np.mean([f["other_boardings"] for f in found]),
                n=len(found)))
    pd.DataFrame(rows).to_csv(args.out, index=False)
    print(f"{len(rows)} pairs written to {args.out}")


def statistics(d: pd.DataFrame) -> dict:
    """The figures of docs/servers.md from a sample."""
    diff = d["freq_min"] - d["otp_mean"]
    return {
        "pairs": len(d),
        "time_corr": float(np.corrcoef(d["freq_min"], d["otp_mean"])[0, 1]),
        "time_mean_diff": float(diff.mean()),
        "time_median_diff": float(diff.median()),
        "time_mean_abs_diff": float(diff.abs().mean()),
        "rail_corr": float(np.corrcoef(d["freq_rail"], d["otp_rail"])[0, 1]),
        "rail_mean_diff": float((d["freq_rail"] - d["otp_rail"]).mean()),
        "other_corr": float(np.corrcoef(d["freq_other"], d["otp_other"])[0, 1]),
    }


def summary(args) -> None:
    d = pd.read_csv(args.sample)
    for k, v in statistics(d).items():
        print(f"{k:20s} {v:.3f}" if isinstance(v, float) else f"{k:20s} {v}")
    band = pd.cut(d["otp_mean"], [0, 60, 120, 180, 600])
    print((d["freq_min"] - d["otp_mean"]).groupby(band, observed=True)
          .agg(["count", "mean", "median"]).round(1).to_string())


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)
    s = sub.add_parser("sample", help="plan OTP itineraries for random pairs")
    s.add_argument("--data-root", default=os.environ.get("IKOB_DATA_ROOT"))
    s.add_argument("--study", default="utrecht_nl")
    s.add_argument("--date", default="2026-09-15")
    s.add_argument("--departures", nargs="+", default=list(DEPARTURES))
    s.add_argument("--origins", type=int, default=20)
    s.add_argument("--per-origin", type=int, default=12)
    s.add_argument("--seed", type=int, default=3)
    s.add_argument("--bbox", type=float, nargs=4, metavar=("W", "E", "S", "N"))
    s.add_argument("--out", required=True)
    s.set_defaults(func=sample)
    m = sub.add_parser("summary", help="statistics of a sample file")
    m.add_argument("sample")
    m.set_defaults(func=summary)
    args = p.parse_args(argv)
    if args.command == "sample" and not args.data_root:
        raise SystemExit("Give --data-root or set IKOB_DATA_ROOT.")
    args.func(args)


if __name__ == "__main__":
    main()
