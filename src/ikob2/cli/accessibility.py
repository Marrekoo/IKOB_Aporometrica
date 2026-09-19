"""
Run the accessibility model for a study area.

    python -m ikob2.cli.accessibility \
        --kwb <wijkenbuurten_2022_v3.gpkg> --skims data/skims/utrecht_nl \
        --sector-jobs output/sector_jobs_2022.csv --out output/run_s0 \
        --modes car bike

Inputs it combines (see docs/pipeline.md): the skim store (origins are
the study buurten, destinations all buurten), the 44 household-type x
income segments (computed on the fly from KWB and the StatLine
snapshots), imputed sector jobs (`cli.segments jobs`), the reference
budgets, the Weibull time margins, and the car cost model.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from ikob2.domain.filter_config import INDEPENDENCE, CopulaSpec
from ikob2.run.accessibility import ModeMatrices, run_accessibility
from ikob2.segments import statline
from ikob2.segments.bridge import load_reference_budgets
from ikob2.segments.config import SegmentConfig
from ikob2.segments.lisa import sector_wages
from ikob2.segments.pipeline import run_pipeline
from ikob2.segments.time_margins import load_time_margins
from ikob2.segments.wfh import (
    lisa_sector_wfh_share,
    sbi_wfh_share,
    sector_education_mix,
    wfh_incidence_by_education,
)
from ikob2.skims.car import (
    ELECTRIC_CAR,
    FOSSIL_CAR,
    DetourModel,
    car_time_and_cost,
    crowfly_km,
)
from ikob2.skims.store import SkimStore

logger = logging.getLogger("ikob2.cli.accessibility")

CAR_MODELS = {"fossil": FOSSIL_CAR, "electric": ELECTRIC_CAR}


def _sector_tables(statline_dir: Path, wage_period: str, wfh_period: str):
    wage_raw = pd.read_csv(
        statline.snapshot_path(statline_dir, statline.WAGE_SNAPSHOT,
                               statline.WAGE_TABLE, wage_period),
        dtype={"BedrijfstakkenBranchesSBI2008": str})
    hw = pd.read_csv(statline.snapshot_path(
        statline_dir, statline.HOME_WORK_SNAPSHOT, statline.HOME_WORK_TABLE,
        wfh_period), dtype=str)
    hw["WerkzameBeroepsbevolking_1"] = pd.to_numeric(
        hw["WerkzameBeroepsbevolking_1"])
    se = pd.read_csv(statline.snapshot_path(
        statline_dir, statline.SECTOR_EDUCATION_SNAPSHOT,
        statline.SECTOR_EDUCATION_TABLE, "2010JJ00"),
        dtype={"Onderwijsniveau": str, "BedrijfstakkenSBI2008": str})
    sbi = sbi_wfh_share(wfh_incidence_by_education(hw),
                        sector_education_mix(se))
    jobs = wage_raw.assign(k=wage_raw["BedrijfstakkenBranchesSBI2008"]
                           .str.strip()).set_index("k")["Banen_1"]
    return sector_wages(wage_raw), lisa_sector_wfh_share(sbi, jobs)


def build_matrices(store, zones, modes, *, detour, car_model, parking_search,
                   max_unreachable=None):
    """Mode -> ModeMatrices over (store origins, all zones)."""
    codes = [str(c) for c in zones.codes]
    idx = {c: i for i, c in enumerate(codes)}
    o_idx = np.array([idx[o] for o in store.origins])
    xy = np.column_stack([zones.centroid_x, zones.centroid_y])
    urb = np.asarray(zones.attributes.get(
        "stedelijkheid_adressen_per_km2", np.full(len(codes), np.nan)))
    out = {}
    for mode in modes:
        t = store.combined(mode, "time", codes, near="near", far="far")
        if mode == "car":
            dist = detour.route_km(crowfly_km(xy[o_idx], xy))
            time, cost = car_time_and_cost(
                t, dist, car_model,
                origin_urbanisation=urb[o_idx] if parking_search else None,
                dest_urbanisation=urb if parking_search else None)
            out[mode] = ModeMatrices(time, cost, car_model.matrix_id)
        elif mode == "bike":
            out[mode] = ModeMatrices(t)
        else:
            raise SystemExit(f"Mode '{mode}' has no time margin yet "
                             f"(margins exist for bike, pt, car).")
    return out, codes


def cmd_run(args) -> None:
    from ikob2.data.geopackage import load_cbs_buurten

    logging.getLogger("ikob2.data.geopackage").setLevel(logging.ERROR)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    zones, _ = load_cbs_buurten(args.kwb)
    store = SkimStore.open(args.skims)
    detour = (DetourModel.load(args.detour) if args.detour
              else DetourModel.constant(1.3))
    if not args.detour:
        logger.warning("No calibrated detour model: using a constant 1.3 "
                       "route/crow-fly factor for car distances.")
    matrices, dest_codes = build_matrices(
        store, zones, args.modes, detour=detour,
        car_model=CAR_MODELS[args.car_model],
        parking_search=not args.no_parking_search)

    seg_cfg = SegmentConfig()
    segs = run_pipeline(args.kwb, args.statline, seg_cfg)
    pop = (segs.population_scaled if args.population_basis
           == "population_scaled" else segs.household_based)

    sector_jobs = pd.read_csv(args.sector_jobs, index_col=0)
    wage, wfh = _sector_tables(Path(args.statline), args.wage_period,
                               args.wfh_period)
    envelope = load_reference_budgets(args.budgets,
                                      censored=args.censored,
                                      legs_per_tour=args.legs_per_tour)
    margins = load_time_margins(args.margins)
    copula = (INDEPENDENCE if args.copula == "independence"
              else CopulaSpec("gumbel", args.theta))

    result = run_accessibility(
        origins=store.origins, destinations=dest_codes, populations=pop,
        sector_jobs=sector_jobs, wfh_share=wfh, sector_wage=wage,
        envelope=envelope, time_margins=margins, matrices=matrices,
        copula=copula, epsilon=args.epsilon)

    t = result.table
    t.to_csv(out_dir / "accessibility.csv", index=False)
    result.summary("income_class").to_csv(out_dir / "summary_income.csv")
    result.summary("household_type").to_csv(out_dir / "summary_household.csv")
    meta = {**result.meta, "created": dt.datetime.now().isoformat(
        timespec="seconds"), "args": {k: str(v) for k, v in vars(args).items()
                                      if k != "func"},
            "detour": detour.meta, "skim_meta": store.meta}
    (out_dir / "run.json").write_text(json.dumps(meta, indent=1, default=str))
    print(f"Wrote {len(t)} rows to {out_dir}.")
    print(result.summary("income_class")["accessibility"].unstack(0)
          .round(0).to_string())


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--log-level", default="INFO")
    p.add_argument("--kwb", required=True)
    p.add_argument("--skims", required=True, help="skim store directory")
    p.add_argument("--sector-jobs", required=True,
                   help="CSV from `cli.segments jobs`")
    p.add_argument("--out", required=True)
    p.add_argument("--statline", default="data/statline")
    p.add_argument("--budgets", default="data/envelope/reference_budgets.csv")
    p.add_argument("--margins", default="data/margins/S_T_work.csv")
    p.add_argument("--modes", nargs="+", default=["car", "bike"])
    p.add_argument("--legs-per-tour", type=float, default=1.0)
    p.add_argument("--censored", choices=["atom", "drop"], default="atom")
    p.add_argument("--population-basis", default="population_scaled",
                   choices=["population_scaled", "household_based"])
    p.add_argument("--copula", choices=["independence", "gumbel"],
                   default="independence")
    p.add_argument("--theta", type=float, default=1.5)
    p.add_argument("--car-model", choices=list(CAR_MODELS), default="fossil")
    p.add_argument("--no-parking-search", action="store_true")
    p.add_argument("--detour", default=None,
                   help="calibrated DetourModel JSON (default: 1.3)")
    p.add_argument("--wage-period", default="2022JJ00")
    p.add_argument("--wfh-period", default="2024JJ00")
    p.add_argument("--epsilon", type=float, default=1e-9)
    p.set_defaults(func=cmd_run)
    args = p.parse_args(argv)
    logging.basicConfig(level=args.log_level,
                        format="%(levelname)s %(name)s: %(message)s")
    args.func(args)


if __name__ == "__main__":
    main()
