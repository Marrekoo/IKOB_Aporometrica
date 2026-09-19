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

from ikob2.domain.filter_config import INDEPENDENCE, CopulaSpec, CurveSpec
from ikob2.run.accessibility import ModeMatrices, run_accessibility
from ikob2.segments import statline
from ikob2.segments.bridge import envelope_segment_names, load_reference_budgets
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
from ikob2.utils.paths import DataLayout

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
                   distance_store=None):
    """Mode -> ModeMatrices over (store origins, all zones).

    distance_store : optional second store holding the car `distance`
        variable (same origins and near/far layers), so that a scenario
        store with different travel times (e.g. peak load) can reuse the
        routed distances of the base store.
    """
    dstore = distance_store or store
    if distance_store is not None:
        for layer in ("near", "far"):
            if (distance_store.origins != store.origins
                    or distance_store.layer(layer).destinations
                    != store.layer(layer).destinations):
                raise ValueError(f"distance store and time store differ in "
                                 f"origins or the '{layer}' layer.")
    codes = [str(c) for c in zones.codes]
    idx = {c: i for i, c in enumerate(codes)}
    o_idx = np.array([idx[o] for o in store.origins])
    xy = np.column_stack([zones.centroid_x, zones.centroid_y])
    urb = np.asarray(zones.attributes.get(
        "stedelijkheid_adressen_per_km2", np.full(len(codes), np.nan)))
    out = {}
    for mode in modes:
        if mode not in ("car", "bike", "pt"):
            raise SystemExit(f"Mode '{mode}' has no time margin (margins "
                             f"exist for bike, pt, car).")
        if mode == "pt":
            # PT is computed at buurt level for every destination
            t = store.block("all", "pt", "time", destinations=codes)
            logger.warning("PT has no fare yet: its cost margin is not "
                           "applied (time-only gate).")
            out[mode] = ModeMatrices(t)
            continue
        t = store.combined(mode, "time", codes, near="near", far="far")
        if mode == "car":
            if ("near", "car", "distance") in dstore.arrays():
                # routed distances (local Valhalla) where available
                dist = dstore.combined("car", "distance", codes, near="near",
                                       far="far")
                fallback = detour.route_km(crowfly_km(xy[o_idx], xy))
                dist = np.where(np.isfinite(dist), dist, fallback)
            else:
                dist = detour.route_km(crowfly_km(xy[o_idx], xy))
            time, cost = car_time_and_cost(
                t, dist, car_model,
                origin_urbanisation=urb[o_idx] if parking_search else None,
                dest_urbanisation=urb if parking_search else None)
            out[mode] = ModeMatrices(time, cost, car_model.matrix_id)
        elif mode == "bike":
            out[mode] = ModeMatrices(t)
    return out, codes


def time_curve(shape: str, cutoff: float, calibration: str = "mean") -> CurveSpec:
    """A common time margin for the impedance-shape comparison: 'step' is
    the hard cut-off at `cutoff` minutes (isochrone); 'exponential' is
    calibrated to it: 'mean' gives the exponential the same mean
    acceptable time (rate 1/cutoff, the paper's moment matching), 'half'
    makes acceptance 50% at the cut-off (rate ln 2 / cutoff)."""
    if shape == "step":
        return CurveSpec("step", (float(cutoff),))
    if shape == "exponential":
        rate = (1.0 / cutoff if calibration == "mean"
                else float(np.log(2.0) / cutoff))
        return CurveSpec("exponential", (rate,))
    raise ValueError(f"Unknown time shape {shape!r}.")


def resolve_paths(args) -> None:
    """Fill unset paths from the data folder layout (--data-root)."""
    if args.data_root:
        lay = DataLayout(Path(args.data_root))
        args.kwb = args.kwb or str(lay.kwb(args.kwb_year))
        args.skims = args.skims or str(lay.skim_dir(args.study))
        args.sector_jobs = args.sector_jobs or str(
            lay.sector_jobs(args.jobs_year))
        args.out = args.out or str(lay.run_dir(args.run))
        args.statline = args.statline or str(lay.statline())
        if not args.detour and lay.detour_model().exists():
            args.detour = str(lay.detour_model())
        survey = lay.inputs / "survey" / "S_T_work.csv"
        if not args.margins and survey.exists():
            args.margins = str(survey)
    args.statline = args.statline or "data/statline"
    args.margins = args.margins or "data/margins/S_T_work.csv"
    missing = [n for n in ("kwb", "skims", "sector_jobs", "out")
               if not getattr(args, n)]
    if missing:
        raise SystemExit(f"Give --{', --'.join(m.replace('_', '-') for m in missing)}"
                         f" or --data-root (with --study and --run).")


def cmd_run(args) -> None:
    from ikob2.data.geopackage import load_cbs_buurten

    resolve_paths(args)

    logging.getLogger("ikob2.data.geopackage").setLevel(logging.ERROR)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    zones, _ = load_cbs_buurten(args.kwb)
    store = SkimStore.open(args.skims)
    dist_store = None
    if args.distance_study:
        dist_store = SkimStore.open(
            Path(args.skims).parent / args.distance_study)
    detour = (DetourModel.load(args.detour) if args.detour
              else DetourModel.constant(1.3))
    if not args.detour:
        logger.warning("No calibrated detour model: using a constant 1.3 "
                       "route/crow-fly factor for car distances.")
    matrices, dest_codes = build_matrices(
        store, zones, args.modes, detour=detour,
        car_model=CAR_MODELS[args.car_model],
        parking_search=not args.no_parking_search,
        distance_store=dist_store)

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
    envelope_arg = envelope
    if args.time_shape != "weibull":
        margins = {(m, w): time_curve(args.time_shape, args.cutoff,
                                      args.exp_calibration)
                   for m in args.modes for w in ("no_wfh", "wfh_possible")}
    if args.no_cost_gate:
        envelope_arg = None
    copula = (INDEPENDENCE if args.copula == "independence"
              else CopulaSpec("gumbel", args.theta))

    result = run_accessibility(
        origins=store.origins, destinations=dest_codes, populations=pop,
        sector_jobs=sector_jobs, wfh_share=wfh, sector_wage=wage,
        envelope=envelope_arg, time_margins=margins, matrices=matrices,
        copula=copula, epsilon=args.epsilon,
        segment_names=envelope_segment_names(envelope))

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
    p.add_argument("--data-root", default=None,
                   help="data folder (utils.paths.DataLayout); fills the "
                        "paths below from --study, --run and the years")
    p.add_argument("--study", default="utrecht_nl",
                   help="skim store name under intermediate/skims")
    p.add_argument("--run", default="run",
                   help="output folder name under outputs/runs")
    p.add_argument("--distance-study", default=None,
                   help="skim store (sibling folder) with the car distances, "
                        "e.g. utrecht_nl when --study is the peak store")
    p.add_argument("--kwb-year", type=int, default=2022)
    p.add_argument("--jobs-year", type=int, default=2022)
    p.add_argument("--kwb", default=None)
    p.add_argument("--skims", default=None, help="skim store directory")
    p.add_argument("--sector-jobs", default=None,
                   help="CSV from `cli.segments jobs`")
    p.add_argument("--out", default=None)
    p.add_argument("--statline", default=None)
    p.add_argument("--budgets", default="data/envelope/reference_budgets.csv")
    p.add_argument("--margins", default=None)
    p.add_argument("--modes", nargs="+", default=["car", "bike"])
    p.add_argument("--time-shape", choices=["weibull", "exponential", "step"],
                   default="weibull",
                   help="weibull: the survey fits by mode and job type; "
                        "exponential / step: one common curve (impedance-"
                        "shape comparison)")
    p.add_argument("--cutoff", type=float, default=45.0,
                   help="cut-off minutes for --time-shape step / exponential")
    p.add_argument("--exp-calibration", choices=["mean", "half"],
                   default="mean",
                   help="exponential calibrated to the cut-off: same mean "
                        "(rate 1/cutoff) or 50%% acceptance at the cut-off")
    p.add_argument("--no-cost-gate", action="store_true",
                   help="time-only accessibility (ignore the cost margin)")
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
