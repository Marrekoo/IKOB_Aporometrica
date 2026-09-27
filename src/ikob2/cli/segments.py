"""
CLI for the household-type x income segments, the jobs and the ODiN tables.

    python -m ikob2.cli.segments --data-root <root> fetch
    python -m ikob2.cli.segments --data-root <root> run
    python -m ikob2.cli.segments --data-root <root> jobs
    python -m ikob2.cli.segments --data-root <root> car-availability
    python -m ikob2.cli.segments --data-root <root> pt-spend

`fetch` is the only step that needs the network; it stores CSV snapshots
of the StatLine tables that the other steps and the runs read. Values come
from ikob2/defaults.toml (`--params`, `--set`, or the dedicated flags);
input and output paths default to the data folder layout
(utils.paths.DataLayout) and can each be given explicitly.
"""

from __future__ import annotations

import argparse
import logging
import os
from dataclasses import replace
from pathlib import Path

from ikob2 import params as params_mod
from ikob2.segments import statline
from ikob2.segments.config import SegmentConfig
from ikob2.segments.pipeline import run_pipeline, write_gpkg
from ikob2.utils.paths import DataLayout

# dedicated flag -> parameter
FLAGS = {
    "income_period": "segments.income_period",
    "children_period": "segments.children_period",
    "wage_period": "accessibility.wage_period",
    "wfh_period": "accessibility.wfh_period",
    "kwb_table": "jobs.kwb_establishments_table",
    "kwb_year": "accessibility.kwb_year",
    "year": "accessibility.jobs_year",
    "train_year": "jobs.train_year",
    "ikob_jobs_year": "jobs.ikob_jobs_year",
    "establishment_weight": "jobs.establishment_weight",
    "municipality": "ownership.municipality",
    "basis": "ownership.car_basis",
    "car_prior": "ownership.car_prior",
    "spend_prior": "ownership.pt_spend_prior",
}


def resolve(args) -> params_mod.Params:
    """Parameters (defaults.toml, --params, --set, flags); every flag
    attribute on `args` is filled with its resolved value."""
    prm = params_mod.from_args(args, FLAGS)
    for name, key in FLAGS.items():
        if hasattr(args, name):
            setattr(args, name, prm.get(key))
    return prm


def layout(args, prm) -> DataLayout | None:
    """The data folder layout, or None without a data root."""
    root = (args.data_root or os.environ.get(params_mod.ENV_ROOT)
            or prm.paths.data_root)
    return DataLayout(Path(root)) if root else None


def fill(args, lay: DataLayout | None, **defaults) -> None:
    """Set unset path arguments from the layout; fail naming the flags that
    have neither a value nor a layout default."""
    missing = []
    for name, make in defaults.items():
        if getattr(args, name):
            continue
        if lay is None:
            missing.append(name)
        else:
            setattr(args, name, str(make(lay)))
    if missing:
        raise SystemExit(f"Give --{', --'.join(m.replace('_', '-') for m in missing)}"
                         f" or --data-root.")


def statline_dir(lay: DataLayout | None) -> Path:
    """StatLine snapshots: <root>/cache/statline."""
    if lay is None:
        raise SystemExit("Give --statline (or --out for fetch) or --data-root.")
    return lay.statline()


def _cfg(args) -> SegmentConfig:
    cfg = SegmentConfig(
        income_period=args.income_period,
        children_period=args.children_period,
        gemeente_codes=(tuple(args.gemeenten) if args.gemeenten else None),
    )
    if args.low_income_col:
        cfg = replace(cfg, kwb_vars={**cfg.kwb_vars,
                                     "p_hh_low_income": args.low_income_col})
    return cfg


def cmd_fetch(args) -> None:
    prm = resolve(args)
    cfg = _cfg(args)
    out = Path(args.out) if args.out else statline_dir(layout(args, prm))
    out.mkdir(parents=True, exist_ok=True)
    income = statline.fetch_income_seed(
        cfg.income_period, population_key=cfg.income_population_key,
        total_col=cfg.income_total_col,
        decile_cols=list(cfg.income_decile_cols.values()),
        household_keys=list(cfg.household_key_map.values()))
    p1 = statline.snapshot_path(out, statline.INCOME_SNAPSHOT,
                                statline.INCOME_TABLE, cfg.income_period)
    income.to_csv(p1, index=False)
    children = statline.fetch_single_parent_counts(
        cfg.children_period, age_total=cfg.children_age_total)
    p2 = statline.snapshot_path(out, statline.CHILDREN_SNAPSHOT,
                                statline.CHILDREN_TABLE, cfg.children_period)
    children.to_csv(p2, index=False)
    wages = statline.fetch_sector_wages(args.wage_period)
    p3 = statline.snapshot_path(out, statline.WAGE_SNAPSHOT,
                                statline.WAGE_TABLE, args.wage_period)
    wages.to_csv(p3, index=False)
    est = statline.fetch_kwb_establishments(args.kwb_table)
    p4 = statline.snapshot_path(out, statline.KWB_ESTABLISHMENTS_SNAPSHOT,
                                args.kwb_table, "")
    est.to_csv(p4, index=False)
    hw = statline.fetch_home_working_by_education(args.wfh_period)
    p5 = statline.snapshot_path(out, statline.HOME_WORK_SNAPSHOT,
                                statline.HOME_WORK_TABLE, args.wfh_period)
    hw.to_csv(p5, index=False)
    se = statline.fetch_sector_education_jobs()
    p6 = statline.snapshot_path(out, statline.SECTOR_EDUCATION_SNAPSHOT,
                                statline.SECTOR_EDUCATION_TABLE, "2010JJ00")
    se.to_csv(p6, index=False)
    print(f"Wrote {p1} ({len(income)} rows), {p2} ({len(children)} rows), "
          f"{p3} ({len(wages)} rows), {p4} ({len(est)} rows), "
          f"{p5} ({len(hw)} rows) and {p6} ({len(se)} rows).")


def cmd_run(args) -> None:
    prm = resolve(args)
    lay = layout(args, prm)
    fill(args, lay, kwb=lambda lay: lay.kwb(args.kwb_year, prm.paths.kwb_version),
         out=DataLayout.segments_gpkg)
    args.statline = args.statline or str(statline_dir(lay))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    cfg = _cfg(args)
    result = run_pipeline(args.kwb, args.statline, cfg)
    write_gpkg(result, args.kwb, args.out, cfg, layer=args.layer)
    result.diagnostics.to_csv(Path(args.out).with_suffix(".diagnostics.csv"),
                              index=False)
    print(f"Wrote {len(result.household_based)} buurten x "
          f"{len(cfg.segment_columns)} segments to {args.out}.")
    for k, v in result.report.items():
        print(f"  {k}: {v}")


def cmd_car(args) -> None:
    from ikob2.segments.car_availability import (car_availability,
                                                 read_odin_persons)
    prm = resolve(args)
    fill(args, layout(args, prm), odin=DataLayout.odin,
         out=lambda lay: lay.car_availability(prm.paths.car_availability_study))
    persons = read_odin_persons(args.odin)
    t = car_availability(persons, args.municipality, basis=args.basis,
                         prior=args.car_prior)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    t.to_csv(args.out, index=False)
    print(f"Wrote {len(t)} segments to {args.out} "
          f"({len(persons)} adult respondents).")


def cmd_pt_spend(args) -> None:
    from ikob2.segments.pt_spend import pt_spend_by_decile, read_odin
    from ikob2.skims.pt_fare import PtFareModel

    prm = resolve(args)
    fill(args, layout(args, prm), odin=DataLayout.odin,
         out=lambda lay: lay.pt_spend(prm.paths.car_availability_study))
    legs = read_odin(args.odin)
    t = pt_spend_by_decile(legs, PtFareModel.from_params(prm.pt_fare),
                           args.municipality, args.spend_prior)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    t.to_csv(args.out, index=False)
    print(f"Wrote public transport fare spending by income decile to {args.out}.")
    print(t[["income_class", "trips_per_year", "mean_fare_eur",
             "spend_eur_year"]].round(2).to_string(index=False))


def cmd_jobs(args) -> None:
    import pandas as pd

    from ikob2.segments import jobs_impute as ji
    from ikob2.segments.jobs import parse_ikob_jobs
    from ikob2.segments.kwb import read_kwb
    from ikob2.segments.lisa import (lisa_gemeente_of_buurten,
                                     parse_lisa_sectors)

    prm = resolve(args)
    lay = layout(args, prm)
    fill(args, lay, kwb=lambda lay: lay.kwb(args.kwb_year, prm.paths.kwb_version),
         lisa=DataLayout.lisa, ikob_jobs=DataLayout.ikob_jobs,
         education=DataLayout.education_jobs,
         out=lambda lay: lay.sector_jobs(args.year))
    if args.establishments is None:
        args.establishments = str(statline.snapshot_path(
            statline_dir(lay), statline.KWB_ESTABLISHMENTS_SNAPSHOT,
            args.kwb_table, ""))
    cfg = _cfg(args)
    kwb = read_kwb(args.kwb, cfg)
    raw = pd.read_excel(args.lisa, sheet_name="LISA Gemeenten per sector")
    train, target = (parse_lisa_sectors(raw, args.train_year),
                     parse_lisa_sectors(raw, args.year))
    gem = lisa_gemeente_of_buurten(kwb, target.index)
    table = pd.read_excel(args.ikob_jobs, sheet_name="buurten-arbeidsplaatsen",
                          header=2)
    jobs = parse_ikob_jobs(table, args.ikob_jobs_year).sum(axis=1)
    edu = ji.parse_education_shares(pd.read_excel(args.education))
    cov = ji.buurt_covariates(kwb, edu)
    model = ji.fit_sector_model(
        train, ji.municipal_covariates(cov, jobs, gem))
    est = None
    if args.establishments:
        from ikob2.segments.establishments import read_establishments
        est = read_establishments(args.establishments)
    result = ji.impute_sector_jobs(
        jobs, gem, target, model, cov, establishments=est,
        establishment_weight=args.establishment_weight)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    result.jobs.rename_axis("buurtcode").to_csv(args.out)
    print(f"Wrote {len(result.jobs)} buurten x {result.jobs.shape[1]} "
          f"sectors ({args.year}) to {args.out}.")
    for k, v in result.report.items():
        print(f"  {k}: {v}")


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)
    params_mod.add_arguments(p)
    p.add_argument("--data-root", default=None,
                   help="data folder (utils.paths.DataLayout); fills the "
                        "input and output paths")
    p.add_argument("--income-period", default=None)
    p.add_argument("--children-period", default=None)
    p.add_argument("--gemeenten", nargs="*", metavar="GMxxxx",
                   help="restrict the study area (default: all)")
    p.add_argument("--low-income-col", help="KWB low-income column override")
    p.add_argument("--log-level", default="INFO")
    sub = p.add_subparsers(dest="command", required=True)

    f = sub.add_parser("fetch", help="download StatLine snapshots")
    f.add_argument("--out", default=None,
                   help="snapshot folder (default: <root>/cache/statline)")
    f.add_argument("--wage-period", default=None)
    f.add_argument("--wfh-period", default=None,
                   help="period of the home-working table (85718NED)")
    f.add_argument("--kwb-table", default=None,
                   help="StatLine KWB table with establishments by SBI "
                        "group (85318NED = 2022)")
    f.set_defaults(func=cmd_fetch)

    r = sub.add_parser("run", help="compute segments")
    r.add_argument("--kwb", default=None)
    r.add_argument("--kwb-year", type=int, default=None)
    r.add_argument("--statline", default=None)
    r.add_argument("--out", default=None)
    r.add_argument("--layer", default="buurt_segments")
    r.set_defaults(func=cmd_run)

    j = sub.add_parser("jobs", help="impute LISA sector jobs onto buurten")
    j.add_argument("--kwb", default=None)
    j.add_argument("--kwb-year", type=int, default=None)
    j.add_argument("--lisa", default=None, help="LISA_Gemeenten_*.xlsx")
    j.add_argument("--ikob-jobs", default=None,
                   help="IKOB job table (Alle_Zones_2030_2040.xlsx): buurt "
                        "job totals")
    j.add_argument("--education", default=None,
                   help="LISA 2016 jobs per buurt by education level "
                        "(Ralph_Sahar_...xlsx)")
    j.add_argument("--establishments", default=None,
                   help="KWB establishment snapshot ('' to ignore; default: "
                        "the snapshot of jobs.kwb_establishments_table)")
    j.add_argument("--establishment-weight", type=float, default=None,
                   help="weight of establishment shares in the buurt job "
                        "totals (0 = IKOB job table only)")
    j.add_argument("--year", type=int, default=None)
    j.add_argument("--train-year", type=int, default=None,
                   help="LISA year matching the education shares")
    j.add_argument("--ikob-jobs-year", default=None,
                   help="year column of the IKOB job table")
    j.add_argument("--kwb-table", default=None)
    j.add_argument("--out", default=None)
    j.set_defaults(func=cmd_jobs)

    c = sub.add_parser("car-availability",
                       help="car availability per segment from ODiN")
    c.add_argument("--odin", default=None,
                   help="cleaned ODiN pool CSV (ODIN_22_23_clean.csv)")
    c.add_argument("--municipality", type=int, default=None,
                   help="CBS municipality number of the study area (344 = "
                        "Utrecht); local cells are shrunk to the national")
    c.add_argument("--basis", choices=["household_car", "car_and_licence"],
                   default=None)
    c.add_argument("--prior", type=float, default=None, dest="car_prior")
    c.add_argument("--out", default=None)
    c.set_defaults(func=cmd_car)

    q = sub.add_parser("pt-spend",
                       help="public transport fare spending per person and "
                            "year by income decile from ODiN (cost of fare "
                            "concessions)")
    q.add_argument("--odin", default=None)
    q.add_argument("--municipality", type=int, default=None)
    q.add_argument("--prior", type=float, default=None, dest="spend_prior",
                   help="persons of prior weight for the national rate")
    q.add_argument("--out", default=None)
    q.set_defaults(func=cmd_pt_spend)

    args = p.parse_args(argv)
    logging.basicConfig(level=args.log_level,
                        format="%(levelname)s %(name)s: %(message)s")
    args.func(args)


if __name__ == "__main__":
    main()
