"""
CLI for the household-type x income segment pipeline.

    python -m ikob2.cli.segments fetch --out data/statline
    python -m ikob2.cli.segments run --kwb "IKOB data/wijkenbuurten_2022_v3.gpkg" \
        --statline data/statline --out output/nl_segments.gpkg

`fetch` is the only step that needs the network; it stores CSV
snapshots of the two StatLine tables that `run` then reads.
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import replace
from pathlib import Path

from ikob2.segments import statline
from ikob2.segments.config import SegmentConfig
from ikob2.segments.pipeline import run_pipeline, write_gpkg


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
    cfg = _cfg(args)
    out = Path(args.out)
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
    print(f"Wrote {p1} ({len(income)} rows), {p2} ({len(children)} rows), "
          f"{p3} ({len(wages)} rows) and {p4} ({len(est)} rows).")


def cmd_run(args) -> None:
    cfg = _cfg(args)
    result = run_pipeline(args.kwb, args.statline, cfg)
    write_gpkg(result, args.kwb, args.out, cfg, layer=args.layer)
    result.diagnostics.to_csv(Path(args.out).with_suffix(".diagnostics.csv"),
                              index=False)
    print(f"Wrote {len(result.household_based)} buurten x "
          f"{len(cfg.segment_columns)} segments to {args.out}.")
    for k, v in result.report.items():
        print(f"  {k}: {v}")


def cmd_jobs(args) -> None:
    import pandas as pd

    from ikob2.segments import jobs_impute as ji
    from ikob2.segments.jobs import parse_legacy_jobs
    from ikob2.segments.kwb import read_kwb
    from ikob2.segments.lisa import (lisa_gemeente_of_buurten,
                                     parse_lisa_sectors)

    cfg = _cfg(args)
    kwb = read_kwb(args.kwb, cfg)
    raw = pd.read_excel(args.lisa, sheet_name="LISA Gemeenten per sector")
    train, target = (parse_lisa_sectors(raw, args.train_year),
                     parse_lisa_sectors(raw, args.year))
    gem = lisa_gemeente_of_buurten(kwb, target.index)
    legacy = pd.read_excel(args.legacy, sheet_name="buurten-arbeidsplaatsen",
                           header=2)
    jobs = parse_legacy_jobs(legacy, args.legacy_year).sum(axis=1)
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
    p.add_argument("--income-period", default="2022JJ00")
    p.add_argument("--children-period", default="2022JJ00")
    p.add_argument("--gemeenten", nargs="*", metavar="GMxxxx",
                   help="restrict the study area (default: all)")
    p.add_argument("--low-income-col", help="KWB low-income column override")
    p.add_argument("--log-level", default="INFO")
    sub = p.add_subparsers(dest="command", required=True)

    f = sub.add_parser("fetch", help="download StatLine snapshots")
    f.add_argument("--out", default="data/statline")
    f.add_argument("--wage-period", default="2022JJ00")
    f.add_argument("--kwb-table", default="85318NED",
                   help="KWB StatLine table for establishments per buurt "
                        "(85318NED = 2022)")
    f.set_defaults(func=cmd_fetch)

    r = sub.add_parser("run", help="compute segments")
    r.add_argument("--kwb", required=True)
    r.add_argument("--statline", default="data/statline")
    r.add_argument("--out", default="output/nl_segments.gpkg")
    r.add_argument("--layer", default="buurt_segments")
    r.set_defaults(func=cmd_run)

    j = sub.add_parser("jobs", help="impute LISA sector jobs onto buurten")
    j.add_argument("--kwb", required=True)
    j.add_argument("--lisa", required=True, help="LISA_Gemeenten_*.xlsx")
    j.add_argument("--legacy", required=True,
                   help="Alle_Zones_2030_2040.xlsx (buurt job totals)")
    j.add_argument("--education", required=True,
                   help="Ralph_Sahar_CBS_buurten_met_banen_naar_"
                        "opleidingsniveau.xlsx (2016 education shares)")
    j.add_argument("--establishments",
                   default="data/statline/kwb_establishments_85318NED.csv",
                   help="KWB establishment snapshot ('' to ignore)")
    j.add_argument("--establishment-weight", type=float, default=0.25,
                   help="weight of establishment shares in the buurt job "
                        "totals (0 = legacy totals only)")
    j.add_argument("--year", type=int, default=2022)
    j.add_argument("--train-year", type=int, default=2016,
                   help="LISA year matching the education shares")
    j.add_argument("--legacy-year", default="2018")
    j.add_argument("--out", default="output/sector_jobs_2022.csv")
    j.set_defaults(func=cmd_jobs)

    args = p.parse_args(argv)
    logging.basicConfig(level=args.log_level,
                        format="%(levelname)s %(name)s: %(message)s")
    args.func(args)


if __name__ == "__main__":
    main()
