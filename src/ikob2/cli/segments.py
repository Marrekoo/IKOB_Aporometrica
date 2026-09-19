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
    print(f"Wrote {p1} ({len(income)} rows) and {p2} ({len(children)} rows).")


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
    f.set_defaults(func=cmd_fetch)

    r = sub.add_parser("run", help="compute segments")
    r.add_argument("--kwb", required=True)
    r.add_argument("--statline", default="data/statline")
    r.add_argument("--out", default="output/nl_segments.gpkg")
    r.add_argument("--layer", default="buurt_segments")
    r.set_defaults(func=cmd_run)

    args = p.parse_args(argv)
    logging.basicConfig(level=args.log_level,
                        format="%(levelname)s %(name)s: %(message)s")
    args.func(args)


if __name__ == "__main__":
    main()
