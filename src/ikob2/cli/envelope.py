"""
The reference-budget envelope from its sources (ikob2.envelope).

    python -m ikob2.cli.envelope --data-root <root> aggregates [--odin FILE ...]
    python -m ikob2.cli.envelope --data-root <root> build

`aggregates` reads ODiN (default: the `envelope.odin` files under
inputs/odin) and writes the aggregate tables to intermediate/envelope/odin/.
It is the only step that reads microdata. The aggregates of ODiN 2023 and
2022-23 are published in data/envelope/odin/ (seeded into
inputs/envelope/odin/), so `build` runs without the microdata.

`build` combines the source tables (inputs/envelope/sources, seeded by
`cli.layout create`) with the aggregates (default:
inputs/envelope/odin/<envelope.aggregates>) and the [envelope] switches, and
writes to intermediate/envelope/:
anchors.csv, envelope.csv (residuals per decile and rent scenario),
tour_bounds.csv, grid.csv (X_M for every scenario) and reference_budgets.csv
(EUR per journey with the defaults; a table per tour is divided by
`legs_per_tour` when the model loads it).
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

from ikob2 import params as params_mod
from ikob2.envelope import income, nibud, odin, xm
from ikob2.envelope.sources import load_sources
from ikob2.utils.paths import DataLayout


def _layout(args, prm) -> DataLayout:
    root = args.data_root or os.environ.get(params_mod.ENV_ROOT) or prm.paths.data_root
    if not root:
        raise SystemExit("Give --data-root or set IKOB_DATA_ROOT.")
    return DataLayout(Path(root))


def cmd_aggregates(args) -> None:
    """Aggregate ODiN into the tables of `envelope.odin` and write them."""
    prm = params_mod.from_args(args)
    lay = _layout(args, prm)
    files = args.odin or [lay.inputs / "odin" / f for f in prm.envelope.odin]
    src = load_sources(lay.envelope_sources())
    import pandas as pd

    data = pd.concat([odin.read_odin(f) for f in files], ignore_index=True)
    tables = odin.aggregates(data, src["odin_household_types"])
    out = Path(args.out) if args.out else lay.envelope_dir() / "odin"
    odin.write_aggregates(tables, out)
    print(f"ODiN aggregates of {len(files)} file(s) written to {out}")


def build(src_dir, agg_dir, prm) -> dict:
    """All stages from source tables and ODiN aggregates; returns the
    tables by name."""
    src = load_sources(src_dir)
    agg = odin.read_aggregates(agg_dir)
    anch = nibud.anchors(src, prm)
    env = nibud.quantile_envelope(anch, income.income_axis(src, prm), src, prm)
    bounds = xm.tour_bounds(agg, prm)
    bund = xm.bundles(agg, src, prm)
    grid = xm.grid(env, bounds, xm.commuting(agg, bund, src, prm), bund, src, prm)
    gate = xm.gate(grid, prm, agg["km_cdf"])
    return {"anchors": anch, "envelope": env, "tour_bounds": bounds,
            "bundles": bund, "grid": grid,
            "reference_budgets": xm.reference_budgets(gate, prm)}


def cmd_build(args) -> None:
    """Build the envelope and write its tables."""
    prm = params_mod.from_args(args)
    lay = _layout(args, prm)
    agg = (Path(args.aggregates) if args.aggregates
           else lay.envelope_aggregates(prm.envelope.aggregates))
    tables = build(lay.envelope_sources(), agg, prm)
    out = Path(args.out) if args.out else lay.envelope_dir()
    out.mkdir(parents=True, exist_ok=True)
    for name, df in tables.items():
        df.to_csv(out / f"{name}.csv", index=False)
    print(f"Envelope written to {out} (reference_budgets.csv: EUR per "
          f"{prm.envelope.unit}; aggregates {agg})")


def main(argv=None) -> None:
    """Command line entry point (`python -m ikob2.cli.envelope`)."""
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)
    params_mod.add_arguments(p)
    p.add_argument("--data-root", default=None)
    p.add_argument("--log-level", default="INFO")
    sub = p.add_subparsers(dest="command", required=True)
    a = sub.add_parser("aggregates", help="ODiN -> aggregate tables")
    a.add_argument("--odin", nargs="+", default=None, help="ODiN CSV file(s)")
    a.add_argument("--out", default=None)
    a.set_defaults(func=cmd_aggregates)
    b = sub.add_parser("build", help="sources + aggregates -> envelope")
    b.add_argument("--aggregates", default=None, help="folder of the ODiN aggregates")
    b.add_argument("--out", default=None)
    b.set_defaults(func=cmd_build)
    args = p.parse_args(argv)
    logging.basicConfig(level=args.log_level,
                        format="%(levelname)s %(name)s: %(message)s")
    args.func(args)


if __name__ == "__main__":
    main()
