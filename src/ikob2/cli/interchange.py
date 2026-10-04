"""The interchangeability ratio R = gain(A) / gain(B) against a baseline.

    python -m ikob2.cli.interchange --data-root <root> \
        --base s0 --a s1 --b s2 --mode pt_v2

writes `interchange_pairs.csv` (one row per origin and segment),
`interchange_origins.csv` (dispersion within each origin) and
`interchange_summary.csv` (pooled) to outputs/comparisons/<a>_over_<b>.
"""

from __future__ import annotations

import argparse

import pandas as pd

from ikob2 import params as params_mod
from ikob2.run.interchange import interchange_ratio
from ikob2.utils.paths import DataLayout


def main(argv=None) -> None:
    """Compute the interchangeability ratio R = gain(A) / gain(B) against a
    baseline run and write the per-pair, per-origin and pooled tables to
    outputs/comparisons/<a>_over_<b>/."""
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)
    params_mod.add_arguments(p)
    p.add_argument("--data-root", default=None)
    p.add_argument("--base", required=True, help="baseline run (S0)")
    p.add_argument("--a", required=True, help="numerator run (fare cut, S1)")
    p.add_argument("--b", required=True, help="denominator run (hubs, S2)")
    p.add_argument("--mode", default=None, help="default analysis.mode")
    p.add_argument("--value", default=None, help="default analysis.value")
    p.add_argument("--quantiles", nargs=2, type=float, default=None,
                   metavar=("HI", "LO"),
                   help="default analysis.interchange_quantiles")
    p.add_argument("--tol", type=float, default=None,
                   help="R undefined where |gain(B)| <= tol x the baseline; "
                        "default analysis.interchange_rel_tol")
    args = p.parse_args(argv)
    prm = params_mod.from_args(args, {
        "mode": "analysis.mode", "value": "analysis.value",
        "quantiles": "analysis.interchange_quantiles",
        "tol": "analysis.interchange_rel_tol"})
    args.mode, args.value = prm.analysis.mode, prm.analysis.value
    args.quantiles = prm.analysis.interchange_quantiles
    args.tol = prm.analysis.interchange_rel_tol
    lay = DataLayout(params_mod.data_root(args.data_root, prm))
    read = lambda r: pd.read_csv(lay.run_dir(r) / "accessibility.csv")  # noqa: E731
    res = interchange_ratio(read(args.base), read(args.a), read(args.b),
                            args.mode, value=args.value, rel_tol=args.tol,
                            quantiles=tuple(args.quantiles),
                            min_segments=prm.analysis.interchange_min_segments)
    out = lay.comparison_dir() / f"{args.a}_over_{args.b}"
    out.mkdir(parents=True, exist_ok=True)
    for name, df in res.items():
        df.to_csv(out / f"interchange_{name}.csv", index=False)
    pd.set_option("display.width", 160)
    print(res["summary"].round(4).T.to_string(header=False))
    print(f"\nTables written to {out}")


if __name__ == "__main__":
    main()
