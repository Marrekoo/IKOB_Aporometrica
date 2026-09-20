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
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)
    params_mod.add_arguments(p)
    p.add_argument("--data-root", default=None)
    p.add_argument("--base", required=True, help="baseline run (S0)")
    p.add_argument("--a", required=True, help="numerator run (fare cut, S1)")
    p.add_argument("--b", required=True, help="denominator run (hubs, S2)")
    p.add_argument("--mode", default="pt_v2")
    p.add_argument("--value", default="accessibility")
    p.add_argument("--quantiles", nargs=2, type=float, default=[0.9, 0.1],
                   metavar=("HI", "LO"))
    p.add_argument("--tol", type=float, default=1e-9)
    args = p.parse_args(argv)
    lay = DataLayout(params_mod.data_root(args.data_root,
                                          params_mod.from_args(args)))
    read = lambda r: pd.read_csv(lay.run_dir(r) / "accessibility.csv")  # noqa: E731
    res = interchange_ratio(read(args.base), read(args.a), read(args.b),
                            args.mode, value=args.value, tol=args.tol,
                            quantiles=tuple(args.quantiles))
    out = lay.comparison_dir() / f"{args.a}_over_{args.b}"
    out.mkdir(parents=True, exist_ok=True)
    for name, df in res.items():
        df.to_csv(out / f"interchange_{name}.csv", index=False)
    pd.set_option("display.width", 160)
    print(res["summary"].round(4).T.to_string(header=False))
    print(f"\nTables written to {out}")


if __name__ == "__main__":
    main()
