"""
Compare two accessibility runs.

    python -m ikob2.cli.compare --data-root "/home/marco/IKOB data" \
        elgeneidy_step45 elgeneidy_exp45
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from ikob2.run.compare import compare_runs
from ikob2 import params as params_mod
from ikob2.utils.paths import DataLayout


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)
    params_mod.add_arguments(p)
    p.add_argument("--data-root", default=None)
    p.add_argument("--value", default="accessibility")
    p.add_argument("run_a")
    p.add_argument("run_b")
    args = p.parse_args(argv)
    lay = DataLayout(params_mod.data_root(args.data_root,
                                          params_mod.from_args(args)))
    a = pd.read_csv(lay.run_dir(args.run_a) / "accessibility.csv")
    b = pd.read_csv(lay.run_dir(args.run_b) / "accessibility.csv")
    res = compare_runs(a, b, value=args.value)
    out = lay.comparison_dir() / f"{args.run_a}__vs__{args.run_b}"
    out.mkdir(parents=True, exist_ok=True)
    for name, df in res.items():
        df.to_csv(out / f"{name}.csv", index=False)
    pd.set_option("display.width", 160)
    print(f"a = {args.run_a}, b = {args.run_b}\n")
    print(res["cells"].round(3).to_string(index=False), "\n")
    print(res["origins"].round(3).to_string(index=False), "\n")
    print(res["income"].round(3).pivot(index="income_class",
          columns="mode", values=["a", "b", "ratio_b_over_a"]).to_string())
    print(f"\nTables written to {out}")


if __name__ == "__main__":
    main()
