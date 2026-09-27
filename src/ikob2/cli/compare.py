"""
Compare two accessibility runs.

    python -m ikob2.cli.compare --data-root <root> \
        elgeneidy_step45 elgeneidy_exp45
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from ikob2.run.compare import compare_runs
from ikob2 import params as params_mod
from ikob2.utils.paths import DataLayout


def _usage(run_dir):
    import json

    f = run_dir / "run.json"
    if not f.exists():
        return None
    return (json.loads(f.read_text()).get("scenario", {})
            .get("lime_usage", {}).get("by_segment"))


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)
    params_mod.add_arguments(p)
    p.add_argument("--data-root", default=None)
    p.add_argument("--value", default="accessibility")
    p.add_argument("--mode", default="pt_v2",
                   help="mode for the effectiveness table (needs runs made "
                        "with --report-usage)")
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
    ua = _usage(lay.run_dir(args.run_a))
    ub = _usage(lay.run_dir(args.run_b))
    if ua and ub:
        from ikob2.run.scenarios import effectiveness
        eff = {}
        for by in ("income_class", "household_type"):
            eff[by] = effectiveness(a, b, ua, ub, args.mode, by)
            eff[by].to_csv(out / f"effectiveness_{by}.csv")
        print(f"\nWho gains against what it costs ({args.mode}); share_ratio "
              f"> 1: more benefit than cost share\n")
        print(eff["income_class"].round(4).to_string())
    print(f"\nTables written to {out}")


if __name__ == "__main__":
    main()
