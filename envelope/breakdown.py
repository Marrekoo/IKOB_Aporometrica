"""
Where the reference-budget envelope comes from: a breakdown of `low` and
`high` over the scenarios of the envelope script (`X_M calc.R`).

    python envelope/breakdown.py <out folder of X_M calc.R> [--csv FILE]

`data/envelope/reference_budgets.csv` holds, per household type and income
decile, `low` = min and `high` = max of X_M (EUR per home-based tour) over the
grid of the R script (its `xm_grid.csv`):

    gamma             0, 0.25, 0.5, 0.75, 1   share of the example basket given up
    N_source          N_min (8 tours/month, fixed), N_emp (ODiN), N_max (highest
                      decile rate of the household type)
    rent_scenario     lo, interp, hi          rent bracket between the anchors
    commute_scenario  none, average, worker   unreimbursed commuting deducted
    pt_basis          nibud_flat, chipkaart   PT tariff (commuting cost only)

For every cell this script reports

  * the scenario that gives `low` and the one that gives `high`;
  * one-at-a-time swings: the range of X_M when one assumption varies over its
    values and the others are held at the central scenario (gamma 0.5 or the
    only value, N_emp, rent interp, commute average, PT nibud_flat), as a share
    of the full width high - low;
  * `b_kind`: `gamma_indexed` (inside the Nibud anchors), or
    `b_bas_above_anchors` (above the highest anchor: the residual after the
    MINIMUM basket, an upper bound; gamma plays no role).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

FACTORS = ("gamma", "N_source", "rent_scenario", "commute_scenario", "pt_basis")
CENTRAL = {"N_source": "N_emp", "rent_scenario": "interp",
           "commute_scenario": "average", "pt_basis": "nibud_flat"}


def central_gamma(d: pd.DataFrame) -> float:
    g = sorted(d["gamma"].unique())
    return 0.5 if 0.5 in g else g[0]


def breakdown(grid: pd.DataFrame) -> pd.DataFrame:
    q = grid[(grid["point_kind"] == "quantile") & grid["X_M"].notna()]
    rows = []
    for (hh, pid), d in q.groupby(["hh_type", "point_id"]):
        lo, hi = d.loc[d["X_M"].idxmin()], d.loc[d["X_M"].idxmax()]
        centre = {**CENTRAL, "gamma": central_gamma(d)}
        width = hi["X_M"] - lo["X_M"]
        rec = {"household_type": hh, "income_class": pid.replace("Q", "D"),
               "b_kind": ", ".join(sorted(d["b_kind"].dropna().unique())),
               "low": lo["X_M"], "high": hi["X_M"], "width": width,
               "high_over_low": hi["X_M"] / lo["X_M"] if lo["X_M"] > 0 else float("nan")}
        base = d
        for f, v in centre.items():
            base = base[base[f] == v] if f != "gamma" else base
        cen = d
        for f, v in centre.items():
            cen = cen[cen[f] == v]
        rec["central"] = cen["X_M"].iloc[0] if len(cen) else float("nan")
        for f in FACTORS:
            others = {k: v for k, v in centre.items() if k != f}
            s = d
            for k, v in others.items():
                s = s[s[k] == v]
            swing = s["X_M"].max() - s["X_M"].min() if len(s) else float("nan")
            rec[f"swing_{f}"] = swing / width if width > 0 else 0.0
        for f in FACTORS:
            rec[f"low_{f}"] = lo[f]
            rec[f"high_{f}"] = hi[f]
        rec["N_low_end"], rec["N_high_end"] = lo["N"], hi["N"]
        rows.append(rec)
    out = pd.DataFrame(rows)
    order = {f"D{k}": k for k in range(1, 11)}
    return out.sort_values(["household_type", "income_class"],
                           key=lambda s: s.map(order) if s.name == "income_class"
                           else s).reset_index(drop=True)


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("out_dir", type=Path, help="the out/ folder of X_M calc.R")
    p.add_argument("--csv", type=Path, default=None)
    args = p.parse_args(argv)
    b = breakdown(pd.read_csv(args.out_dir / "xm_grid.csv"))
    if args.csv:
        b.to_csv(args.csv, index=False)
    pd.set_option("display.width", 200)
    show = ["household_type", "income_class", "b_kind", "low", "central", "high",
            "high_over_low", "low_N_source", "high_N_source", "low_gamma",
            "high_gamma", "N_low_end", "N_high_end"]
    print(b[show].round(2).to_string(index=False))
    print("\nOne-at-a-time swing as a share of the full width (others central):")
    sw = [f"swing_{f}" for f in FACTORS]
    print(b[["household_type", "income_class"] + sw].round(2).to_string(index=False))


if __name__ == "__main__":
    main()
