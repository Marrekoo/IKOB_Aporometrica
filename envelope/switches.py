"""
Sensitivity of the reference budgets to each method choice
(defaults.toml [envelope]): every switch set to its alternative, one at a
time, against the defaults.

    python envelope/switches.py [--out envelope/results]

Every variant is built from the repository's source tables and published
ODiN aggregates, and expressed per one-way journey as the model reads it
(tables per tour divided by `accessibility.legs_per_tour`). Writes
switch_effects.csv (every variant x cell: low, central, high, EUR per
journey) and switch_summary.csv (per variant: the median ratio to the
defaults of low, central and high, over D2-D10 and over D2-D4, where the
money gate binds).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from ikob2.cli.envelope import build
from ikob2.params import DEFAULTS

ROOT = Path(__file__).resolve().parents[1] / "data" / "envelope"
BASE = "defaults"
VARIANTS = {
    BASE: {},
    "unit: per tour": {"unit": "tour"},
    "n_lower: fixed 8 tours": {"n_lower": "fixed"},
    "income_bridge: none": {"income_bridge": "none"},
    "aggregates: ODiN 2023": {"aggregates": "2023"},
    "car_all_tariffs: false": {"car_all_tariffs": False},
    "price_base: published": {"price_base": "published"},
    "spread: all assumptions": {"spread": "all"},
    "gamma_anchor: 0": {"gamma_anchor": 0.0},
}


def variant(changes: dict) -> pd.DataFrame:
    """The reference budgets of the defaults with `changes`, EUR per journey."""
    prm = DEFAULTS.with_values({f"envelope.{k}": v for k, v in changes.items()})
    t = build(ROOT / "sources", ROOT / "odin" / prm.envelope.aggregates, prm)["reference_budgets"]
    div = 1.0 if prm.envelope.unit == "journey" else prm.accessibility.legs_per_tour
    for c in ("low", "central", "high"):
        t[c] = t[c] / div
    return t


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--out", type=Path, default=Path(__file__).parent / "results")
    args = p.parse_args(argv)
    rows = [variant(ch).assign(variant=name) for name, ch in VARIANTS.items()]
    eff = pd.concat(rows, ignore_index=True)[
        ["variant", "household_type", "income_class", "low", "central", "high",
         "upper_bound", "gate_slack"]]
    base = eff[eff["variant"] == BASE].set_index(
        ["household_type", "income_class"])[["low", "central", "high"]]
    summ = []
    for name, d in eff.groupby("variant", sort=False):
        d = d.set_index(["household_type", "income_class"])[["low", "central", "high"]]
        r = (d / base).dropna()
        dec = r.index.get_level_values("income_class").str[1:].astype(int)
        rec = {"variant": name}
        for label, sel in (("D2-D10", dec >= 2), ("D2-D4", (dec >= 2) & (dec <= 4))):
            for c in ("low", "central", "high"):
                rec[f"{c}_{label}"] = float(np.median(r.loc[sel, c]))
        summ.append(rec)
    summ = pd.DataFrame(summ)
    args.out.mkdir(parents=True, exist_ok=True)
    eff.round(4).to_csv(args.out / "switch_effects.csv", index=False)
    summ.round(3).to_csv(args.out / "switch_summary.csv", index=False)
    pd.set_option("display.width", 200)
    print("Median ratio to the defaults (EUR per journey as the model reads it):")
    print(summ.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
