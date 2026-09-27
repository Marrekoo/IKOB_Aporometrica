"""
Effect of each method choice (defaults.toml [envelope]) on the reference
budgets: from the settings of X_M calc.R (envelope/x_m_calc.toml), one
choice at a time, and the adopted method (the defaults) as a whole.

    python envelope/switches.py [--out envelope/results]

Every variant is built from the repository's source tables and published
ODiN aggregates, and expressed per one-way journey as the model reads it
(tables per tour divided by `accessibility.legs_per_tour`). Writes
switch_effects.csv (every variant x cell: low, central, high, EUR per
journey) and switch_summary.csv (per variant: the median ratio to the
baseline of low, central and high, over D2-D10 and over D2-D4, where the
money gate binds).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from ikob2.cli.envelope import build
from ikob2.params import DEFAULTS, load

ROOT = Path(__file__).resolve().parents[1] / "data" / "envelope"
R_SETTINGS = Path(__file__).resolve().parent / "x_m_calc.toml"
ADOPTED = "adopted method (defaults)"
VARIANTS = {
    "X_M calc.R settings": {},
    "unit: per journey": {"unit": "journey"},
    "n_lower: lowest decile": {"n_lower": "lowest_decile"},
    "income_bridge: per adult": {"income_bridge": "per_adult"},
    "aggregates: ODiN 2022-23": {"aggregates": "2022_2023"},
    "car_all_tariffs": {"car_all_tariffs": True},
    "price_base: 2022 euros": {"price_base": "2022"},
    "spread: gamma only": {"spread": "gamma"},
    "gamma_anchor: 0 (not adopted)": {"gamma_anchor": 0.0},
    ADOPTED: None,
}


def variant(changes: dict) -> pd.DataFrame:
    """The reference budgets of one variant, EUR per journey (None: the
    defaults; otherwise changes to the X_M calc.R settings)."""
    prm = DEFAULTS if changes is None else load(R_SETTINGS).with_values(
        {f"envelope.{k}": v for k, v in changes.items()})
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
    base = eff[eff["variant"] == next(iter(VARIANTS))].set_index(
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
    print("Median ratio to the baseline (EUR per journey as the model reads it):")
    print(summ.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
