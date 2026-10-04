"""How many digits the paper's results deserve, from runs with perturbed inputs.

    python -m ikob2.cli.batch --plan paper/perturbation.toml --data-root <root> --jobs 2
    python -m ikob2.cli.precision --data-root <root>

Reads the perturbed runs pert_<draw>_<scenario> of the plan and the
unperturbed reference runs <reference prefix>_<scenario> (paper.precision_*),
and writes to outputs/comparisons/precision/:

  precision.csv  per statistic (baseline level and gain per person by income
                 class, total gain, gain per euro): the reference value, the
                 mean and standard deviation over the draws, the relative
                 standard deviation, the significant digits of the reference
                 (run.precision) and the reference rounded to them;
  draws.csv      the statistics of every draw.

Costs: the yearly public cost of a price cut from the reference scenario
run's run.json (--report-usage), and of the extra hubs from their hub file
(run.costs.hub_annual_cost); the cost is taken as given (not perturbed).
"""

from __future__ import annotations

import argparse
import json
import logging
import tomllib
from pathlib import Path

import pandas as pd

from ikob2 import params as params_mod
from ikob2.run.precision import statistics, summarise
from ikob2.utils.paths import DataLayout

logger = logging.getLogger("ikob2.cli.precision")


def _acc(lay: DataLayout, run: str) -> pd.DataFrame | None:
    f = lay.run_dir(run) / "accessibility.csv"
    return pd.read_csv(f) if f.exists() else None


def _cost(lay: DataLayout, run: str, costs) -> float | None:
    """Yearly public cost of a scenario run: price compensation plus hubs."""
    from ikob2.run.costs import hub_annual_cost

    f = lay.run_dir(run) / "run.json"
    if not f.exists():
        return None
    meta = json.loads(f.read_text())
    price = ((meta.get("scenario") or {}).get("cost") or {}).get("compensation_eur_year", 0.0)
    suffix = (meta.get("parameters", {}).get("shared_bike", {}).get("egress_suffix") or {})
    n = sum(len(pd.read_csv(lay.s2_hubs(s.lstrip("_")))) for s in suffix.values()
            if lay.s2_hubs(s.lstrip("_")).exists())
    total = price + (hub_annual_cost(costs, n)["public"] if n else 0.0)
    return total or None


def main(argv=None) -> None:
    """Command line entry point (`python -m ikob2.cli.precision`)."""
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)
    params_mod.add_arguments(p)
    p.add_argument("--data-root", default=None)
    p.add_argument("--plan", default=None, help="perturbation plan (default paper.precision_plan)")
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args(argv)
    logging.basicConfig(level=args.log_level, format="%(levelname)s %(name)s: %(message)s")
    prm = params_mod.from_args(args, {"plan": "paper.precision_plan"})
    lay = DataLayout(params_mod.data_root(args.data_root, prm))
    plan = tomllib.loads(Path(prm.paper.precision_plan).read_text())
    draws, scens = list(plan["specs"]), list(plan["scenarios"])
    mode = prm.analysis.mode
    ref_runs = {s: _acc(lay, f"{prm.paper.precision_reference}_{s}") for s in scens}
    missing = [s for s, t in ref_runs.items() if t is None]
    if missing:
        raise SystemExit(f"No reference runs for {missing}.")
    costs = {s: _cost(lay, f"{prm.paper.precision_cost_runs}_{s}", prm.costs) for s in scens}
    reference = statistics(ref_runs, mode, costs)
    rows = []
    for d in draws:
        runs = {s: _acc(lay, f"pert_{d}_{s}") for s in scens}
        if any(t is None for t in runs.values()):
            logger.warning("draw %s incomplete; skipped", d)
            continue
        rows.append(statistics(runs, mode, costs).assign(draw=d))
    if not rows:
        raise SystemExit("No complete draws.")
    draws_df = pd.concat(rows, ignore_index=True)
    out = lay.comparison_dir() / "precision"
    out.mkdir(parents=True, exist_ok=True)
    summary = summarise(reference, draws_df)
    summary.to_csv(out / "precision.csv", index=False)
    draws_df.to_csv(out / "draws.csv", index=False)
    pd.set_option("display.width", 200)
    print(summary.round(4).to_string(index=False))
    print(f"{draws_df['draw'].nunique()} draws; tables written to {out}")


if __name__ == "__main__":
    main()
