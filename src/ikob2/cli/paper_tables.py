"""Tables of the paper from the runs of a specification grid.

    python -m ikob2.cli.paper_tables --data-root <root> \
        --tags m0 m0u m0s m1 m1p m2 m3t1.25 m3t1.5 m3t2 m3t4 m3tinf

Reads outputs/runs/<prefix>_<tag>_<scenario> for the baseline s0 and the
scenarios of --scenarios (default s1 s2c s2t s3c s3t s4), with prefix `sp`
(income-matched jobs, the full application) and `spc` (common jobs, the
controlled comparison), and `spt_<tag>_s0` (time only, for the gap; M3 uses
the M2 time-only run, M0u and M0s that of M0, M1c that of M1) and writes to
outputs/comparisons/specs/:

  baseline_by_spec.csv     accessibility, raw and normalised, atom, per
                           specification and income class;
  incidence_by_spec.csv    gain of each scenario by income class and household
                           type, per specification;
  interchange_by_spec.csv  R = gain(A) / gain(B) for each pair of --pairs,
                           over the origins of paper.pair_zones for that pair
                           (column origins_used)
                           (default s1:s2c, the citywide price cut over the
                           citywide hubs, and s4:s2t, both in the target
                           buurten): pooled dispersion (CV, interquantile
                           ratio, spread), median, undefined pairs, full and
                           controlled;
  interchange_by_class.csv R per income class: the shares of segments where B
                           adds nothing (R undefined) and where A adds nothing
                           (R = 0), the median R, and the median R where both
                           add something;
  gap_by_spec.csv          the reachability gap at s0 and after the price
                           scenarios of --gap-scenarios (default s1 s4), and
                           the atom;
  correlation_by_spec.csv  Pearson and Spearman correlation between the
                           specifications of the accessibility levels and of
                           the gains of each scenario across origin x segment
                           cells (Santana Palacios and El-Geneidy 2022: levels
                           agree, incidence does not).

With --targeting LABEL=RUN ... (and --zone, --base-run) it also writes
targeting_summary.csv and targeting_cells.csv to outputs/comparisons/
targeting/: person-based against location-based price cuts
(`run.targeting`).
"""

from __future__ import annotations

import argparse
import logging

import pandas as pd

from ikob2 import params as params_mod
from ikob2.run.gap import reachability_gap, summarise
from ikob2.run.interchange import gains, interchange_ratio
from ikob2.utils.paths import DataLayout

logger = logging.getLogger("ikob2.cli.paper_tables")


def _read(lay: DataLayout, name: str) -> pd.DataFrame | None:
    f = lay.run_dir(name) / "accessibility.csv"
    return pd.read_csv(f) if f.exists() else None


def _wmean(df: pd.DataFrame, col: str, by: str) -> pd.Series:
    w = df["population"]
    return (df[col] * w).groupby(df[by]).sum() / w.groupby(df[by]).sum().where(
        lambda s: s > 0)


def correlations(runs: dict, mode: str) -> pd.DataFrame:
    """Correlation between specifications across (origin, segment) cells,
    decile 1 excluded (its atom makes it a special case): the levels of S0 and
    the gains of each scenario. runs: tag -> {scenario: table} with "s0"."""
    rows = []
    series: dict = {"levels": {}}
    for tag, by_scen in runs.items():
        a = by_scen["s0"][by_scen["s0"]["mode"] == mode].set_index(KEY_COLS)
        keep = a["income_class"] != "D1"
        series["levels"][tag] = a["accessibility"].where(keep)
        for scen, run in by_scen.items():
            if scen == "s0" or run is None:
                continue
            b = run[run["mode"] == mode].set_index(KEY_COLS)
            series.setdefault(f"gain_{scen}", {})[tag] = (
                b["accessibility"].reindex(a.index) - a["accessibility"]).where(keep)
    for measure, d in series.items():
        if len(d) < 2:
            continue
        df = pd.DataFrame(d)
        for method in ("pearson", "spearman"):
            c = df.corr(method=method)
            for x in c.index:
                for y in c.columns:
                    rows.append({"measure": measure, "method": method,
                                 "spec_a": x, "spec_b": y, "correlation": c.loc[x, y]})
    return pd.DataFrame(rows)


KEY_COLS = ["buurtcode", "segment"]


TIME_ONLY_TAG = {"m1c": "m1", "m0u": "m0", "m0s": "m0"}


def tables(lay: DataLayout, tags, mode: str, prefix: str = "sp",
           controlled_prefix: str = "spc", timeonly_prefix: str = "spt",
           scenarios=("s1", "s2c", "s2t", "s3c", "s3t", "s4"),
           pairs=(("s1", "s2c"), ("s4", "s2t")),
           gap_scenarios=("s1", "s4"),
           pair_origins: dict | None = None) -> dict[str, pd.DataFrame]:
    """The paper tables for the specification `tags` from the runs
    <prefix>_<tag>_<s0|scenario> (income-matched jobs), <controlled_prefix>_...
    (common jobs) and <timeonly_prefix>_<tag>_s0 (time only, for the gap),
    for `mode`. Returns name -> table: baseline_by_spec, incidence_by_spec,
    interchange_by_spec, gap_by_spec and correlation_by_spec. Missing runs
    are skipped with a warning. `pair_origins` maps a pair "A:B" to the
    origins over which its R is computed (a scenario that only reaches some
    origins, such as a price cut by address, has R = 0 elsewhere by design);
    other pairs use every origin."""
    base, inc, rat, gap, rat_cls = [], [], [], [], []
    runs = {}
    for tag in tags:
        s0 = _read(lay, f"{prefix}_{tag}_s0")
        if s0 is None:
            logger.warning("no run %s_%s_s0", prefix, tag)
            continue
        by_scen = {"s0": s0, **{sc: _read(lay, f"{prefix}_{tag}_{sc}")
                                for sc in scenarios}}
        for sc, run in by_scen.items():
            if run is None:
                logger.warning("no run %s_%s_%s", prefix, tag, sc)
        d = s0[s0["mode"] == mode]
        runs[tag] = by_scen
        base.append(pd.DataFrame({
            "spec": tag,
            "accessibility": _wmean(d, "accessibility", "income_class"),
            "normalised": _wmean(d, "accessibility_normalised", "income_class"),
            "atom": _wmean(d, "atom", "income_class")}).rename_axis(
            "income_class").reset_index())
        for scen in scenarios:
            run = by_scen[scen]
            if run is None:
                continue
            g = (gains(s0, run, mode)).rename("gain").reset_index().merge(
                d[["buurtcode", "segment", "household_type", "income_class",
                   "population"]], on=["buurtcode", "segment"])
            for by in ("income_class", "household_type"):
                tot = (g["gain"] * g["population"]).sum()
                per = _wmean(g, "gain", by)
                share = (g["gain"] * g["population"]).groupby(g[by]).sum() / tot \
                    if tot != 0 else per * float("nan")
                inc.append(pd.DataFrame({
                    "spec": tag, "scenario": scen, "by": by,
                    "gain_per_person": per, "gain_share": share}
                ).rename_axis("group").reset_index())
        for a_scen, b_scen in pairs:
            for pre, label in ((prefix, "full"), (controlled_prefix, "controlled")):
                c0, ca, cb = (_read(lay, f"{pre}_{tag}_{s}")
                              for s in ("s0", a_scen, b_scen))
                if any(x is None for x in (c0, ca, cb)):
                    continue
                only = (pair_origins or {}).get(f"{a_scen}:{b_scen}")
                if only is not None:
                    keep = set(only)
                    c0, ca, cb = (x[x["buurtcode"].astype(str).isin(keep)]
                                  for x in (c0, ca, cb))
                res = interchange_ratio(c0, ca, cb, mode)
                r = res["summary"].iloc[0]
                origins = "all" if only is None else f"{len(set(only))} origins"
                rat.append({"spec": tag, "pair": f"{a_scen}/{b_scen}",
                            "comparison": label, "origins_used": origins,
                            **r.to_dict()})
                rat_cls.append(res["by_class"].assign(
                    spec=tag, pair=f"{a_scen}/{b_scen}", comparison=label,
                    origins_used=origins))
        # gap: time-only runs are per time margin, so M3 shares M2's, the
        # dual cut-offs M0's and M1c M1's
        t_tag = TIME_ONLY_TAG.get(tag, "m2" if tag.startswith("m3") else tag)
        t0 = _read(lay, f"{timeonly_prefix}_{t_tag}_s0")
        if t0 is not None:
            for scen in ("s0", *gap_scenarios):
                run = by_scen.get(scen, _read(lay, f"{prefix}_{tag}_{scen}"))
                if run is None:
                    continue
                gsum = summarise(reachability_gap(run, t0, mode))
                gap.append(gsum.assign(spec=tag, scenario=scen).rename_axis(
                    "income_class").reset_index())
    out = {}
    for name, rows in (("baseline_by_spec", base), ("incidence_by_spec", inc),
                       ("gap_by_spec", gap)):
        out[name] = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    out["interchange_by_spec"] = pd.DataFrame(rat)
    out["interchange_by_class"] = (pd.concat(rat_cls, ignore_index=True)
                                   if rat_cls else pd.DataFrame())
    out["correlation_by_spec"] = correlations(runs, mode)
    return out


def targeting_tables(lay: DataLayout, base_run: str, runs: dict, zone_file,
                     mode: str, target_classes, costs=None) -> dict[str, pd.DataFrame]:
    """run.targeting for the runs `runs` (label -> run name) against
    `base_run`, with the zone of `zone_file` (column buurtcode). A run with
    extra hubs (`shared_bike.egress_suffix`) gets their yearly public cost
    (`run.costs.hub_annual_cost`, the hubs of its hub file) as
    `hub_cost_eur_year`."""
    from ikob2.run.costs import hub_annual_cost
    import json

    from ikob2.run.targeting import targeting

    base = _read(lay, base_run)
    if base is None:
        raise SystemExit(f"No run {base_run}.")
    loaded = {}
    for label, name in runs.items():
        t = _read(lay, name)
        if t is None:
            logger.warning("no run %s", name)
            continue
        meta = json.loads((lay.run_dir(name) / "run.json").read_text())
        suffix = (meta.get("parameters", {}).get("shared_bike", {})
                  .get("egress_suffix") or {})
        n_hubs = 0
        for s in suffix.values():
            f = lay.s2_hubs(s.lstrip("_"))
            if f.exists():
                n_hubs += len(pd.read_csv(f))
            else:
                logger.warning("no hub file %s for run %s", f, name)
        if n_hubs and costs is not None:
            meta["hub_cost_eur_year"] = hub_annual_cost(costs, n_hubs)["public"]
            meta["hubs"] = n_hubs
        loaded[label] = (t, meta)
    zone = pd.read_csv(zone_file, dtype={"buurtcode": str})["buurtcode"]
    return targeting(base, loaded, zone, mode, target_classes)


def main(argv=None) -> None:
    """Command line entry point (`python -m ikob2.cli.paper_tables`); writes
    the tables to outputs/comparisons/specs/."""
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)
    params_mod.add_arguments(p)
    p.add_argument("--data-root", default=None)
    p.add_argument("--tags", nargs="+", default=None, help="default paper.tags")
    p.add_argument("--mode", default=None, help="default analysis.mode")
    p.add_argument("--scenarios", nargs="+", default=None,
                   help="default paper.scenarios")
    p.add_argument("--pairs", nargs="+", default=None, metavar="A:B",
                   help="R = gain(A) / gain(B); default paper.pairs")
    p.add_argument("--gap-scenarios", nargs="+", default=None,
                   help="price scenarios after which the gap is reported; "
                        "default paper.gap_scenarios")
    p.add_argument("--targeting", nargs="*", default=None, metavar="LABEL=RUN",
                   help="scenario runs of the targeting tables; default "
                        "paper.targeting; none: no targeting tables")
    p.add_argument("--base-run", default=None,
                   help="baseline run of the targeting tables; default "
                        "paper.base_run")
    p.add_argument("--zone", default=None,
                   help="price zone of the targeting tables; default paper.zone")
    p.add_argument("--target-classes", nargs="+", default=None,
                   help="default paper.target_classes")
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args(argv)
    logging.basicConfig(level=args.log_level,
                        format="%(levelname)s %(name)s: %(message)s")
    prm = params_mod.from_args(args, {
        "tags": "paper.tags", "mode": "analysis.mode",
        "scenarios": "paper.scenarios", "pairs": "paper.pairs",
        "gap_scenarios": "paper.gap_scenarios", "targeting": "paper.targeting",
        "base_run": "paper.base_run", "zone": "paper.zone",
        "target_classes": "paper.target_classes"})
    paper = prm.paper
    args.tags, args.mode = paper.tags, prm.analysis.mode
    args.scenarios, args.pairs = paper.scenarios, paper.pairs
    args.gap_scenarios, args.targeting = paper.gap_scenarios, paper.targeting
    args.base_run, args.zone = paper.base_run, paper.zone
    args.target_classes = paper.target_classes
    lay = DataLayout(params_mod.data_root(args.data_root, prm))
    from ikob2.utils.paths import resolve_input
    pair_origins = {
        pair: pd.read_csv(resolve_input(f, lay.inputs / "tariffs"),
                          dtype={"buurtcode": str})["buurtcode"].tolist()
        for pair, f in prm.paper.pair_zones.to_dict().items()}
    res = tables(lay, args.tags, args.mode, scenarios=tuple(args.scenarios),
                 pairs=tuple(tuple(x.split(":", 1)) for x in args.pairs),
                 gap_scenarios=tuple(args.gap_scenarios),
                 pair_origins=pair_origins)
    out = lay.comparison_dir() / "specs"
    out.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 200)
    for name, df in res.items():
        df.to_csv(out / f"{name}.csv", index=False)
        print(f"{name}: {len(df)} rows")
    if len(res["interchange_by_spec"]):
        cols = ["spec", "pair", "comparison", "pooled_median_R", "pooled_cv",
                "pooled_iqr_ratio", "pooled_spread", "pairs_undefined"]
        print(res["interchange_by_spec"][cols].round(3).to_string(index=False))
    print(f"Tables written to {out}")
    if args.targeting:
        from ikob2.utils.paths import resolve_input
        zone = resolve_input(args.zone, lay.inputs / "tariffs")
        tt = targeting_tables(lay, args.base_run,
                              dict(x.split("=", 1) for x in args.targeting),
                              zone, args.mode, args.target_classes, prm.costs)
        tdir = lay.comparison_dir() / "targeting"
        tdir.mkdir(parents=True, exist_ok=True)
        for name, df in tt.items():
            df.to_csv(tdir / f"targeting_{name}.csv", index=False)
        print(tt["summary"].round(3).to_string(index=False))
        print(f"Targeting tables written to {tdir}")


if __name__ == "__main__":
    main()
