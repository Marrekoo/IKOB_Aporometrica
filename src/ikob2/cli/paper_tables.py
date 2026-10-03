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
  interchange_by_spec.csv  R = gain(A) / gain(B) for each pair of --pairs
                           (default s1:s2c, the citywide price cut over the
                           citywide hubs, and s4:s2t, both in the target
                           buurten): pooled dispersion (CV, interquantile
                           ratio, spread), median, undefined pairs, full and
                           controlled;
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
           gap_scenarios=("s1", "s4")) -> dict[str, pd.DataFrame]:
    """The paper tables for the specification `tags` from the runs
    <prefix>_<tag>_<s0|scenario> (income-matched jobs), <controlled_prefix>_...
    (common jobs) and <timeonly_prefix>_<tag>_s0 (time only, for the gap),
    for `mode`. Returns name -> table: baseline_by_spec, incidence_by_spec,
    interchange_by_spec, gap_by_spec and correlation_by_spec. Missing runs
    are skipped with a warning."""
    base, inc, rat, gap = [], [], [], []
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
                r = interchange_ratio(c0, ca, cb, mode)["summary"].iloc[0]
                rat.append({"spec": tag, "pair": f"{a_scen}/{b_scen}",
                            "comparison": label, **r.to_dict()})
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
    out["correlation_by_spec"] = correlations(runs, mode)
    return out


def targeting_tables(lay: DataLayout, base_run: str, runs: dict, zone_file,
                     mode: str, target_classes) -> dict[str, pd.DataFrame]:
    """run.targeting for the runs `runs` (label -> run name) against
    `base_run`, with the zone of `zone_file` (column buurtcode)."""
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
    p.add_argument("--tags", nargs="+", required=True)
    p.add_argument("--mode", default="pt_v2")
    p.add_argument("--scenarios", nargs="+",
                   default=["s1", "s2c", "s2t", "s3c", "s3t", "s4"])
    p.add_argument("--pairs", nargs="+", default=["s1:s2c", "s4:s2t"],
                   metavar="A:B", help="R = gain(A) / gain(B)")
    p.add_argument("--gap-scenarios", nargs="+", default=["s1", "s4"],
                   help="price scenarios after which the gap is reported")
    p.add_argument("--targeting", nargs="*", default=[], metavar="LABEL=RUN",
                   help="scenario runs for the targeting tables, e.g. "
                        "S1=scen_s1 S1a=scen_s1a S4=scen_s4 S4a=scen_s4a")
    p.add_argument("--base-run", default="scen_s0",
                   help="baseline run of the targeting tables")
    p.add_argument("--zone", default=None,
                   help="price zone of the targeting tables (default "
                        "paths.lime_price_zones, else the S4 zone file)")
    p.add_argument("--target-classes", nargs="+", default=["D2", "D3", "D4"])
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args(argv)
    logging.basicConfig(level=args.log_level,
                        format="%(levelname)s %(name)s: %(message)s")
    lay = DataLayout(params_mod.data_root(args.data_root,
                                          params_mod.from_args(args)))
    res = tables(lay, args.tags, args.mode, scenarios=tuple(args.scenarios),
                 pairs=tuple(tuple(x.split(":", 1)) for x in args.pairs),
                 gap_scenarios=tuple(args.gap_scenarios))
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
        prm = params_mod.from_args(args)
        zone = resolve_input(args.zone or prm.paths.lime_price_zones
                             or "lime_price_zones_overvecht_kanaleneiland.csv",
                             lay.inputs / "tariffs")
        tt = targeting_tables(lay, args.base_run,
                              dict(x.split("=", 1) for x in args.targeting),
                              zone, args.mode, args.target_classes)
        tdir = lay.comparison_dir() / "targeting"
        tdir.mkdir(parents=True, exist_ok=True)
        for name, df in tt.items():
            df.to_csv(tdir / f"targeting_{name}.csv", index=False)
        print(tt["summary"].round(3).to_string(index=False))
        print(f"Targeting tables written to {tdir}")


if __name__ == "__main__":
    main()
