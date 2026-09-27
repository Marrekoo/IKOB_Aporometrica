"""Tables of the paper from the runs of a specification grid.

    python -m ikob2.cli.paper_tables --data-root <root> \
        --tags m1 m1p m2 m3t1.25 m3t1.5 m3t2 m3t4 m3tinf

Reads outputs/runs/<prefix>_<tag>_<scenario> for the scenarios s0 (baseline),
s1 (fare cut) and s2 (hubs), with prefix `sp` (income-matched jobs, the full
application) and `spc` (common jobs, the controlled comparison), and
`spt_<tag>_<s0|s2>` (time only, for the gap; M3 uses the M2 time-only run) and
writes to outputs/comparisons/specs/:

  baseline_by_spec.csv     4.2  accessibility, raw and normalised, atom, per
                                specification and income class;
  incidence_by_spec.csv    4.3  gain of S1 and S2 by income class and household
                                type, per specification;
  interchange_by_spec.csv  4.1  R = gain(S1) / gain(S2): pooled dispersion (CV,
                                interquantile ratio, spread), median, undefined
                                pairs, full and controlled;
  gap_by_spec.csv          4.4  the reachability gap before and after S1, and
                                the atom;
  correlation_by_spec.csv       Pearson and Spearman correlation between the
                                specifications of the accessibility levels and
                                of the gains of S1 and S2 across origin x
                                segment cells (Santana Palacios and El-Geneidy
                                2022: levels agree, incidence does not).
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
    the gains of S1 and S2. runs: tag -> (s0, s1, s2) tables."""
    rows = []
    idx = None
    series = {"levels": {}, "gain_s1": {}, "gain_s2": {}}
    for tag, (s0, s1, s2) in runs.items():
        a = s0[s0["mode"] == mode].set_index(KEY_COLS)
        idx = a.index if idx is None else idx
        keep = a["income_class"] != "D1"
        series["levels"][tag] = a["accessibility"].where(keep)
        for name, run in (("gain_s1", s1), ("gain_s2", s2)):
            if run is not None:
                b = run[run["mode"] == mode].set_index(KEY_COLS)
                series[name][tag] = (b["accessibility"] - a["accessibility"]).where(keep)
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


def tables(lay: DataLayout, tags, mode: str, prefix: str = "sp",
           controlled_prefix: str = "spc", timeonly_prefix: str = "spt"
           ) -> dict[str, pd.DataFrame]:
    base, inc, rat, gap = [], [], [], []
    runs = {}
    for tag in tags:
        s0, s1, s2 = (_read(lay, f"{prefix}_{tag}_{s}") for s in ("s0", "s1", "s2"))
        if s0 is None:
            logger.warning("no run %s_%s_s0", prefix, tag)
            continue
        d = s0[s0["mode"] == mode]
        runs[tag] = (s0, s1, s2)
        base.append(pd.DataFrame({
            "spec": tag,
            "accessibility": _wmean(d, "accessibility", "income_class"),
            "normalised": _wmean(d, "accessibility_normalised", "income_class"),
            "atom": _wmean(d, "atom", "income_class")}).rename_axis(
            "income_class").reset_index())
        if s1 is not None and s2 is not None:
            for scen, run in (("s1", s1), ("s2", s2)):
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
            for pre, label in ((prefix, "full"), (controlled_prefix, "controlled")):
                c0, c1, c2 = (_read(lay, f"{pre}_{tag}_{s}") for s in ("s0", "s1", "s2"))
                if any(x is None for x in (c0, c1, c2)):
                    continue
                s = interchange_ratio(c0, c1, c2, mode)["summary"].iloc[0]
                rat.append({"spec": tag, "comparison": label, **s.to_dict()})
        # gap: time-only runs are per time margin, so M3 shares M2's
        # M3 has the time margin of M2, M1c that of M1
        t_tag = {"m1c": "m1"}.get(tag, "m2" if tag.startswith("m3") else tag)
        t0 = _read(lay, f"{timeonly_prefix}_{t_tag}_s0")
        if t0 is not None:
            for scen, run in (("s0", s0), ("s1", s1)):
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


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)
    params_mod.add_arguments(p)
    p.add_argument("--data-root", default=None)
    p.add_argument("--tags", nargs="+", required=True)
    p.add_argument("--mode", default="pt_v2")
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args(argv)
    logging.basicConfig(level=args.log_level,
                        format="%(levelname)s %(name)s: %(message)s")
    lay = DataLayout(params_mod.data_root(args.data_root,
                                          params_mod.from_args(args)))
    res = tables(lay, args.tags, args.mode)
    out = lay.comparison_dir() / "specs"
    out.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 200)
    for name, df in res.items():
        df.to_csv(out / f"{name}.csv", index=False)
        print(f"{name}: {len(df)} rows")
    if len(res["interchange_by_spec"]):
        cols = ["spec", "comparison", "pooled_median_R", "pooled_cv",
                "pooled_iqr_ratio", "pooled_spread", "pairs_undefined"]
        print(res["interchange_by_spec"][cols].round(3).to_string(index=False))
    print(f"Tables written to {out}")


if __name__ == "__main__":
    main()
