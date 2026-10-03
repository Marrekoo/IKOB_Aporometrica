"""
Compare two accessibility runs (the impedance-shape test).

The question of the El-Geneidy-style comparison: do two impedance
specifications, calibrated to the same cut-off, give the same picture?
Two things can differ: the RANKING (which origins and segments come out
better off) and the LEVELS and their distribution across groups.
`compare_runs` reports both from the long tables of two runs.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ikob2.params import DEFAULTS
from scipy import stats


def _wmean(values: pd.Series, weights: pd.Series) -> float:
    w = weights.to_numpy(dtype=float)
    return float((values.to_numpy(dtype=float) * w).sum() / w.sum()) \
        if w.sum() > 0 else np.nan


def compare_runs(a: pd.DataFrame, b: pd.DataFrame, value: str = "accessibility",
                 top_share: float = DEFAULTS.analysis.compare_top_share
                 ) -> dict[str, pd.DataFrame]:
    """Compare two run tables cell by cell.

    Returns
      'cells'    per mode: number of cells, Pearson and Spearman
                 correlation over all origin x segment cells, mean
                 within-segment Spearman across origins, mean ratio b/a;
      'origins'  per mode: Spearman and top-decile overlap of the
                 population-weighted mean accessibility per origin;
      'income'   per mode and income class: population-weighted mean of
                 both runs and their ratio (b/a);
      'segments' per mode and segment: ratio of population-weighted means.
    """
    keys = ["mode", "buurtcode", "segment"]
    m = a[keys + ["household_type", "income_class", "population", value]] \
        .merge(b[keys + [value]], on=keys, suffixes=("_a", "_b"))
    if m.empty:
        raise ValueError("The two runs share no origin x segment x mode "
                         "cells.")
    va, vb = f"{value}_a", f"{value}_b"

    cells, origins, income, segments = [], [], [], []
    for mode, d in m.groupby("mode"):
        x, y = d[va].to_numpy(float), d[vb].to_numpy(float)
        ok = np.isfinite(x) & np.isfinite(y)
        within = []
        for _, s in d.groupby("segment"):
            if s[va].nunique() > 1 and s[vb].nunique() > 1:
                within.append(stats.spearmanr(s[va], s[vb])[0])
        cells.append({
            "mode": mode, "cells": int(ok.sum()),
            "pearson": float(np.corrcoef(x[ok], y[ok])[0, 1]),
            "spearman": float(stats.spearmanr(x[ok], y[ok])[0]),
            "within_segment_spearman": float(np.mean(within))
            if within else np.nan,
            "mean_ratio_b_over_a": _wmean(d[vb], d["population"])
            / _wmean(d[va], d["population"])})

        o = d.groupby("buurtcode").apply(
            lambda g: pd.Series({va: _wmean(g[va], g["population"]),
                                 vb: _wmean(g[vb], g["population"])}),
            include_groups=False).dropna()
        k = max(1, int(round(len(o) * top_share)))
        top_a = set(o[va].nlargest(k).index)
        top_b = set(o[vb].nlargest(k).index)
        origins.append({
            "mode": mode, "origins": len(o),
            "spearman": float(stats.spearmanr(o[va], o[vb])[0]),
            f"top_{int(top_share * 100)}pct_overlap": len(top_a & top_b) / k})

        for ic, g in d.groupby("income_class"):
            ma, mb = _wmean(g[va], g["population"]), _wmean(g[vb], g["population"])
            income.append({"mode": mode, "income_class": ic,
                           "a": ma, "b": mb,
                           "ratio_b_over_a": mb / ma if ma else np.nan})
        for sg, g in d.groupby("segment"):
            ma, mb = _wmean(g[va], g["population"]), _wmean(g[vb], g["population"])
            segments.append({"mode": mode, "segment": sg, "a": ma, "b": mb,
                             "ratio_b_over_a": mb / ma if ma else np.nan})
    return {"cells": pd.DataFrame(cells), "origins": pd.DataFrame(origins),
            "income": pd.DataFrame(income), "segments": pd.DataFrame(segments)}
