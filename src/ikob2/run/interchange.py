"""The interchangeability ratio (paper, Section 4.1).

    R[i, s] = da(A)[i, s] / da(B)[i, s]

the accessibility gain of intervention A (the fare cut, S1) over that of
intervention B (more hubs, S2), for origin i and segment s, both measured
against the same baseline. Under a generalised cost with common inputs R does
not depend on s; under the gate it does, and the between-segment dispersion of
R is the paper's central diagnostic.

    * R is undefined where the gain of B is zero (|da(B)| <= `tol`); those
      pairs are returned with `defined = False` and counted per origin.
    * Dispersion is computed WITHIN an origin (across its segments) and pooled
      afterwards (population-weighted over the origins), which removes the
      between-origin composition; the statistics are an interquantile ratio
      (Q_hi / Q_lo of R across segments) and the coefficient of variation.
    * A segment's R is the ratio of its per-person gains; populations enter
      only the pooling.

Inputs are accessibility tables of runs (`AccessibilityResult.table`, or
accessibility.csv).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

KEY = ["buurtcode", "segment"]


def gains(base: pd.DataFrame, run: pd.DataFrame, mode: str,
          value: str = "accessibility") -> pd.Series:
    """Per (origin, segment) change of `value` for `mode` against a baseline."""
    a = base[base["mode"] == mode].set_index(KEY)[value]
    b = run[run["mode"] == mode].set_index(KEY)[value]
    if not a.index.equals(b.index):
        b = b.reindex(a.index)
    return (b - a).rename("gain")


def interchange_ratio(base: pd.DataFrame, run_a: pd.DataFrame,
                      run_b: pd.DataFrame, mode: str = "pt_v2", *,
                      value: str = "accessibility", tol: float = 1e-9,
                      quantiles: tuple[float, float] = (0.9, 0.1),
                      min_segments: int = 3) -> dict[str, pd.DataFrame]:
    """R per origin and segment, the dispersion per origin and pooled.

    Returns
      pairs   : one row per (origin, segment): da_a, da_b, R, defined and the
                segment columns (household_type, income_class, population);
      origins : per origin: segments with R defined, median R, the
                interquantile ratio and the CV of R across its segments, and
                whether its dispersion could be computed (`min_segments`);
      summary : population-weighted pooled statistics and the counts of
                undefined pairs and origins (a one-row frame).
    """
    da = gains(base, run_a, mode, value)
    db = gains(base, run_b, mode, value)
    meta = base[base["mode"] == mode].set_index(KEY)[
        ["household_type", "income_class", "population"]]
    pairs = meta.join(da.rename("da_a")).join(db.rename("da_b"))
    pairs["defined"] = pairs["da_b"].abs() > tol
    pairs["R"] = np.where(pairs["defined"], pairs["da_a"] / pairs["da_b"].where(
        pairs["defined"]), np.nan)
    pairs = pairs.reset_index()

    q_hi, q_lo = quantiles

    def stats(g: pd.DataFrame) -> pd.Series:
        r = g.loc[g["defined"], "R"]
        ok = len(r) >= min_segments
        lo, hi = (r.quantile(q_lo), r.quantile(q_hi)) if ok else (np.nan,) * 2
        return pd.Series({
            "segments": len(g), "defined": int(len(r)),
            "median_R": r.median() if len(r) else np.nan,
            "mean_R": r.mean() if len(r) else np.nan,
            "q_lo": lo, "q_hi": hi,
            # ratio of the upper to the lower quantile of R; undefined when
            # the lower quantile is not positive (a segment with no gain)
            "iqr_ratio": hi / lo if ok and lo > 0 else np.nan,
            "spread": hi - lo if ok else np.nan,
            "cv": r.std(ddof=0) / abs(r.mean()) if ok and r.mean() != 0
            else np.nan,
            "population": g["population"].sum()})

    origins = pairs.groupby("buurtcode").apply(stats, include_groups=False)
    w = origins["population"]

    def pooled(col: str) -> float:
        ok = origins[col].notna() & (w > 0)
        return float((origins.loc[ok, col] * w[ok]).sum() / w[ok].sum()) \
            if ok.any() else np.nan

    summary = pd.DataFrame([{
        "mode": mode, "quantiles": f"{q_hi}/{q_lo}",
        "pairs": len(pairs), "pairs_undefined": int((~pairs["defined"]).sum()),
        "origins": len(origins),
        "origins_undefined": int((origins["defined"] == 0).sum()),
        "origins_with_dispersion": int(origins["cv"].notna().sum()),
        "pooled_cv": pooled("cv"), "pooled_iqr_ratio": pooled("iqr_ratio"),
        "pooled_spread": pooled("spread"), "pooled_median_R": pooled("median_R"),
    }])
    return {"pairs": pairs, "origins": origins.reset_index(),
            "summary": summary}
