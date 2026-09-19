"""Buurt-level household-type and income marginals, and the municipal
seed tables they are derived from."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from ikob2.segments.config import LOW_INCOME_CLASSES, SegmentConfig
from ikob2.segments.kwb import clean_pct

logger = logging.getLogger(__name__)

_GM = r"^GM[0-9]{4}$"


# ── Municipal seed (86161NED) ────────────────────────────────────────

def build_gemeente_seed(raw: pd.DataFrame, cfg: SegmentConfig,
                        gemeente_codes=None) -> pd.DataFrame:
    """Long table (gemeentecode, household_type, income_class, n_hh)
    over the full grid of municipalities x types x classes.

    Decile columns are percentages of the household total (which is
    published x 1000). 'onbekend' is the non-negative remainder to the
    total. Empty cells get 1e-6 so the log-linear model stays defined.
    """
    df = raw.copy()
    df["gemeentecode"] = df["RegioS"].astype(str).str.strip()
    df["KenmerkenVanHuishoudens"] = (
        df["KenmerkenVanHuishoudens"].astype(str).str.strip())
    key_to_type = {v: k for k, v in cfg.household_key_map.items()}
    df = df[df["KenmerkenVanHuishoudens"].isin(key_to_type)]
    if gemeente_codes is not None:
        df = df[df["gemeentecode"].isin(gemeente_codes)]
    else:
        df = df[df["gemeentecode"].str.match(_GM)]
    if df.empty:
        raise ValueError("No income-seed rows left after filtering; check "
                         "periods, keys and gemeente codes.")
    df["household_type"] = df["KenmerkenVanHuishoudens"].map(key_to_type)
    df["total_hh"] = pd.to_numeric(df[cfg.income_total_col],
                                   errors="coerce") * 1000

    parts = []
    for cls, col in cfg.income_decile_cols.items():
        part = df[["gemeentecode", "household_type"]].copy()
        part["income_class"] = cls
        part["n_hh"] = df["total_hh"] * clean_pct(df[col]) / 100
        parts.append(part[part["n_hh"].notna()])
    long = pd.concat(parts, ignore_index=True)
    long = long.groupby(["gemeentecode", "household_type", "income_class"],
                        as_index=False)["n_hh"].sum()

    if "onbekend" in cfg.income_classes:
        known = long.groupby(["gemeentecode", "household_type"],
                             as_index=False)["n_hh"].sum().rename(
                                 columns={"n_hh": "known_hh"})
        tot = df.groupby(["gemeentecode", "household_type"],
                         as_index=False)["total_hh"].sum(min_count=1)
        unk = tot.merge(known, how="left",
                        on=["gemeentecode", "household_type"])
        unk["known_hh"] = unk["known_hh"].fillna(0.0)
        unk["n_hh"] = np.maximum(unk["total_hh"] - unk["known_hh"], 0.0)
        unk["income_class"] = "onbekend"
        long = pd.concat(
            [long, unk[["gemeentecode", "household_type",
                        "income_class", "n_hh"]]], ignore_index=True)

    grid = pd.MultiIndex.from_product(
        [sorted(long["gemeentecode"].unique()),
         list(cfg.household_types), list(cfg.income_classes)],
        names=["gemeentecode", "household_type", "income_class"]
    ).to_frame(index=False)
    out = grid.merge(long, how="left",
                     on=["gemeentecode", "household_type", "income_class"])
    out["n_hh"] = np.where(out["n_hh"].isna() | (out["n_hh"] <= 0),
                           1e-6, out["n_hh"])
    return out


def single_parent_shares(raw: pd.DataFrame, cfg: SegmentConfig,
                         gemeente_codes=None) -> pd.Series:
    """gemeentecode -> single-parent share of households with children
    (71487ned); municipalities without children fall back to the
    configured share."""
    df = raw.copy()
    df["gemeentecode"] = df["RegioS"].astype(str).str.strip()
    df = df[df["LeeftijdKindEren"].astype(str).str.strip()
            == cfg.children_age_total]
    df = df[df["gemeentecode"].str.match(_GM)]
    if gemeente_codes is not None:
        df = df[df["gemeentecode"].isin(gemeente_codes)]
    g = df.groupby("gemeentecode").agg(
        with_children=("TotaalHuishoudensMetKinderen_1",
                       lambda s: pd.to_numeric(s, errors="coerce").sum()),
        single_parent=("TotaalEenouderhuishoudensMetKinderen_13",
                       lambda s: pd.to_numeric(s, errors="coerce").sum()),
    )
    share = np.where(g["with_children"] > 0,
                     g["single_parent"] / g["with_children"],
                     cfg.fallback_single_parent_share_of_with_children)
    return pd.Series(np.clip(share, 0.0, 1.0), index=g.index,
                     name="single_parent_share_of_with_children")


# ── Household-type marginals ─────────────────────────────────────────

def household_marginals(kwb: pd.DataFrame, sp_shares: pd.Series,
                        cfg: SegmentConfig) -> pd.DataFrame:
    """Households per type per buurt, summing to KWB 'huishoudens'.

    Missing/zero composition falls back to a fixed national mix; the
    four counts are then rescaled to the buurt's household total.
    """
    fb = pd.Series(cfg.fallback_hh_composition, dtype=float)
    fb = fb / fb.sum()

    share = kwb["gemeentecode"].map(sp_shares).fillna(
        cfg.fallback_single_parent_share_of_with_children)
    hh = kwb["huishoudens"]

    single = kwb["hh_eenpersoons"]
    couple = (np.maximum(kwb["hh_zonder_kind"] - kwb["hh_eenpersoons"], 0.0)
              if cfg.kwb_no_child_includes_single else kwb["hh_zonder_kind"])
    raw = pd.DataFrame({
        "single": np.maximum(single, 0.0),
        "couple": np.maximum(couple, 0.0),
        "single_parent": np.maximum(kwb["hh_met_kind"] * share, 0.0),
        "couple_children": np.maximum(kwb["hh_met_kind"] * (1 - share), 0.0),
    })
    # skipna=False: any missing component makes the raw total missing,
    # which (like the R script) triggers the fallback mix.
    total_raw = raw.sum(axis=1, skipna=False)
    use_fallback = (hh > 0) & (total_raw.isna() | (total_raw <= 0))

    counts = raw.copy()
    for t in raw.columns:
        counts[t] = np.where(use_fallback, hh * fb[t], raw[t])
    total = counts.sum(axis=1, skipna=False)
    scale = np.where(total > 0, hh / total, np.nan)
    for t in counts.columns:
        counts[t] = counts[t] * scale

    out = pd.concat([kwb[["buurtcode", "gemeentecode"]],
                     hh.rename("huishoudens"), counts], axis=1)
    out["check_total"] = counts.sum(axis=1, skipna=False)
    out["use_fallback"] = use_fallback
    n_fb = int(use_fallback.sum())
    if n_fb:
        logger.warning("%d buurten used the fallback household "
                       "composition (missing/zero marginals).", n_fb)
    return out[["buurtcode", "gemeentecode", "huishoudens",
                *cfg.household_types, "check_total", "use_fallback"]]


# ── Income-class marginals ───────────────────────────────────────────

def decile_marginals(kwb: pd.DataFrame, seed: pd.DataFrame,
                     cfg: SegmentConfig) -> pd.DataFrame:
    """Income-class shares per buurt (columns = income classes).

    The municipal income shape (all household types pooled) is used
    within the low block (D1-D4) and the rest separately, each rescaled
    so the D1-D4 total equals the buurt's own KWB low-income share.
    Also returns 'low_target_used' and 'calibration_source'.
    """
    classes = list(cfg.income_classes)
    low_mask = np.array([c in LOW_INCOME_CLASSES for c in classes])

    pooled = seed.groupby(["gemeentecode", "income_class"])["n_hh"].sum()
    shape = pooled.unstack("income_class").reindex(columns=classes)
    shape = shape.div(shape.sum(axis=1), axis=0)

    p = kwb["p_laag40"]
    valid = p.between(1, 99)
    fallback = (kwb.loc[valid].groupby("gemeentecode")["p_laag40"].median())

    S = shape.reindex(kwb["gemeentecode"]).to_numpy(dtype=float)
    S = np.where(np.isnan(S), 0.0, S)
    no_shape = S.sum(axis=1) <= 0
    if no_shape.any():
        logger.warning("%d buurten have no municipal income shape; using "
                       "a uniform shape.", int(no_shape.sum()))
        S[no_shape] = 1.0 / len(classes)
    S = S / S.sum(axis=1, keepdims=True)

    gm_low = np.clip(S[:, low_mask].sum(axis=1), 1e-6, 1 - 1e-6)

    fb = kwb["gemeentecode"].map(fallback)
    fb_valid = fb.between(1, 99)
    if cfg.local_income_calibration == "none":
        target = gm_low.copy()
        source = np.array(["none"] * len(kwb), dtype=object)
    else:
        target = np.where(valid, p / 100,
                          np.where(fb_valid, fb / 100, gm_low))
        source = np.where(valid, "p_laag40",
                          np.where(fb_valid, "gemeente_median_p_laag40",
                                   "gemeente_shape")).astype(object)
    target = np.clip(target, 1e-6, 1 - 1e-6)

    scale_low = target / gm_low
    scale_high = (1 - target) / (1 - gm_low)
    S = np.where(low_mask[None, :], S * scale_low[:, None],
                 S * scale_high[:, None])
    S = S / S.sum(axis=1, keepdims=True)

    out = pd.DataFrame(S, columns=classes)
    out.insert(0, "buurtcode", kwb["buurtcode"].to_numpy())
    out["low_target_used"] = target
    out["calibration_source"] = source
    mean_low = float(S[:, low_mask].sum(axis=1).mean())
    logger.info("Mean imputed D1-D4 share: %.1f%%; calibration sources: %s",
                100 * mean_low,
                dict(pd.Series(source).value_counts()))
    if mean_low < 0.05:
        logger.warning("Mean D1-D4 share is below 5%%: check the KWB "
                       "low-income column mapping.")
    return out
