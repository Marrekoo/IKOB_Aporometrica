"""
Baskets and residuals (X_M calc.R, sections 3 and 6).

  * `basket_aggregates`: per Nibud household, the total of the minimum basket
    and its parts: shelter (rent), mobility (`mob_min`) and the rest
    (`m_bas`, the minimum non-mobility basket).
  * `anchors`: the anchor points of each household type: net income `y`,
    rent, the minimum basket `m_bas` and an example basket `m_ex`. At
    Warnaar's anchors `m_ex` is backed out of the published residual b_norm,
    read as the residual at gamma = `gamma_anchor` (0.5: the midpoint of the
    two residuals, as in X_M calc.R; 0: the example-basket residual itself):

        b_norm = y - rent - kappa ((1 - a) m_ex + a m_bas),  a = gamma_anchor

    At the social-assistance point and outside Warnaar's anchors it is
    extended along the household type's own slope in standardised income;
    single parents, who have no published b_norm, borrow the relative slack
    (m_ex / m_bas - 1) of the donor household types.
  * `quantile_envelope`: per household type, income decile and rent scenario,
    the residual after the example basket (`b_ex`) and after the minimum
    basket (`b_bas`); gamma interpolates between them:

        b(gamma) = (1 - gamma) b_ex + gamma b_bas
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ikob2.envelope.sources import Sources

HH_TYPES = ("single", "couple", "single_parent", "couple_children")
ADULTS = {"single": 1, "couple": 2, "single_parent": 1, "couple_children": 2}


def basket_aggregates(src: Sources) -> pd.DataFrame:
    """Per Nibud household: total, shelter, mob_min, m_bas (EUR/month).
    Raises when a column does not reproduce its published total or the parts
    do not add up."""
    b = src["nibud_basket"].merge(src["nibud_posts"], on="post", how="left")
    if b["role"].isna().any():
        raise ValueError("A basket post has no role in nibud_posts.")
    eur = b["eur"]
    agg = b.assign(
        total=eur,
        shelter=eur.where(b["role"] == "shelter"),
        mob_min=eur.where(b["role"] == "mobility"),
        m_bas=eur.where(~b["role"].isin(["shelter", "mobility"]))
    ).groupby("hh_key")[["total", "shelter", "mob_min", "m_bas"]].sum(min_count=1)
    hh = src["nibud_households"].set_index("hh_key")
    err = (agg["total"] - hh["published_total"].reindex(agg.index)).abs()
    if (err >= 0.5).any():
        raise ValueError(f"Basket does not reproduce the published totals: "
                         f"{err[err >= 0.5].to_dict()}")
    ident = agg["total"] - agg["shelter"] - agg["mob_min"] - agg["m_bas"]
    if (ident.abs() >= 0.5).any():
        raise ValueError("total != shelter + mob_min + m_bas.")
    return agg.join(hh[["hh_type", "aow", "n_child"]])


def type_baskets(src: Sources) -> pd.DataFrame:
    """The basket of each household type of the model (index hh_type)."""
    agg = basket_aggregates(src)
    keys = src["nibud_type_keys"].set_index("hh_type")["hh_key"]
    return agg.loc[keys.values].set_index(keys.index)[
        ["total", "shelter", "mob_min", "m_bas"]]


def _extend(y_std: np.ndarray, m_ex: np.ndarray, m_bas: np.ndarray) -> np.ndarray:
    """Fill missing example baskets along the straight line through the
    lowest and highest ones (in standardised income), at least m_bas."""
    k = np.flatnonzero(np.isfinite(m_ex))
    if len(k) < 2:
        return np.full(len(m_ex), np.nan)
    i_hi = k[np.argmax(m_ex[k])]
    i_lo = k[np.argmin(m_ex[k])]
    dx = y_std[i_hi] - y_std[i_lo]
    slope = (m_ex[i_hi] - m_ex[i_lo]) / dx if dx != 0 else 0.0
    j = k[np.argmin(y_std[k])]
    line = np.maximum(m_ex[j] + slope * (y_std - y_std[j]), m_bas)
    return np.where(np.isfinite(m_ex), m_ex, line)


def anchors(src: Sources, prm) -> pd.DataFrame:
    """Anchor points per household type with y_disp, y_std, rent, m_bas,
    m_ex (and its donor bracket m_ex_lo, m_ex_hi), kappa_row, b_bas, b_ex."""
    e = prm.envelope
    eqv = src.eqv()
    kappa = src.kappa
    tb = type_baskets(src)

    war = src["warnaar_anchors"].assign(price_base="warnaar")
    bij = (tb.reset_index()[["hh_type", "shelter"]]
           .sort_values("hh_type", key=lambda s: s.map(
               {t: k for t, k in src["nibud_type_keys"][["hh_type", "hh_key"]]
                .itertuples(index=False)}))
           .rename(columns={"shelter": "rent"})
           .merge(src["bijstand_published"][["hh_type", "y_disp"]], on="hh_type")
           .assign(level="bijstand", b_norm=np.nan, price_base="vdb2023"))
    cc = war[war["hh_type"] == "couple_children"]
    ratio_y = eqv["single_parent"] / eqv["couple_children"]
    ratio_r = (src.rent(e.rent_lineage, "single_parent")
               / src.rent(e.rent_lineage, "couple_children"))
    sp = cc.assign(hh_type="single_parent", y_disp=cc["y_disp"] * ratio_y,
                   rent=cc["rent"] * ratio_r, b_norm=np.nan)
    a = pd.concat([war, bij, sp], ignore_index=True)[
        ["hh_type", "level", "y_disp", "rent", "b_norm", "price_base"]]
    a = a.merge(tb[["m_bas"]], left_on="hh_type", right_index=True, how="left")
    a["eqv"] = a["hh_type"].map(eqv)
    a["y_std"] = a["y_disp"] / a["eqv"]
    a["kappa_row"] = np.where(a["price_base"] == "vdb2023", 1.0, kappa)

    g0 = float(e.gamma_anchor)
    if not 0.0 <= g0 < 1.0:
        raise ValueError("gamma_anchor must be in [0, 1).")
    m_ex_pub = (((a["y_disp"] - a["rent"] - a["b_norm"]) / a["kappa_row"]
                 - g0 * a["m_bas"]) / (1.0 - g0))
    a["m_ex_pub"] = m_ex_pub
    a["basket_inverted"] = np.isfinite(m_ex_pub) & (m_ex_pub < a["m_bas"] - e.slack_tol)
    flr = np.where(np.isfinite(m_ex_pub), np.maximum(m_ex_pub, a["m_bas"]), np.nan)
    a["m_ex_own"] = np.nan
    for idx in a.groupby("hh_type", sort=False).groups.values():
        idx = list(idx)
        a.loc[idx, "m_ex_own"] = _extend(a.loc[idx, "y_std"].to_numpy(float),
                                         flr[idx], a.loc[idx, "m_bas"].to_numpy(float))

    donors = a[a["hh_type"].isin(e.sp_donors) & np.isfinite(a["m_ex_own"])]
    rate = donors.assign(s=(donors["m_ex_own"] - donors["m_bas"]) / donors["m_bas"])
    if (rate["s"] < -e.slack_tol).any() or rate.empty:
        raise ValueError("No usable donor slack rates.")
    s_lo = rate.groupby("level")["s"].min()
    s_hi = rate.groupby("level")["s"].max()
    s_pt = rate[rate["hh_type"] == e.sp_donor_point].set_index("level")["s"]
    imp = ~np.isfinite(a["m_ex_own"])
    need = set(a.loc[imp, "level"]) - set(s_pt.index)
    if need:
        raise ValueError(f"No donor slack rate for level(s) {sorted(need)}.")
    lvl = a["level"]
    a["imputed_basket"] = imp
    a["m_ex"] = np.where(imp, a["m_bas"] * (1 + lvl.map(s_pt)), a["m_ex_own"])
    a["m_ex_lo"] = np.where(imp, a["m_bas"] * (1 + lvl.map(s_lo)), a["m_ex_own"])
    a["m_ex_hi"] = np.where(imp, a["m_bas"] * (1 + lvl.map(s_hi)), a["m_ex_own"])
    a["b_bas"] = a["y_disp"] - a["rent"] - a["kappa_row"] * a["m_bas"]
    a["b_ex"] = a["y_disp"] - a["rent"] - a["kappa_row"] * a["m_ex"]
    if (a["b_ex"] > a["b_bas"] + e.slack_tol).any():
        raise ValueError("b_ex exceeds b_bas: the gamma ladder is inverted.")
    rt = a[np.isfinite(a["b_norm"]) & ~a["basket_inverted"]]
    if (((1 - g0) * rt["b_ex"] + g0 * rt["b_bas"] - rt["b_norm"]).abs() >= 0.01).any():
        raise ValueError("The m_ex inversion does not reproduce b_norm.")
    return a


def _pw_linear(x, y, z, extrap_lo: bool, extrap_hi: bool) -> float:
    """Linear interpolation of (x, y) at z; linear extrapolation from the two
    outermost points where allowed, else NaN."""
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    o = np.argsort(x, kind="stable")
    x, y = x[o], y[o]
    n = len(x)
    if n < 1 or not np.isfinite(z):
        return np.nan
    if n == 1:
        return y[0] if (extrap_lo and extrap_hi) else np.nan
    if z < x[0]:
        return y[0] + (y[1] - y[0]) / (x[1] - x[0]) * (z - x[0]) if extrap_lo else np.nan
    if z > x[-1]:
        return y[-1] + (y[-1] - y[-2]) / (x[-1] - x[-2]) * (z - x[-1]) if extrap_hi else np.nan
    return float(np.interp(z, x, y))


def _bracket_rent(y, rent, z) -> tuple[float, float]:
    """Rents of the nearest anchors below and above z (the outermost ones
    beyond the anchors), as (lower, higher)."""
    o = np.argsort(y, kind="stable")
    ya, ra = y[o], rent[o]
    below, above = np.flatnonzero(ya <= z), np.flatnonzero(ya >= z)
    lo = ra[below.max()] if len(below) else ra[0]
    hi = ra[above.min()] if len(above) else ra[-1]
    return min(lo, hi), max(lo, hi)


def quantile_envelope(anch: pd.DataFrame, axis: pd.DataFrame, src: Sources,
                      prm) -> pd.DataFrame:
    """Residuals per household type, income decile and rent scenario (CBS
    incomes plus the basic health premium per adult with `income_bridge`
    per_adult; baskets uprated by kappa with `quantile_kappa`):
    y_disp, rent, m_bas, m_ex, b_ex, b_bas, slack and the flags
    env_bas_ok (b_bas defined), gamma_identified (a usable example basket)
    and b_kind (gamma_indexed, point, b_bas_above_anchors)."""
    e = prm.envelope
    kappa = src.kappa if e.quantile_kappa else 1.0
    if e.income_bridge not in ("none", "per_adult"):
        raise ValueError("income_bridge must be 'none' or 'per_adult'.")
    rows = []
    for t, d in anch.groupby("hh_type", sort=False):
        bridge = e.basic_premium_eur * ADULTS[t] if e.income_bridge == "per_adult" else 0.0
        y, r, mx = (d[c].to_numpy(float) for c in ("y_std", "rent", "m_ex"))
        m_bas, eqv = float(d["m_bas"].iloc[0]), float(d["eqv"].iloc[0])
        for q in axis.itertuples(index=False):
            z = q.y_std
            below, above = z < y.min(), z > y.max()
            rlo, rhi = _bracket_rent(y, r, z)
            rint = min(max(_pw_linear(y, r, z, True, True), r.min()), r.max())
            mraw = _pw_linear(y, mx, z, False, False)
            censored = not q.env_eligible
            no_rent = censored or below
            if censored or below or above:
                mraw = np.nan
            base = dict(hh_type=t, point_id=q.quantile, y_std=z,
                        y_disp=z * eqv + bridge,
                        env_eligible=q.env_eligible, m_bas=m_bas,
                        m_ex=max(mraw, m_bas) if np.isfinite(mraw) else np.nan,
                        below_floor=below, above_ceiling=above, kappa_row=kappa)
            for name, rent in (("lo", rlo), ("interp", rint), ("hi", rhi)):
                if name in e.rent_scenarios:
                    rows.append({**base, "rent_scenario": name,
                                 "rent": np.nan if no_rent else rent})
    env = pd.DataFrame(rows)
    env["b_ex"] = env["y_disp"] - env["rent"] - env["kappa_row"] * env["m_ex"]
    env["b_bas"] = env["y_disp"] - env["rent"] - env["kappa_row"] * env["m_bas"]
    env["slack"] = env["m_ex"] - env["m_bas"]
    env["env_bas_ok"] = np.isfinite(env["b_bas"]) & env["env_eligible"]
    env["env_ex_ok"] = np.isfinite(env["b_ex"]) & env["env_eligible"]
    env["gamma_identified"] = (env["env_bas_ok"] & env["env_ex_ok"]
                               & np.isfinite(env["slack"])
                               & (env["slack"] > e.slack_min_eur))
    env["b_kind"] = np.select(
        [~env["env_bas_ok"], ~env["env_ex_ok"] & env["above_ceiling"],
         ~env["env_ex_ok"], env["gamma_identified"]],
        [None, "b_bas_above_anchors", "b_bas_no_basket", "gamma_indexed"],
        default="point")
    return env
