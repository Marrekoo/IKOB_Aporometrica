"""
Tours per month, commuting costs and the budget per tour X_M
(X_M calc.R, sections 8 and 13), from the residual envelope (`nibud`) and the
ODiN aggregates (`odin`).

The unit (`envelope.unit`) is a priced home-based tour (as X_M calc.R) or a
priced one-way journey; every count below is then of tours or of journeys.

Tours per household and month, per household type and income decile:

  * a rate per person per month for each age band: the decile's own rate
    where it has at least `min_band_n` persons, else the household type's,
    else the decile's over all types; children under 6 count as 0;
  * summed over the mean household composition, times (1 - share of tours
    as car passenger) for households of more than one person
    (`N_emp`, discretionary tours);
  * made non-decreasing over the deciles (pairwise averaging, as in the R
    script); `N_max` is the highest value of the household type; `N_min` is
    `n_min_fixed` (in journeys: times the journeys per tour) or, with
    `n_lower` = lowest_decile, the lowest value of the household type.

Commuting: unreimbursed commute tours per month (the household average, or
that of an unreimbursed commuter) times their mean cost (car km at the car
class's rate, other km at the cheapest PT rate).

The grid: every combination of gamma, N (N_min, N_emp, N_max), rent
scenario, commuting scenario and PT tariff basis. For each, the residual
b(gamma) minus fixed car costs and commuting is spread over N tours; X_M is
the largest budget per tour over the household's feasible bundles (public
transport, or the car class of the household type).

The envelope per cell: `low` and `high` are the minimum and maximum of X_M
over the whole grid (`spread` all, as X_M calc.R) or over gamma only with
the other assumptions at the central scenario (`spread` gamma); `central`
is X_M at the central scenario (gamma 0.5, N_emp, rent interp, commuting
average, PT nibud_flat). Flags: `upper_bound` (the cell lies above the
highest Nibud anchor: the residual after the minimum basket, an upper
bound), `gate_slack` (the high budget reaches more than `slack_share` of
ODiN's non-commute tours: the gate hardly binds at the top).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ikob2.envelope.sources import Sources

AGE_BAND_OF = {"hh_lft1": ("u6", False), "hh_lft2": ("a6_11", True),
               "hh_lft3": ("a12_17", True), "hh_lft4": ("a18p", True)}
QUANTILES = [f"Q{k}" for k in range(1, 11)]
CENTRAL = {"N_source": "N_emp", "rent_scenario": "interp",
           "commute_scenario": "average", "pt_basis": "nibud_flat"}
UNITS = ("tour", "journey")


def _unit(prm) -> str:
    u = prm.envelope.unit
    if u not in UNITS:
        raise ValueError(f"envelope.unit must be one of {UNITS}.")
    return u


def pairwise_pava(y: np.ndarray) -> np.ndarray:
    """Make y non-decreasing by repeatedly replacing the first decreasing
    pair with its mean (the R script's `pava`; unchanged when y has a
    missing value)."""
    y = np.asarray(y, dtype=float).copy()
    if len(y) < 2 or np.isnan(y).any():
        return y
    while True:
        v = np.flatnonzero(np.diff(y) < 0)
        if not len(v):
            return y
        i = v[0]
        y[i:i + 2] = y[i:i + 2].mean()


def tour_bounds(agg: dict, prm) -> pd.DataFrame:
    """Per household type and decile: N_emp, N_iso, N_min, N_max and
    N_commit_avg (tours or journeys per household and month)."""
    e = prm.envelope
    j = "j" if _unit(prm) == "journey" else ""
    dpm = e.days_per_month
    b = agg["band_rates"].rename(columns={f"{j}rate_disc": "_disc", f"{j}rate_comm": "_comm"})
    cell = b[b["level"] == "cell"].copy()
    typ = b[b["level"] == "type"][["hh_type", "age_band", "_disc", "n"]]
    qua = b[b["level"] == "quantile"][["quantile", "age_band", "_disc", "n"]]
    cell = (cell.merge(typ, on=["hh_type", "age_band"], suffixes=("", "_h"))
                .merge(qua, on=["quantile", "age_band"], suffixes=("", "_q")))
    cell["rate_used"] = np.where(
        cell["n"] >= e.min_band_n, cell["_disc"],
        np.where(cell["n_h"] >= e.min_band_n, cell["_disc_h"], cell["_disc_q"])) * dpm
    cell["rate_comm"] = cell["_comm"] * dpm

    comp = agg["composition"].melt(id_vars=["hh_type", "quantile"],
                                   value_vars=list(AGE_BAND_OF),
                                   var_name="hh_col", value_name="n_members")
    comp["age_band"] = comp["hh_col"].map(lambda c: AGE_BAND_OF[c][0])
    comp["observed"] = comp["hh_col"].map(lambda c: AGE_BAND_OF[c][1])
    comp = comp.merge(cell[["hh_type", "quantile", "age_band", "rate_used", "rate_comm"]],
                      on=["hh_type", "quantile", "age_band"], how="left")
    for c in ("rate_used", "rate_comm"):
        comp[c] = np.where(comp["observed"], comp[c].fillna(0.0), 0.0)
    comp["c_disc"] = comp["n_members"] * comp["rate_used"]
    comp["c_comm"] = comp["n_members"] * comp["rate_comm"]
    n = comp.groupby(["hh_type", "quantile"])[["c_disc", "c_comm"]].sum(min_count=0)

    ps = agg["passenger_share"].set_index("hh_type")[f"pass_share{'_j' if j else ''}"]
    defl = pd.Series(np.where(ps.index == "single", 1.0, 1.0 - ps), index=ps.index)
    n = n.reset_index()
    n["defl"] = n["hh_type"].map(defl)
    n["N_emp"] = n["c_disc"] * n["defl"]
    n["N_commit_avg"] = n["c_comm"] * n["defl"]
    n["q"] = n["quantile"].map({q: i for i, q in enumerate(QUANTILES)})
    n = n.sort_values(["hh_type", "q"])
    n["N_iso"] = n.groupby("hh_type")["N_emp"].transform(
        lambda s: pd.Series(pairwise_pava(s.to_numpy()), index=s.index))
    n["N_max"] = n.groupby("hh_type")["N_iso"].transform("max")
    if e.n_lower == "fixed":
        per_tour = float(agg["units"]["journeys_per_tour"].iloc[0]) if j else 1.0
        n["N_min"] = e.n_min_fixed * per_tour
    elif e.n_lower == "lowest_decile":
        n["N_min"] = n.groupby("hh_type")["N_iso"].transform("min")
    else:
        raise ValueError("envelope.n_lower must be 'fixed' or 'lowest_decile'.")
    return n.drop(columns=["q", "c_disc", "c_comm"]).reset_index(drop=True)


def bundles(agg: dict, src: Sources, prm) -> pd.DataFrame:
    """Priced bundles: one PT bundle per tariff basis and the car classes,
    with fixed EUR/month, EUR/km and fixed EUR per tour (or journey)."""
    e = prm.envelope
    km = agg["pt_km"].iloc[0]
    tot = km["train_km"] + km["btm_km"]
    w_train = km["train_km"] / tot if tot > 0 else 0.5
    chip = w_train * e.pt_train_eur_km + (1 - w_train) * e.pt_btm_eur_km
    boardings = e.boardings_per_journey if _unit(prm) == "journey" else e.boardings_per_tour
    pt = {"nibud_flat": (e.pt_nibud_eur_km, 0.0),
          "chipkaart": (chip, boardings * e.pt_btm_board_eur)}
    rows = [{"bundle": "pt_payg", "kind": "pt", "pt_basis": k, "fixed_eur": 0.0,
             "cost_per_km": v[0], "cost_fixed_per_tour": v[1]}
            for k, v in pt.items() if k in e.pt_bases]
    cars = src["car_bundles"]
    # X_M calc.R builds the bundles with tidyr::crossing(), which sorts, and
    # keeps the first row per car bundle: the car classes therefore belong to
    # the alphabetically first PT basis only ("chipkaart"). This changes
    # max_km, not X_M (PT, without fixed costs, always leaves more per tour).
    # `car_all_tariffs` puts them in every PT-tariff scenario.
    car_bases = list(e.pt_bases) if e.car_all_tariffs else [sorted(e.pt_bases)[0]]
    for basis in car_bases:
        for r in cars[cars["kind"] == "car"].itertuples(index=False):
            rows.append({"bundle": r.bundle, "kind": "car", "pt_basis": basis,
                         "fixed_eur": r.fixed_eur * src.kappa,
                         "cost_per_km": r.eur_per_km * src.kappa,
                         "cost_fixed_per_tour": 0.0})
    return pd.DataFrame(rows)


def commuting(agg: dict, bund: pd.DataFrame, src: Sources, prm) -> pd.DataFrame:
    """Per household type and decile: EUR per unreimbursed commute tour (or
    journey) and the commute tours (journeys) per month of an unreimbursed
    commuter."""
    e = prm.envelope
    j = _unit(prm) == "journey"
    car_class = src["car_class"].set_index("hh_type")["bundle"]
    car_rate = (bund[bund["kind"] == "car"].drop_duplicates("bundle")
                .set_index("bundle")["cost_per_km"])
    pt = bund[bund["kind"] == "pt"]
    pt_rate, pt_fix = pt["cost_per_km"].min(), pt["cost_fixed_per_tour"].min()
    c = agg["commute_tours"].copy()
    kc, kp = ("km_car_j", "km_pt_j") if j else ("km_car", "km_pt")
    c["eur_per_tour"] = (c[kc] * c["hh_type"].map(car_class).map(car_rate)
                         + c[kp] * pt_rate + c["share_pt"] * pt_fix)
    r = agg["commuter_rates"]
    col = "jrate" if j else "rate"
    cell = r[r["level"] == "cell"][["hh_type", "quantile", col, "n"]].rename(columns={col: "rate"})
    pool = r[r["level"] == "quantile"][["quantile", col]].rename(columns={col: "rate_pool"})
    w = cell.merge(pool, on="quantile", how="left")
    w["n_commute_pm"] = np.where(w["n"] >= e.min_commuter_n, w["rate"], w["rate_pool"]) \
        * e.days_per_month
    return c[["hh_type", "quantile", "eur_per_tour"]].merge(
        w[["hh_type", "quantile", "n_commute_pm"]], on=["hh_type", "quantile"], how="outer")


def grid(env: pd.DataFrame, nb: pd.DataFrame, comm: pd.DataFrame,
         bund: pd.DataFrame, src: Sources, prm) -> pd.DataFrame:
    """X_M (EUR per tour or journey) and max_km for every cell and scenario."""
    e = prm.envelope
    d = env.merge(nb[["hh_type", "quantile", "N_min", "N_iso", "N_max", "N_commit_avg"]],
                  left_on=["hh_type", "point_id"], right_on=["hh_type", "quantile"], how="left")
    d = d.merge(comm, on=["hh_type", "quantile"], how="left")
    scen = pd.MultiIndex.from_product(
        [e.gamma_grid, ["N_min", "N_emp", "N_max"], e.commute_scenarios],
        names=["gamma", "N_source", "commute_scenario"]).to_frame(index=False)
    d = d.merge(scen, how="cross")
    d = d[d["gamma_identified"] | (d["gamma"] == 0)]
    d["N"] = np.select([d["N_source"] == "N_min", d["N_source"] == "N_emp"],
                       [d["N_min"], d["N_iso"]], d["N_max"])
    d["b"] = np.where(d["gamma_identified"],
                      (1 - d["gamma"]) * d["b_ex"] + d["gamma"] * d["b_bas"], d["b_bas"])
    d["commute_cost"] = np.select(
        [d["commute_scenario"] == "average", d["commute_scenario"] == "worker"],
        [(d["N_commit_avg"] * d["eur_per_tour"]).fillna(0.0),
         (d["n_commute_pm"] * d["eur_per_tour"]).fillna(0.0)], 0.0)
    car_class = src["car_class"].set_index("hh_type")["bundle"]
    d = d.merge(bund, how="cross")
    d = d[(d["kind"] == "pt") | (d["bundle"] == d["hh_type"].map(car_class))]
    d["residual"] = d["b"] - (d["fixed_eur"] + d["commute_cost"] + e.urbanity_adj_eur)
    d["feasible"] = d["env_bas_ok"] & np.isfinite(d["residual"]) & (d["residual"] >= 0)
    ok_n = np.isfinite(d["N"]) & (d["N"] > 0)
    d["budget"] = np.where(ok_n, d["residual"] / d["N"].where(ok_n, 1.0), np.nan)
    d["max_km"] = np.where(d["feasible"] & np.isfinite(d["budget"]),
                           np.maximum(0.0, (d["budget"] - d["cost_fixed_per_tour"])
                                      / d["cost_per_km"]), np.nan)
    d["budget_f"] = d["budget"].where(d["feasible"])
    keys = ["hh_type", "point_id", "rent_scenario", "commute_scenario", "gamma",
            "N_source", "pt_basis"]
    g = d.groupby(keys, sort=False)
    out = g.agg(X_M=("budget_f", "max"), max_km=("max_km", "max"), N=("N", "first"),
                b=("b", "first"), b_kind=("b_kind", "first"),
                gamma_identified=("gamma_identified", "first")).reset_index()
    return out


def gate(g: pd.DataFrame, prm=None, km_cdf: pd.DataFrame | None = None) -> pd.DataFrame:
    """Per household type and decile: X_M_lo, X_M_hi, km_max_M_lo,
    km_max_M_hi (over the grid, or over gamma with `spread` gamma),
    X_M_central, upper_bound and (with km_cdf) gate_slack."""
    spread = prm.envelope.spread if prm is not None else "all"
    if spread not in ("all", "gamma"):
        raise ValueError("envelope.spread must be 'all' or 'gamma'.")
    central = g
    for k, v in CENTRAL.items():
        central = central[central[k] == v]
    sel = central if spread == "gamma" else g
    k = sel.groupby(["hh_type", "point_id"], sort=False)
    out = k.agg(X_M_lo=("X_M", "min"), X_M_hi=("X_M", "max"),
                km_max_M_lo=("max_km", "min"), km_max_M_hi=("max_km", "max")).reset_index()
    mid = central[(central["gamma"] == 0.5) | ~central["gamma_identified"]]
    out = out.merge(mid.groupby(["hh_type", "point_id"])["X_M"].first().rename("X_M_central"),
                    on=["hh_type", "point_id"], how="left")
    ub = g.groupby(["hh_type", "point_id"])["b_kind"].agg(
        lambda s: bool((s == "b_bas_above_anchors").any())).rename("upper_bound")
    out = out.merge(ub, on=["hh_type", "point_id"], how="left")
    if km_cdf is not None and prm is not None:
        c = km_cdf[km_cdf["unit"] == prm.envelope.unit]
        share = np.interp(out["km_max_M_hi"].fillna(0.0), c["km"], c["share"], left=0.0)
        out["gate_slack"] = np.where(out["km_max_M_hi"].notna(),
                                     share > prm.envelope.slack_share, False)
    return out


def reference_budgets(gt: pd.DataFrame, prm=None) -> pd.DataFrame:
    """The table of data/envelope/reference_budgets.csv (EUR per tour or per
    journey, rounded to cents; D1 without bounds), with the columns central,
    unit, upper_bound and gate_slack after the envelope itself."""
    t = gt.assign(income_class=gt["point_id"].str.replace("Q", "D"))
    t = t.rename(columns={"hh_type": "household_type", "X_M_lo": "low", "X_M_hi": "high",
                          "km_max_M_lo": "km_low", "km_max_M_hi": "km_high",
                          "X_M_central": "central"})
    cols = ["household_type", "income_class", "low", "high", "km_low", "km_high"]
    extra = [c for c in ("central", "upper_bound", "gate_slack") if c in t.columns]
    t = t[cols + extra].copy()
    if prm is not None:
        t.insert(len(cols) + 1, "unit", prm.envelope.unit)
    order = {f"D{k}": k for k in range(1, 11)}
    t = t.sort_values(["household_type", "income_class"],
                      key=lambda s: s.map(order) if s.name == "income_class" else s)
    for c in ("low", "high", "km_low", "km_high", "central"):
        if c in t.columns:
            t[c] = t[c].round(2)
    return t.reset_index(drop=True)
