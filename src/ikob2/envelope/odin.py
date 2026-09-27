"""
ODiN -> aggregate tables for the envelope (X_M calc.R, sections 9, 12, 13).

The only stage that reads microdata. It returns weighted means and person
counts per cell and nothing else, so the aggregates can be inspected (and,
where the ODiN licence allows, shared) without the microdata. Prices and the
modelling rules (pooling thresholds, deflators, bounds) are applied in `xm`.

Definitions follow X_M calc.R:

  * persons: household type (HHSam), income class (HHGestInkG, 11 =
    unknown), age band, person weight FactorP, commuting mode and whether
    the commute is reimbursed;
  * income classes -> deciles: ODiN's classes do not have exactly 10% of the
    weight, so each class is split over the deciles it overlaps
    (`crosswalk`); a person counts in each decile with weight
    FactorP x share;
  * tours: regular trips (Verpl = 1) without touring (MotiefV 9) or business
    (2, 3), in order; a new tour starts after every trip home (Doel = 1);
    tours with a priced main mode (KHvm 1, 2, 3, 4, 7) count;
  * rates are per survey day (tours on the diary day); `xm` scales them to a
    month.

Tables returned (dict name -> DataFrame):

  crosswalk        income, quantile, share
  band_rates       level (cell | type | quantile), hh_type, quantile, age_band,
                   rate_disc, rate_comm (tours per day: non-commute, commute
                   unreimbursed), n (persons)
  composition      hh_type, quantile, hh_size, hh_lft1..hh_lft4 (weighted means)
  passenger_share  hh_type, pass_share (share of tours with a car-passenger leg)
  commute_tours    hh_type, quantile, km_car, km_pt, share_pt, n (unreimbursed
                   commute tours: weighted mean km by car, by other modes)
  commuter_rates   level (cell | quantile), hh_type, quantile, rate (commute
                   tours per day of unreimbursed commuters), n
  pt_km            train_km, btm_km (km of train and bus/tram/metro trips)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

COLS = {"OPID": "person", "Verpl": "verpl", "VerplNr": "trip_no",
        "FactorP": "w_person", "KHvm": "mode", "MotiefV": "motive",
        "Doel": "doel", "AfstV": "dist_hm", "VertUur": "vert_uur",
        "HHPers": "hh_size", "HHSam": "hh_comp", "HHLft1": "hh_lft1",
        "HHLft2": "hh_lft2", "HHLft3": "hh_lft3", "HHLft4": "hh_lft4",
        "HHGestInkG": "income", "Leeftijd": "age", "WrkVervw": "wrk_vervw",
        "WrkVerg": "wrk_verg", "BetWerk": "bet_werk"}
PERSON_VARS = ("w_person", "hh_size", "hh_comp", "hh_lft1", "hh_lft2",
               "hh_lft3", "hh_lft4", "income", "age", "wrk_vervw", "wrk_verg",
               "bet_werk")
PRICED = (1, 2, 3, 4, 7)            # KHvm: car driver, passenger, train, BTM, other
CAR_PASSENGER, TRAIN, BTM = 2, 3, 4
CAR_MODES = (1, 2)
HOME, COMMUTE, TOURING, BUSINESS = 1, 1, 9, (2, 3)
WRK_CAR, WRK_PT = (3, 4, 5, 6), (7, 8)
INCOME_UNKNOWN, HH_UNKNOWN = 11, (10, 11)
AGE_BANDS = (("u6", 5), ("a6_11", 11), ("a12_17", 17))


def read_odin(path: str | Path) -> pd.DataFrame:
    """The columns of COLS, renamed; numeric codes rounded to integers."""
    raw = pd.read_csv(path, sep=";", decimal=",", usecols=list(COLS),
                      low_memory=False).rename(columns=COLS)
    raw["person"] = raw["person"].astype(str)
    for c in raw.columns:
        if c not in ("person", "w_person", "dist_hm"):
            raw[c] = pd.to_numeric(raw[c], errors="coerce").round()
    return raw


def persons(odin: pd.DataFrame, hh_types: pd.DataFrame) -> pd.DataFrame:
    """One row per person: the first non-missing value of each person
    variable, household type, age band and commute status."""
    p = odin.groupby("person", sort=True)[list(PERSON_VARS)].first()
    for c in ("hh_size", "hh_lft1", "hh_lft2", "hh_lft3", "hh_lft4"):
        p[c] = p[c].where(~p[c].isin(HH_UNKNOWN))
    p["income"] = p["income"].where(p["income"] != INCOME_UNKNOWN)
    p["hh_type"] = p["hh_comp"].map(dict(zip(hh_types["hh_comp"], hh_types["hh_type"])))
    age = p["age"]
    p["age_band"] = np.select([age <= lim for _, lim in AGE_BANDS],
                              [b for b, _ in AGE_BANDS], default="a18p")
    p["commute_mode"] = np.select(
        [p["wrk_vervw"].isin(WRK_CAR), p["wrk_vervw"].isin(WRK_PT)],
        ["car", "pt"], default=None)
    reimb = p["wrk_verg"].where(p["wrk_verg"].isin((0, 1))) == 1
    p["reimbursed"] = reimb.where(p["wrk_verg"].isin((0, 1)))
    works = p["bet_werk"].isin((2, 3))
    p["unreimb_commuter"] = (works & p["commute_mode"].isin(["car", "pt"])
                             & (p["reimbursed"] == False))  # noqa: E712
    return p


def crosswalk(p: pd.DataFrame, n_q: int = 10) -> pd.DataFrame:
    """Share of each income class (by person weight) in each decile."""
    w = p.dropna(subset=["income"]).groupby("income")["w_person"].sum().sort_index()
    share = w / w.sum()
    c_hi = share.cumsum()
    c_lo = c_hi - share
    rows = []
    for inc in w.index:
        for k in range(1, n_q + 1):
            q_lo, q_hi = (k - 1) / n_q, k / n_q
            ov = max(0.0, min(c_hi[inc], q_hi) - max(c_lo[inc], q_lo))
            s = ov / (c_hi[inc] - c_lo[inc])
            if s > 1e-12:
                rows.append({"income": inc, "quantile": f"Q{k}", "share": s})
    return pd.DataFrame(rows)


def _r_cumsum(x: pd.Series) -> pd.Series:
    """Cumulative sum that stays missing from the first missing value on."""
    out = x.astype(float).cumsum()
    first_na = x.isna().to_numpy().argmax() if x.isna().any() else len(x)
    out.iloc[first_na:] = np.nan
    return out


def tours(odin: pd.DataFrame, p: pd.DataFrame) -> pd.DataFrame:
    """Priced home-based tours: person, any_commute, any_pass, mode_max, km,
    dep_hour, with the person's household type and commute status.

    A person's tour number rises by one after every trip home. As in the R
    script, a trip whose destination is missing leaves the numbering
    undefined from the next trip on; those trips form one tour of their own."""
    t = odin[(odin["verpl"] == 1) & odin["trip_no"].notna()]
    t = t[t["motive"].notna() & (t["motive"] != TOURING) & ~t["motive"].isin(BUSINESS)]
    t = t.assign(km=t["dist_hm"] / 10.0).sort_values(["person", "trip_no"], kind="stable")
    home = (t["doel"] == HOME).astype(float).where(t["doel"].notna())
    prev = home.groupby(t["person"]).shift(1)
    prev[~t.duplicated("person")] = 0.0
    t = t.assign(tour=prev.groupby(t["person"]).transform(_r_cumsum) + 1,
                 _km=t["km"].fillna(-np.inf),
                 priced=t["mode"].isin(PRICED),
                 passenger=t["mode"] == CAR_PASSENGER,
                 commute=t["motive"] == COMMUTE)
    g = t.groupby(["person", "tour"], dropna=False, sort=False)
    out = g.agg(any_priced=("priced", "max"), any_pass=("passenger", "max"),
                any_commute=("commute", "max"), km=("km", "sum"),
                dep_hour=("vert_uur", lambda s: s.iloc[0]))
    out["mode_max"] = t.loc[g["_km"].idxmax().to_numpy(), "mode"].to_numpy()
    out = out.reset_index()
    out = out[out["any_priced"]]
    return out.merge(p[["hh_type", "reimbursed", "unreimb_commuter", "w_person"]],
                     left_on="person", right_index=True, how="left")


def _wmean_na(d: pd.DataFrame, col: str, keys) -> pd.Series:
    """Weighted mean of col (weight w_eff) per group, skipping missing values."""
    ok = d[col].notna()
    x = d[ok].assign(_num=d.loc[ok, col] * d.loc[ok, "w_eff"])
    g = x.groupby(list(keys))
    return g["_num"].sum() / g["w_eff"].sum()


def _wmean(d: pd.DataFrame, col: str, keys) -> pd.DataFrame:
    """Weighted mean of col (weight w_eff) and person count per group."""
    x = d.assign(_num=d[col] * d["w_eff"])
    g = x.groupby(list(keys), dropna=False)
    return pd.DataFrame({"rate": g["_num"].sum() / g["w_eff"].sum(),
                         "n": g["person"].nunique()}).reset_index()


def aggregates(odin: pd.DataFrame, hh_types: pd.DataFrame) -> dict:
    """All aggregate tables (see the module docstring)."""
    p = persons(odin, hh_types)
    cw = crosswalk(p)
    pq = (p.reset_index().merge(cw, on="income")
          .assign(w_eff=lambda d: d["w_person"] * d["share"]))
    tr = tours(odin, p)
    by_person = tr.groupby("person").agg(
        n_all=("any_commute", "size"),
        n_disc=("any_commute", lambda s: int((~s).sum())),
        n_commute=("any_commute", "sum"),
        pass_share=("any_pass", "mean"))
    committed = tr[tr["any_commute"] & (tr["reimbursed"] == False)]  # noqa: E712
    by_person["n_committed"] = committed.groupby("person").size().reindex(
        by_person.index).fillna(0)

    rf = pq.merge(by_person, left_on="person", right_index=True, how="left")
    for c in ("n_all", "n_disc", "n_commute", "n_committed"):
        rf[c] = rf[c].fillna(0)
    rf = rf[rf["hh_type"].notna() & (rf["w_eff"] > 0)]

    cell = ("hh_type", "quantile", "age_band")
    disc = [_wmean(rf, "n_disc", k).rename(columns={"rate": "rate_disc"}).assign(level=lv)
            for lv, k in (("cell", cell), ("type", ("hh_type", "age_band")),
                          ("quantile", ("quantile", "age_band")))]
    comm = _wmean(rf, "n_committed", cell).rename(columns={"rate": "rate_comm"})
    band = pd.concat(disc, ignore_index=True).merge(
        comm.assign(level="cell").drop(columns="n"), on=["level", *cell], how="left")

    comp_cols = ["hh_size", "hh_lft1", "hh_lft2", "hh_lft3", "hh_lft4"]
    comp = pd.DataFrame({c: _wmean_na(rf, c, ("hh_type", "quantile"))
                         for c in comp_cols}).reset_index()

    ps = rf[(rf["n_all"] > 0) & rf["pass_share"].notna()]
    pass_share = _wmean_na(ps, "pass_share", ("hh_type",)).rename(
        "pass_share").reset_index()

    ct = tr[tr["any_commute"] & tr["dep_hour"].notna() & tr["hh_type"].notna()]
    ct = ct.merge(pq[["person", "quantile", "w_eff"]], on="person")
    ct = ct[np.isfinite(ct["km"]) & (ct["km"] > 0) & (ct["reimbursed"] == False)]  # noqa: E712
    is_car = ct["mode_max"].isin(CAR_MODES)
    ct = ct.assign(km_car=ct["km"] * is_car, km_pt=ct["km"] * ~is_car, pt=(~is_car).astype(float))
    commute = pd.DataFrame({c: _wmean_na(ct, c, ("hh_type", "quantile"))
                            for c in ("km_car", "km_pt", "pt")}).rename(
        columns={"pt": "share_pt"})
    commute["n"] = ct.groupby(["hh_type", "quantile"]).size()
    commute = commute.reset_index()

    uc = pq[pq["unreimb_commuter"]].merge(by_person[["n_commute"]], left_on="person",
                                          right_index=True, how="left")
    uc["n_commute"] = uc["n_commute"].fillna(0)
    commuters = pd.concat([
        _wmean(uc, "n_commute", ("hh_type", "quantile")).assign(level="cell"),
        _wmean(uc, "n_commute", ("quantile",)).assign(level="quantile")], ignore_index=True)

    trips = odin[(odin["verpl"] == 1) & odin["trip_no"].notna()]
    trips = trips[trips["motive"].notna() & (trips["motive"] != TOURING)
                  & ~trips["motive"].isin(BUSINESS)]
    km = trips["dist_hm"] / 10.0
    pt_km = pd.DataFrame({"train_km": [km[trips["mode"] == TRAIN].sum()],
                          "btm_km": [km[trips["mode"] == BTM].sum()]})
    return {"crosswalk": cw, "band_rates": band, "composition": comp,
            "passenger_share": pass_share, "commute_tours": commute,
            "commuter_rates": commuters, "pt_km": pt_km}


def write_aggregates(tables: dict, folder: str | Path) -> None:
    """Write the aggregate tables as CSV files."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    for name, df in tables.items():
        df.to_csv(folder / f"{name}.csv", index=False)


def read_aggregates(folder: str | Path) -> dict:
    """Read the tables written by `write_aggregates`."""
    folder = Path(folder)
    names = ("crosswalk", "band_rates", "composition", "passenger_share",
             "commute_tours", "commuter_rates", "pt_km")
    return {n: pd.read_csv(folder / f"{n}.csv") for n in names}
