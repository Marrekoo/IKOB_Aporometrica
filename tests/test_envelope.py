"""The reference-budget envelope (ikob2.envelope): source tables, stages,
and the regression against data/envelope/reference_budgets.csv."""

import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ikob2.envelope import income, nibud, odin, xm
from ikob2.envelope.sources import load_sources
from ikob2.params import DEFAULTS as P

SOURCES = Path(__file__).resolve().parents[1] / "data" / "envelope" / "sources"
REFERENCE = SOURCES.parent / "reference_budgets.csv"


@pytest.fixture(scope="module")
def pub_prm():
    """The defaults at the sources' own price levels, where the published
    anchors and saldi can be checked as printed."""
    return P.with_values({"envelope.price_base": "published"})


@pytest.fixture(scope="module")
def src():
    return load_sources(SOURCES)


# ── sources and baskets ──────────────────────────────────────────────

def test_every_source_table_names_its_source():
    for f in SOURCES.glob("*.csv"):
        s = pd.read_csv(f)["source"]
        assert s.notna().all() and (s.str.len() > 10).all(), f.name


def test_basket_reproduces_the_published_totals_and_parts(src):
    agg = nibud.basket_aggregates(src)
    assert len(agg) == 12
    # the minimum non-mobility basket and mobility of the four model types
    tb = nibud.type_baskets(src)
    assert tb["m_bas"].to_dict() == {"single": 1076, "couple": 1734,
                                     "single_parent": 1745, "couple_children": 2288}
    assert tb["mob_min"].to_dict() == {"single": 36, "couple": 73,
                                       "single_parent": 97, "couple_children": 133}


def test_a_wrong_basket_entry_is_refused(src):
    bad = dict(src.tables)
    b = bad["nibud_basket"].copy()
    b.loc[(b.post == "voeding") & (b.hh_key == "single"), "eur"] += 10
    bad["nibud_basket"] = b
    with pytest.raises(ValueError, match="published totals"):
        nibud.basket_aggregates(type(src)(bad))


# ── anchors, income axis, residuals ──────────────────────────────────

def test_anchors_reproduce_b_norm_and_the_published_saldi(src, pub_prm):
    a = nibud.anchors(src, pub_prm)
    war = a[a["b_norm"].notna()]
    # Warnaar's couple at minimum wage (b_norm 316) implies an example basket
    # below the minimum basket: floored at m_bas, no gamma range there
    inv = war[war["basket_inverted"]]
    assert list(zip(inv["hh_type"], inv["level"])) == [("couple", "wml")]
    assert (inv["m_ex"] == inv["m_bas"]).all()
    ok = war[~war["basket_inverted"]]
    np.testing.assert_allclose((ok["b_ex"] + ok["b_bas"]) / 2, ok["b_norm"], atol=1e-9)
    # social assistance: y - rent - m_bas - mob_min is the published saldo
    tb = nibud.type_baskets(src)
    bij = a[a["level"] == "bijstand"].set_index("hh_type")
    pub = src["bijstand_published"].set_index("hh_type")
    implied = bij["y_disp"] - bij["rent"] - tb["m_bas"] - tb["mob_min"]
    assert (implied.reindex(pub.index) - pub["saldo"]).abs().max() < 2
    # single parents borrow the donors' slack: m_ex between the donor bounds
    sp = a[a["hh_type"] == "single_parent"]
    assert sp["imputed_basket"].all()
    assert (sp["m_ex_lo"] <= sp["m_ex"] + 1e-9).all() and (sp["m_ex"] <= sp["m_ex_hi"] + 1e-9).all()


def test_income_axis(src, pub_prm):
    ax = income.income_axis(src, pub_prm).set_index("quantile")
    assert ax.loc["Q1", "y_std"] == pytest.approx(19.9e3 / 12)       # p10, censored
    assert not ax.loc["Q1", "env_eligible"] and ax.loc["Q2":, "env_eligible"].all()
    assert (np.diff(ax["y_std"]) > 0).all()
    p80, p90 = 48.9e3 / 12, 58.6e3 / 12                              # Pareto p95
    alpha = np.log(2) / np.log(p90 / p80)
    assert ax.loc["Q10", "y_std"] == pytest.approx(p90 * 2 ** (1 / alpha))
    # D5 (p45) lies between p40 and p50
    assert 30.8e3 / 12 < ax.loc["Q5", "y_std"] < 34.6e3 / 12


def test_envelope_residuals(src, pub_prm):
    env = nibud.quantile_envelope(nibud.anchors(src, pub_prm), income.income_axis(src, pub_prm),
                                  src, pub_prm)
    ok = env[env["env_bas_ok"]]
    np.testing.assert_allclose(
        ok["b_bas"], ok["y_disp"] - ok["rent"] - src.kappa * ok["m_bas"])
    assert (ok["b_ex"].fillna(-np.inf) <= ok["b_bas"] + 1e-9).all()
    assert not env.loc[env["point_id"] == "Q1", "env_bas_ok"].any()
    above = env[env["b_kind"] == "b_bas_above_anchors"]
    assert above["above_ceiling"].all() and above["m_ex"].isna().all()


# ── ODiN stage on a synthetic diary ──────────────────────────────────

def _diary():
    """Two persons: p1 home-work-home-shop-home by car; p2 with a missing
    destination in the middle."""
    rows = []
    def trip(person, nr, doel, mode, motive, km, hh=2, inc=5):
        rows.append({"person": person, "verpl": 1, "trip_no": nr, "w_person": 1.0,
                     "mode": mode, "motive": motive, "doel": doel,
                     "dist_hm": km * 10, "vert_uur": 8, "hh_size": 2, "hh_comp": hh,
                     "hh_lft1": 0, "hh_lft2": 0, "hh_lft3": 0, "hh_lft4": 2,
                     "income": inc, "age": 40, "wrk_vervw": 3, "wrk_verg": 0,
                     "bet_werk": 2})
    trip("p1", 1, 4, 1, 1, 10)      # to work (commute)
    trip("p1", 2, 1, 1, 1, 10)      # home: tour 1 ends
    trip("p1", 3, 5, 1, 4, 3)       # shopping
    trip("p1", 4, 1, 1, 4, 3)       # home: tour 2 ends
    trip("p2", 1, 5, 5, 4, 2)       # bicycle (unpriced)
    trip("p2", 2, np.nan, 1, 4, 5)  # destination missing
    trip("p2", 3, 1, 1, 4, 5)       # numbering undefined from here on
    trip("p2", 4, 5, 1, 4, 5)
    return pd.DataFrame(rows)


def test_tours_follow_trips_home_and_missing_destinations(src):
    d = _diary()
    p = odin.persons(d, src["odin_household_types"])
    t = odin.tours(d, p)
    p1 = t[t["person"] == "p1"].sort_values("tour")
    assert list(p1["tour"]) == [1.0, 2.0]
    assert list(p1["km"]) == [20.0, 6.0] and list(p1["any_commute"]) == [True, False]
    p2 = t[t["person"] == "p2"]
    # tour 1: bicycle + car trip with the missing destination; the trips
    # after it share one undefined tour
    assert len(p2) == 2 and p2["tour"].isna().sum() == 1
    assert p2.loc[p2["tour"].isna(), "km"].item() == 10.0


def test_crosswalk_splits_classes_over_deciles():
    p = pd.DataFrame({"income": [1, 1, 2, 3], "w_person": [1.0, 1.0, 1.0, 1.0]})
    cw = odin.crosswalk(p, n_q=4)                    # classes: 50%, 25%, 25%
    assert cw.groupby("income")["share"].sum().round(12).eq(1).all()
    assert cw[cw.income == 1].set_index("quantile")["share"].to_dict() == {"Q1": 0.5, "Q2": 0.5}


# ── tours per month and X_M ──────────────────────────────────────────

def test_pairwise_pava():
    np.testing.assert_allclose(xm.pairwise_pava([1, 3, 2, 4]), [1, 2.5, 2.5, 4])
    np.testing.assert_allclose(xm.pairwise_pava([3, 2, 1]), [2, 2, 2])
    assert np.isnan(xm.pairwise_pava([1, np.nan, 0])).any()   # unchanged


def test_budget_per_tour_is_residual_over_tours(src, pub_prm):
    env = pd.DataFrame([{"hh_type": "single", "point_id": "Q5", "rent_scenario": "lo",
                         "b_ex": 500.0, "b_bas": 900.0, "env_bas_ok": True,
                         "gamma_identified": True, "b_kind": "gamma_indexed"}])
    nb = pd.DataFrame([{"hh_type": "single", "quantile": "Q5", "N_min": 8.0,
                        "N_iso": 20.0, "N_max": 40.0, "N_commit_avg": 4.0}])
    comm = pd.DataFrame([{"hh_type": "single", "quantile": "Q5",
                          "eur_per_tour": 5.0, "n_commute_pm": 10.0}])
    prm = pub_prm.with_values({"envelope.unit": "tour", "envelope.n_lower": "fixed"})
    agg = {"pt_km": pd.DataFrame({"train_km": [1.0], "btm_km": [1.0]})}
    bund = xm.bundles(agg, src, prm)
    g = xm.grid(env, nb, comm, bund, src, prm).set_index(
        ["gamma", "N_source", "commute_scenario", "pt_basis"])
    # gamma 0.5: b = 700; average commuting: 4 x 5 = 20; N_emp 20 -> 34
    assert g.loc[(0.5, "N_emp", "average", "nibud_flat"), "X_M"] == pytest.approx(680 / 20)
    assert g.loc[(1.0, "N_min", "none", "nibud_flat"), "X_M"] == pytest.approx(900 / 8)
    assert g.loc[(0.0, "N_max", "worker", "chipkaart"), "X_M"] == pytest.approx(450 / 40)
    gt = xm.gate(g.reset_index())
    assert gt["X_M_lo"].item() == pytest.approx(450 / 40)
    assert gt["X_M_hi"].item() == pytest.approx(900 / 8)


# ── regression: the published table ──────────────────────────────────

AGGREGATES = SOURCES.parent / "odin"


def test_defaults_reproduce_the_adopted_table():
    from ikob2.cli.envelope import build

    got = build(SOURCES, AGGREGATES / P.envelope.aggregates, P)["reference_budgets"]
    pd.testing.assert_frame_equal(got, pd.read_csv(REFERENCE), check_dtype=False)
    assert set(got["unit"]) == {"journey"}


def test_price_base_2022_converts_every_input_from_its_own_date(src, pub_prm):
    pf = nibud.price_factors(src, P)
    assert pf["basket"] == pytest.approx(121.43 / 123.23)
    assert pf["warnaar"] == pytest.approx(121.43 / 123.23 / src.kappa)
    assert pf["income"] == pytest.approx(121.43 / 126.04)
    assert pf["kappa"] == 1.0
    assert nibud.price_factors(src, pub_prm)["basket"] == 1.0
    a0 = nibud.anchors(src, pub_prm).set_index(["hh_type", "level"])
    a1 = nibud.anchors(src, P).set_index(["hh_type", "level"])
    war = a0["price_base"] == "warnaar"
    # an anchor residual is the published one in 2022 euros
    np.testing.assert_allclose(a1.loc[war, "b_bas"], a0.loc[war, "b_bas"] * pf["warnaar"])


def test_published_aggregates_match_a_fresh_aggregation_when_odin_is_there():
    root = os.environ.get("IKOB_DATA_ROOT")
    f = Path(root) / "inputs" / "odin" / "ODIN_23.csv" if root else None
    if f is None or not f.exists():
        pytest.skip("ODiN 2023 microdata not available (IKOB_DATA_ROOT)")
    src = load_sources(SOURCES)
    fresh = odin.aggregates(odin.read_odin(f), src["odin_household_types"])
    pub = odin.read_aggregates(AGGREGATES / "2023")
    for name, df in fresh.items():
        np.testing.assert_allclose(df.select_dtypes("number").to_numpy(float),
                                   pub[name].select_dtypes("number").to_numpy(float),
                                   rtol=1e-12, atol=1e-12, err_msg=name)


@pytest.mark.parametrize("switch, value", [
    ("unit", "tour"), ("n_lower", "fixed"), ("spread", "all"),
    ("gamma_anchor", 0.0), ("income_bridge", "none"), ("price_base", "published"),
    ("aggregates", "2023"), ("car_all_tariffs", False)])
def test_every_switch_builds_a_valid_envelope(switch, value):
    from ikob2.cli.envelope import build
    from ikob2.segments.bridge import validate_envelope

    prm = P.with_values({f"envelope.{switch}": value})
    agg = AGGREGATES / prm.envelope.aggregates
    t = build(SOURCES, agg, prm)["reference_budgets"]
    ok = t.dropna(subset=["low", "high"])
    assert len(ok) == 36 and (ok["low"] <= ok["central"] + 1e-9).all() \
        and (ok["central"] <= ok["high"] + 1e-9).all()
    validate_envelope(ok[["household_type", "income_class", "low", "high"]])
    assert set(t["unit"]) == {prm.envelope.unit}
