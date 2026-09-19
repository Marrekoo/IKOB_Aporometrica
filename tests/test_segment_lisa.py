"""LISA sector jobs, the buurt imputation, and sector -> income pools."""

import numpy as np
import pandas as pd
import pytest

from ikob2.segments.config import INCOME_CLASSES
from ikob2.segments.jobs import sector_income_weights, sector_pools
from ikob2.segments.jobs_impute import (
    COVARIATES,
    SectorModel,
    buurt_covariates,
    fit_sector_model,
    impute_sector_jobs,
    municipal_covariates,
    parse_education_shares,
)
from ikob2.segments.lisa import (
    SECTOR_TO_SBI,
    SECTORS,
    lisa_gemeente_of_buurten,
    parse_lisa_sectors,
    sector_wages,
)


# ── LISA table ───────────────────────────────────────────────────────

LABELS = {"L01": "L01. Landbouw & visserij", "L02": "L02. Industrie"}


def _raw(year=2022, drop=()):
    rows = []
    for year_ in (2016, year):
        for g, base in (("Alpha", 100), ("Beta", 300)):
            for i, s in enumerate(SECTORS):
                if (g, s) in drop and year_ == year:
                    continue
                rows.append({"Jaar": year_, "Provincie": "P", "COROP_gebied": "C",
                             "Gemeente": g, "LISA_sector": f"{s}. x{i}",
                             "Banen": base + 10 * i + (5 if year_ == 2016 else 0),
                             "Vestigingen": 1})
    return pd.DataFrame(rows)


def test_parse_lisa_selects_year_and_orders_sectors():
    wide = parse_lisa_sectors(_raw(), 2022)
    assert list(wide.columns) == list(SECTORS)
    assert list(wide.index) == ["Alpha", "Beta"]
    assert wide.loc["Alpha", "L01"] == 100 and wide.loc["Beta", "L15"] == 440
    assert parse_lisa_sectors(_raw(), 2016).loc["Alpha", "L01"] == 105


def test_parse_lisa_absent_cells_are_zero_and_errors():
    wide = parse_lisa_sectors(_raw(drop={("Alpha", "L03")}), 2022)
    assert wide.loc["Alpha", "L03"] == 0
    with pytest.raises(ValueError, match="No LISA rows for year 1999"):
        parse_lisa_sectors(_raw(), 1999)
    bad = _raw()
    bad.loc[bad.index[-1], "LISA_sector"] = "L99. nope"   # a 2022 row
    with pytest.raises(ValueError, match="differ from the expected 15"):
        parse_lisa_sectors(bad, 2022)
    neg = _raw()
    neg.loc[neg.index[-1], "Banen"] = -5
    with pytest.raises(ValueError, match="negative"):
        parse_lisa_sectors(neg, 2022)


def test_lisa_gemeente_mapping_exact_alias_and_unmapped():
    kwb = pd.DataFrame({
        "buurtcode": ["BU1", "BU2", "BU3", "BU4", "BU5"],
        "gemeentenaam": ["Utrecht", "Hengelo", "Brielle", "Buitenland",
                         "Weesp"]})
    names = ["Utrecht", "Hengelo (O.)", "Voorne aan Zee", "Amsterdam"]
    out = lisa_gemeente_of_buurten(kwb, names)
    assert out["BU1"] == "Utrecht" and out["BU2"] == "Hengelo (O.)"
    assert out["BU3"] == "Voorne aan Zee" and out["BU5"] == "Amsterdam"
    assert pd.isna(out["BU4"])


# ── Wages ────────────────────────────────────────────────────────────

def _wage_raw(overrides=None):
    rows = []
    for sector, sbi in SECTOR_TO_SBI.items():
        for j, k in enumerate(sbi):
            rows.append({"BedrijfstakkenBranchesSBI2008": k + " ",
                         "Banen_1": 100.0 * (j + 1), "Uurloon_3": 20.0 + j})
    df = pd.DataFrame(rows)
    return df


def test_sector_wages_are_job_weighted_over_sbi_sections():
    w = sector_wages(_wage_raw())
    assert list(w.index) == list(SECTORS)
    # L10 = L,M,N with jobs 100/200/300 and wages 20/21/22
    assert w["L10"] == pytest.approx((100 * 20 + 200 * 21 + 300 * 22) / 600)
    assert w["L01"] == 20.0


def test_sector_wages_missing_section_errors():
    raw = _wage_raw()
    raw = raw[~raw["BedrijfstakkenBranchesSBI2008"].str.startswith("389100")]
    with pytest.raises(KeyError, match="389100"):
        sector_wages(raw)


# ── Covariates ───────────────────────────────────────────────────────

def _edu():
    return pd.DataFrame({
        "BU_CODE": ["BU1", "BU2", "BU3"],
        "Praktisch": [0.2, 0.3, 0.1], "Middelbaar": [0.5, 0.4, 0.3],
        "Hoger": [0.3, 0.3, 0.6],
        "Aantal_Middelbaar": [50, 40, 30], "Aantal_Onbekend": [0, 0, 0],
        "Aantal_Praktisch": [20, 30, 10], "Aantal_Theoretisch": [30, 30, 60]})


def test_education_shares_and_buurt_covariates():
    edu = parse_education_shares(_edu())
    assert edu.loc["BU1", "edu_jobs"] == 100
    buurten = pd.DataFrame({"buurtcode": ["BU1", "BU2", "BU9"],
                            "stedelijkheid": [1.0, 5.0, 3.0],
                            "gem_woz": [200.0, np.nan, -1.0]})
    cov = buurt_covariates(buurten, edu)
    assert list(cov.columns) == list(COVARIATES)
    assert cov.loc["BU1", "ln_woz"] == pytest.approx(np.log(200))
    assert np.isnan(cov.loc["BU2", "ln_woz"]) and np.isnan(cov.loc["BU9", "ln_woz"])
    assert np.isnan(cov.loc["BU9", "edu_hoger"])


def test_municipal_covariates_weighted_mean_ignores_missing():
    cov = pd.DataFrame({"a": [1.0, 3.0, np.nan]},
                       index=["b1", "b2", "b3"])
    weights = pd.Series([1.0, 3.0, 100.0], index=cov.index)
    gem = pd.Series(["G", "G", "G"], index=cov.index)
    out = municipal_covariates(cov, weights, gem)
    assert out.loc["G", "a"] == pytest.approx((1 * 1 + 3 * 3) / 4)


# ── Model ────────────────────────────────────────────────────────────

def _synthetic_municipalities(n=80, seed=0):
    rng = np.random.default_rng(seed)
    x = pd.DataFrame({c: rng.uniform(0, 1, n) for c in COVARIATES},
                     index=[f"G{i}" for i in range(n)])
    alpha = rng.normal(0, 0.3, len(SECTORS))
    beta = rng.normal(0, 1.2, (len(SECTORS), len(COVARIATES)))
    eta = alpha + x.to_numpy() @ beta.T
    share = np.exp(eta) / np.exp(eta).sum(axis=1, keepdims=True)
    total = rng.uniform(2e4, 3e5, n)
    jobs = pd.DataFrame(share * total[:, None], index=x.index,
                        columns=list(SECTORS))
    return x, jobs, share


def test_sector_model_recovers_composition():
    x, jobs, share = _synthetic_municipalities()
    model = fit_sector_model(jobs, x)
    pred = model.shares(x.to_numpy())
    np.testing.assert_allclose(pred.sum(axis=1), 1.0)
    tv = 0.5 * np.abs(pred - share).sum(axis=1).mean()
    assert tv < 0.03
    # better than ignoring the covariates
    null = np.tile(jobs.sum() / jobs.to_numpy().sum(), (len(x), 1))
    assert tv < 0.5 * np.abs(null - share).sum(axis=1).mean()


def test_sector_model_clips_to_fitted_support():
    x, jobs, _ = _synthetic_municipalities()
    model = fit_sector_model(jobs, x)
    far = np.full((1, len(COVARIATES)), 50.0)
    edge = np.tile(model.upper, (1, 1))
    np.testing.assert_allclose(model.shares(far), model.shares(edge))
    unclipped = SectorModel(model.alpha, model.beta, model.covariates,
                            model.sectors)
    assert not np.allclose(unclipped.shares(far), model.shares(far))


def test_sector_model_needs_enough_municipalities():
    x, jobs, _ = _synthetic_municipalities(n=10)
    with pytest.raises(ValueError, match="at least 20"):
        fit_sector_model(jobs, x)


# ── Imputation ───────────────────────────────────────────────────────

def _impute_setup():
    x, jobs, _ = _synthetic_municipalities()
    model = fit_sector_model(jobs, x)
    lisa = pd.DataFrame(
        {s: [1000.0 * (k + 1), 500.0 * (k + 1)] for k, s in enumerate(SECTORS)},
        index=["Alpha", "Beta"])
    codes = ["A1", "A2", "A3", "B1", "B2", "X1"]
    gem = pd.Series(["Alpha"] * 3 + ["Beta"] * 2 + [np.nan], index=codes)
    buurt_jobs = pd.Series([10.0, 30.0, 60.0, 1.0, 1.0, 5.0], index=codes)
    rng = np.random.default_rng(1)
    cov = pd.DataFrame(rng.uniform(0, 1, (6, len(COVARIATES))),
                       index=codes, columns=list(COVARIATES))
    return model, lisa, gem, buurt_jobs, cov


def test_imputation_reproduces_both_marginals():
    model, lisa, gem, buurt_jobs, cov = _impute_setup()
    res = impute_sector_jobs(buurt_jobs, gem, lisa, model, cov)
    J = res.jobs
    assert "X1" not in J.index                       # uncovered buurt
    for name, members in (("Alpha", ["A1", "A2", "A3"]),
                          ("Beta", ["B1", "B2"])):
        np.testing.assert_allclose(J.loc[members].sum(), lisa.loc[name],
                                   atol=1e-5)
        np.testing.assert_allclose(
            J.loc[members].sum(axis=1).to_numpy(),
            buurt_jobs[members].to_numpy() / buurt_jobs[members].sum()
            * lisa.loc[name].sum(), atol=1e-5)
    assert (J.to_numpy() >= 0).all()
    assert res.report["not_converged"] == 0
    assert res.report["total_jobs_imputed"] == pytest.approx(
        res.report["total_jobs_lisa"])


def test_single_buurt_municipality_gets_the_whole_vector():
    model, lisa, gem, buurt_jobs, cov = _impute_setup()
    gem = gem.copy()
    gem["A1"], gem["A2"], gem["A3"] = "Alpha", np.nan, np.nan
    res = impute_sector_jobs(buurt_jobs, gem, lisa, model, cov)
    np.testing.assert_allclose(res.jobs.loc["A1"], lisa.loc["Alpha"], atol=1e-6)


def test_municipality_without_buurt_jobs_splits_equally():
    model, lisa, gem, buurt_jobs, cov = _impute_setup()
    buurt_jobs = buurt_jobs.copy()
    buurt_jobs[["B1", "B2"]] = 0.0
    res = impute_sector_jobs(buurt_jobs, gem, lisa, model, cov)
    np.testing.assert_allclose(res.jobs.loc["B1"].sum(),
                               res.jobs.loc["B2"].sum())
    assert res.report["no_buurt_jobs_equal_split"] == 1


def test_lisa_municipality_without_buurten_is_reported():
    model, lisa, gem, buurt_jobs, cov = _impute_setup()
    lisa = pd.concat([lisa, lisa.rename(index={"Alpha": "Gamma",
                                               "Beta": "Delta"})])
    res = impute_sector_jobs(buurt_jobs, gem, lisa, model, cov)
    assert set(res.report["lisa_without_buurten"]) == {"Gamma", "Delta"}


def test_covariates_steer_the_within_municipality_split():
    model, lisa, gem, buurt_jobs, cov = _impute_setup()
    equal_jobs = pd.Series(1.0, index=buurt_jobs.index)
    cov = cov.copy()
    cov.loc["A1"] = model.upper
    cov.loc["A2"] = model.lower
    res = impute_sector_jobs(equal_jobs, gem, lisa, model, cov)
    shares = res.jobs.loc[["A1", "A2"]].div(
        res.jobs.loc[["A1", "A2"]].sum(axis=1), axis=0)
    expected = model.shares(np.vstack([model.upper, model.lower]))
    # the seed's ordering of sectors between the two buurten survives
    k = int(np.argmax(np.abs(expected[0] - expected[1])))
    assert np.sign(shares.iloc[0, k] - shares.iloc[1, k]) \
        == np.sign(expected[0, k] - expected[1, k])


def test_missing_covariates_fall_back_to_municipal_mean():
    model, lisa, gem, buurt_jobs, cov = _impute_setup()
    cov = cov.copy()
    cov.loc["A1"] = np.nan
    res = impute_sector_jobs(buurt_jobs, gem, lisa, model, cov)
    assert np.isfinite(res.jobs.to_numpy()).all()


# ── Sector -> income weights and pools ───────────────────────────────

def test_income_weights_hand_example():
    wage = pd.Series({"s1": 10.0, "s2": 20.0, "s3": 30.0})
    jobs = pd.Series({"s1": 20.0, "s2": 50.0, "s3": 30.0})
    W = sector_income_weights(wage, jobs)
    # s1 covers rank [0, .2]: D1, D2 half each; s2 [.2, .7]: D3..D7 0.2 each
    assert W.loc["D1", "s1"] == pytest.approx(0.5)
    assert W.loc["D2", "s1"] == pytest.approx(0.5)
    for k in range(3, 8):
        assert W.loc[f"D{k}", "s2"] == pytest.approx(0.2)
    assert W.loc["D8", "s3"] == pytest.approx(1 / 3)
    assert (W.loc["onbekend"] == 1.0).all()
    ranked = W.drop(index="onbekend")
    np.testing.assert_allclose(ranked.sum(axis=0), 1.0)      # partition


def test_income_weights_order_follows_wage_not_input_order():
    wage = pd.Series({"a": 30.0, "b": 10.0})
    jobs = pd.Series({"a": 50.0, "b": 50.0})
    W = sector_income_weights(wage, jobs)
    assert W.loc["D1", "b"] == pytest.approx(0.2) and W.loc["D1", "a"] == 0
    assert W.loc["D10", "a"] == pytest.approx(0.2)


def test_income_weights_validation():
    wage = pd.Series({"a": 1.0, "b": 2.0})
    with pytest.raises(ValueError, match="sum to a positive"):
        sector_income_weights(wage, pd.Series({"a": 0.0, "b": 0.0}))
    with pytest.raises(ValueError, match="wages are incomplete"):
        sector_income_weights(pd.Series({"a": 1.0, "b": np.nan}),
                              pd.Series({"a": 1.0, "b": 1.0}))


def test_sector_pools_partition_the_jobs_and_align_zones():
    wage = pd.Series({"a": 1.0, "b": 2.0, "c": 3.0})
    national = pd.Series({"a": 30.0, "b": 30.0, "c": 40.0})
    W = sector_income_weights(wage, national)
    J = pd.DataFrame({"a": [10.0, 5.0], "b": [0.0, 20.0], "c": [3.0, 7.0]},
                     index=["Z1", "Z2"])
    pools = sector_pools(J, ["Z2", "Z9", "Z1"], W)
    assert set(pools) == set(INCOME_CLASSES)
    total = sum(pools[c] for c in INCOME_CLASSES if c != "onbekend")
    np.testing.assert_allclose(total, [32.0, 0.0, 13.0], rtol=1e-6)
    np.testing.assert_allclose(pools["onbekend"], [32.0, 0.0, 13.0], rtol=1e-6)
    with pytest.raises(KeyError, match="lack column"):
        sector_pools(J.drop(columns="c"), ["Z1"], W)
