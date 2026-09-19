"""Establishment counts as information for the sector-job imputation."""

import numpy as np
import pandas as pd
import pytest

from ikob2.segments.establishments import (
    GROUPS,
    complete_group_counts,
    read_establishments,
)
from ikob2.segments.jobs_impute import (
    COVARIATES,
    fit_sector_model,
    impute_sector_jobs,
    within_municipality_tv,
)
from ikob2.segments.lisa import SECTOR_TO_KWB_GROUP, SECTORS

from test_segment_lisa import _synthetic_municipalities


def test_sector_to_group_mapping_is_complete():
    assert set(SECTOR_TO_KWB_GROUP) == set(SECTORS)
    assert set(SECTOR_TO_KWB_GROUP.values()) == set(GROUPS)


# ── Reading ──────────────────────────────────────────────────────────

def _est_frame(codes, **cols):
    df = pd.DataFrame({"buurtcode": codes,
                       "gemeentenaam": "x", "total": cols.pop("total"),
                       **{g: cols.get(g, 0.0) for g in GROUPS}})
    return df


def test_read_establishments(tmp_path):
    p = tmp_path / "e.csv"
    _est_frame([" BU1 ", "BU2"], total=[10, 20]).to_csv(p, index=False)
    est = read_establishments(p)
    assert list(est.index) == ["BU1", "BU2"]
    assert list(est.columns) == ["total", *GROUPS]
    with pytest.raises(FileNotFoundError, match="fetch"):
        read_establishments(tmp_path / "missing.csv")
    _est_frame(["BU1"], total=[1]).drop(columns="O-Q").to_csv(p, index=False)
    with pytest.raises(KeyError, match="O-Q"):
        read_establishments(p)


# ── Completing suppressed cells ──────────────────────────────────────

def _est(rows):
    return pd.DataFrame(rows, columns=["total", *GROUPS],
                        index=[f"B{i}" for i in range(len(rows))])


def test_suppressed_groups_are_filled_from_municipal_composition():
    nan = np.nan
    est = _est([
        [10, 5, 5] + [0] * 6,                # fully observed: 50% A, 50% B-F
        [20, 10, 10] + [0] * 6,              # fully observed
        [8] + [nan] * 8,                     # suppressed groups, total known
    ])
    gem = pd.Series(["G"] * 3, index=est.index)
    out = complete_group_counts(est, gem)
    assert out.loc["B0", "A"] == 5 and out.loc["B1", "B-F"] == 10   # untouched
    assert out.loc["B2", "A"] == pytest.approx(4.0)
    assert out.loc["B2", "B-F"] == pytest.approx(4.0)
    assert out.loc["B2", list(GROUPS)].sum() == pytest.approx(8.0)


def test_only_suppressed_cells_are_filled_and_national_fallback_used():
    nan = np.nan
    est = _est([[10, 4, nan, 0, 0, 0, 0, 0, 0],
                [10, 5, 5] + [0] * 6])
    gem = pd.Series(["G", "H"], index=est.index)     # G has no observed buurt
    out = complete_group_counts(est, gem)
    assert out.loc["B0", "A"] == 4                   # observed cell kept
    assert out.loc["B0", "B-F"] > 0                  # national composition
    assert np.isfinite(out.to_numpy()).all()


def test_buurt_without_row_uses_job_based_fallback_and_zero_stays_zero():
    est = _est([[10, 10] + [0] * 7, [30, 30] + [0] * 7])
    idx = ["B0", "B1", "B2", "B3"]
    gem = pd.Series(["G"] * 4, index=idx)
    jobs = pd.Series([100.0, 300.0, 200.0, 0.0], index=idx)
    out = complete_group_counts(est, gem, jobs)
    # 40 establishments per 400 jobs in G -> B2 (200 jobs) gets ~20
    assert out.loc["B2"].sum() == pytest.approx(20.0)
    assert out.loc["B3"].sum() == 0.0
    no_fallback = complete_group_counts(est, gem)
    assert no_fallback.loc["B2"].sum() == 0.0


# ── Imputation with establishments ───────────────────────────────────

def _setup():
    x, jobs, _ = _synthetic_municipalities()
    model = fit_sector_model(jobs, x)
    lisa = pd.DataFrame(
        {s: [1000.0 * (k + 1), 800.0] for k, s in enumerate(SECTORS)},
        index=["Alpha", "Beta"])
    codes = ["A1", "A2", "A3", "B1", "B2"]
    gem = pd.Series(["Alpha"] * 3 + ["Beta"] * 2, index=codes)
    legacy = pd.Series([10.0, 30.0, 60.0, 50.0, 50.0], index=codes)
    cov = pd.DataFrame(np.tile(model.lower * 0.5 + model.upper * 0.5,
                               (5, 1)), index=codes,
                       columns=list(COVARIATES))     # identical buurten
    est = pd.DataFrame(0.0, index=codes, columns=["total", *GROUPS])
    est["total"] = 10.0
    est["G+I"] = 10.0
    return model, lisa, gem, legacy, cov, est


def test_marginals_stay_exact_with_establishments():
    model, lisa, gem, legacy, cov, est = _setup()
    res = impute_sector_jobs(legacy, gem, lisa, model, cov,
                             establishments=est)
    np.testing.assert_allclose(res.jobs.loc[["A1", "A2", "A3"]].sum(),
                               lisa.loc["Alpha"], atol=1e-5)
    np.testing.assert_allclose(res.jobs.loc[["B1", "B2"]].sum(),
                               lisa.loc["Beta"], atol=1e-5)
    assert res.report["not_converged"] == 0


def test_establishment_weight_controls_the_buurt_totals():
    model, lisa, gem, legacy, cov, est = _setup()
    est.loc[["A1", "A2", "A3"], "total"] = [60.0, 30.0, 10.0]
    est.loc[["A1", "A2", "A3"], "G+I"] = [60.0, 30.0, 10.0]   # groups add up
    members = ["A1", "A2", "A3"]
    r0 = impute_sector_jobs(legacy, gem, lisa, model, cov, establishments=est,
                            establishment_weight=0.0).jobs.loc[members]
    np.testing.assert_allclose(
        r0.sum(axis=1) / r0.to_numpy().sum(), [0.1, 0.3, 0.6], atol=1e-6)
    r1 = impute_sector_jobs(legacy, gem, lisa, model, cov, establishments=est,
                            establishment_weight=1.0).jobs.loc[members]
    np.testing.assert_allclose(
        r1.sum(axis=1) / r1.to_numpy().sum(), [0.6, 0.3, 0.1], atol=1e-6)
    half = impute_sector_jobs(legacy, gem, lisa, model, cov,
                              establishments=est,
                              establishment_weight=0.5).jobs.loc[members]
    np.testing.assert_allclose(
        half.sum(axis=1) / half.to_numpy().sum(), [0.35, 0.3, 0.35],
        atol=1e-6)
    with pytest.raises(ValueError, match="establishment_weight"):
        impute_sector_jobs(legacy, gem, lisa, model, cov, establishments=est,
                           establishment_weight=1.5)


def test_group_establishments_steer_sector_placement():
    model, lisa, gem, legacy, cov, est = _setup()
    est = est.copy()
    est.loc["A1", ["G+I", "O-Q"]] = [0.0, 40.0]      # only O-Q here
    est.loc["A2", ["G+I", "O-Q"]] = [40.0, 0.0]      # only G+I here
    est.loc["A3", ["G+I", "O-Q"]] = [0.0, 0.0]
    equal = pd.Series(1.0, index=legacy.index)
    res = impute_sector_jobs(equal, gem, lisa, model, cov,
                             establishments=est, establishment_weight=0.0)
    J = res.jobs
    care = ["L11", "L12", "L13"]
    retail = ["L05", "L07"]
    assert J.loc["A1", care].sum() > J.loc["A2", care].sum()
    assert J.loc["A2", retail].sum() > J.loc["A1", retail].sum()


def test_buurt_without_establishment_row_still_gets_jobs():
    model, lisa, gem, legacy, cov, est = _setup()
    est = est.drop(index="A3")
    res = impute_sector_jobs(legacy, gem, lisa, model, cov, establishments=est)
    assert res.jobs.loc["A3"].sum() > 0
    assert np.isfinite(res.jobs.to_numpy()).all()


def test_without_establishments_behaviour_is_unchanged():
    model, lisa, gem, legacy, cov, est = _setup()
    a = impute_sector_jobs(legacy, gem, lisa, model, cov)
    b = impute_sector_jobs(legacy, gem, lisa, model, cov, establishments=None)
    pd.testing.assert_frame_equal(a.jobs, b.jobs)


# ── Validation metric ────────────────────────────────────────────────

def test_within_municipality_tv():
    idx = ["a", "b", "c", "d", "e", "f"]
    gem = pd.Series(["G"] * 3 + ["H"] * 3, index=idx)
    truth = pd.Series([1, 1, 2, 5, 5, 0.0], index=idx) + 1e-12
    assert within_municipality_tv(truth, truth, gem) == pytest.approx(0.0)
    uniform = pd.Series(1.0, index=idx)
    # G: shares (.25,.25,.5) vs uniform -> TV .1667; H: (.5,.5,0) -> TV .3333
    tv = within_municipality_tv(uniform, truth, gem)
    weights = np.array([4.0, 10.0])
    assert tv == pytest.approx(np.average([1 / 6, 1 / 3], weights=weights))
    with pytest.raises(ValueError, match="No municipality"):
        within_municipality_tv(uniform, truth, gem, min_buurten=4)
