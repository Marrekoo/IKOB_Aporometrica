"""Legacy job table and quantile-matched job pools."""

import os

import numpy as np
import pandas as pd
import pytest

from ikob2.core.numerics import DTYPE
from ikob2.domain.filter_config import CurveSpec
from ikob2.domain.state import ModelState
from ikob2.engine.runner import SegmentedRunner
from ikob2.segments.bridge import build_segments
from ikob2.segments.config import INCOME_CLASSES
from ikob2.segments.jobs import (
    JOB_GROUPS,
    job_pools,
    load_legacy_jobs,
    parse_legacy_jobs,
    quantile_weights,
)

CLASSES = list(INCOME_CLASSES)


# ── Quantile weights ─────────────────────────────────────────────────

def test_aligned_quantiles_map_deciles_to_single_groups():
    # cumulative shares 0.2, 0.5, 0.8, 1.0 fall on decile boundaries
    W = quantile_weights([0.2, 0.3, 0.3, 0.2])
    expected = {"D1": "laag", "D2": "laag", "D3": "middellaag",
                "D4": "middellaag", "D5": "middellaag", "D6": "middelhoog",
                "D7": "middelhoog", "D8": "middelhoog", "D9": "hoog",
                "D10": "hoog"}
    for cls, group in expected.items():
        assert W.loc[cls, group] == pytest.approx(1.0), cls
        assert W.loc[cls].sum() == pytest.approx(1.0)


def test_boundary_straddling_decile_is_split():
    W = quantile_weights([0.25, 0.25, 0.25, 0.25])
    # D3 covers rank [0.2, 0.3]; laag ends at 0.25 -> half and half
    assert W.loc["D3", "laag"] == pytest.approx(0.5)
    assert W.loc["D3", "middellaag"] == pytest.approx(0.5)
    assert W.loc["D5", "middellaag"] == pytest.approx(1.0)


def test_weights_shape_rows_and_monotonicity():
    W = quantile_weights([1.685, 2.303, 2.375, 2.294])      # legacy totals
    assert list(W.index) == CLASSES and list(W.columns) == list(JOB_GROUPS)
    ranked = W.loc[[c for c in CLASSES if c != "onbekend"]]
    np.testing.assert_allclose(ranked.sum(axis=1), 1.0)
    assert (W.loc["onbekend"] == 1.0).all()
    mean_group = (ranked.to_numpy() * np.arange(4)).sum(axis=1)
    assert np.all(np.diff(mean_group) >= -1e-12)     # richer -> higher jobs


def test_weights_proportions_only_and_validation():
    a = quantile_weights([1, 2, 3, 4])
    b = quantile_weights([10, 20, 30, 40])
    pd.testing.assert_frame_equal(a, b)
    for bad in ([1, 2, 3], [0, 0, 0, 0], [1, -1, 2, 2], [1, np.nan, 1, 1]):
        with pytest.raises(ValueError):
            quantile_weights(bad)


def test_empty_group_gets_zero_weight():
    W = quantile_weights([0.5, 0.0, 0.0, 0.5])
    ranked = W.drop(index="onbekend")
    assert (ranked["middellaag"] == 0).all()
    assert (ranked["middelhoog"] == 0).all()
    assert W.loc["D1", "laag"] == 1.0 and W.loc["D10", "hoog"] == 1.0


# ── Legacy table parsing ─────────────────────────────────────────────

def _raw(year="2018"):
    cols = ["Rijlabels"] + [f"Som van arb_{year}_{g}" for g in JOB_GROUPS] \
        + [f"Som van arb_2030H_{g}" for g in JOB_GROUPS]
    rows = [["BU00010001", 10, 20, 30, 40, 1, 1, 1, 1],
            ["BU00010002", 5, 5, 5, 5, 2, 2, 2, 2],
            ["Eindtotaal", 15, 25, 35, 45, 3, 3, 3, 3]]
    return pd.DataFrame(rows, columns=cols)


def test_parse_selects_year_and_drops_totals():
    jobs = parse_legacy_jobs(_raw())
    assert list(jobs.index) == ["BU00010001", "BU00010002"]
    assert jobs.loc["BU00010001"].tolist() == [10, 20, 30, 40]
    assert parse_legacy_jobs(_raw(), "2030H").loc["BU00010002"].tolist() == [2] * 4


def test_parse_errors():
    with pytest.raises(KeyError, match="arb_2040L_laag"):
        parse_legacy_jobs(_raw(), "2040L")
    dup = pd.concat([_raw(), _raw().iloc[:1]])
    with pytest.raises(ValueError, match="duplicate"):
        parse_legacy_jobs(dup)
    neg = _raw()
    neg.iloc[0, 1] = -1
    with pytest.raises(ValueError, match="negative"):
        parse_legacy_jobs(neg)


# ── Pools ────────────────────────────────────────────────────────────

def test_job_pools_mix_groups_and_align_zones():
    jobs = parse_legacy_jobs(_raw())
    W = quantile_weights([0.25] * 4)
    pools = job_pools(jobs, ["BU00010002", "BU9", "BU00010001"], W)
    assert set(pools) == set(CLASSES)
    # D1 sees only laag jobs; zone order and the unknown zone are respected
    np.testing.assert_allclose(pools["D1"], [5, 0, 10])
    np.testing.assert_allclose(pools["D3"], [5, 0, 0.5 * 10 + 0.5 * 20])
    np.testing.assert_allclose(pools["onbekend"], [20, 0, 100])
    assert pools["D1"].dtype == np.float32


def test_default_weights_come_from_the_table_totals():
    jobs = parse_legacy_jobs(_raw())
    default = job_pools(jobs, list(jobs.index))
    explicit = job_pools(jobs, list(jobs.index),
                         quantile_weights(jobs.sum().to_numpy()))
    for cls in CLASSES:
        np.testing.assert_array_equal(default[cls], explicit[cls])


def test_job_pools_validation():
    jobs = parse_legacy_jobs(_raw())
    with pytest.raises(ValueError, match="duplicates"):
        job_pools(jobs, ["BU00010001", "BU00010001"])
    with pytest.raises(KeyError, match="lacks group"):
        job_pools(jobs.drop(columns="hoog"), ["BU00010001"])


def test_pools_drive_hansen_with_income_matched_jobs():
    n = 3
    rng = np.random.default_rng(0)
    time = rng.uniform(5, 40, (n, n)).astype(DTYPE)
    np.fill_diagonal(time, 0.0)
    codes = ["BU00010001", "BU00010002", "BU00010003"]
    jobs = pd.DataFrame(
        {"laag": [10., 0., 5.], "middellaag": [0., 10., 5.],
         "middelhoog": [5., 5., 0.], "hoog": [0., 0., 20.]}, index=codes)
    pools = job_pools(jobs, codes, quantile_weights([1, 1, 1, 1]))
    state = ModelState.create(
        generalized_cost=time, population=np.ones(n),
        opportunities=jobs.sum(axis=1).to_numpy(), decay_type="exponential",
        decay_beta=0.05, decay_epsilon=0.0)
    segs = build_segments(CurveSpec("weibull", (2.0, 45.0)),
                          pool_by="income_class",
                          only=["single_D1", "single_D10"])
    out = SegmentedRunner(decay_epsilon=None).run_hansen(
        state, segs, cost_matrices={"time": time}, opportunities=pools)
    w = np.exp(-(time / 45.0) ** 2)
    np.testing.assert_allclose(out["single_D1"], w @ jobs["laag"], rtol=1e-4)
    np.testing.assert_allclose(out["single_D10"], w @ jobs["hoog"], rtol=1e-4)


# ── The real legacy file (optional) ──────────────────────────────────

LEGACY = os.environ.get("IKOB_LEGACY_ALLZONES", "")


@pytest.mark.skipif(not os.path.exists(LEGACY),
                    reason="set IKOB_LEGACY_ALLZONES to Alle_Zones_2030_2040.xlsx")
def test_real_legacy_jobs_shape_and_totals():
    jobs = load_legacy_jobs(LEGACY, "2018")
    assert len(jobs) == 14327 and jobs.notna().all().all()
    assert jobs.sum().sum() == pytest.approx(8_657_025, rel=1e-3)
    W = quantile_weights(jobs.sum().to_numpy())
    assert W.loc["D1", "laag"] > 0.99 and W.loc["D10", "hoog"] > 0.99
