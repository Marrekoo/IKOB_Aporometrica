"""IKOB job table (buurt totals) and its use with sector pools."""

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
    load_ikob_jobs,
    parse_ikob_jobs,
    sector_income_weights,
    sector_pools,
)

CLASSES = list(INCOME_CLASSES)


# ── IKOB table parsing ─────────────────────────────────────────────

def _raw(year="2018"):
    cols = ["Rijlabels"] + [f"Som van arb_{year}_{g}" for g in JOB_GROUPS] \
        + [f"Som van arb_2030H_{g}" for g in JOB_GROUPS]
    rows = [["BU00010001", 10, 20, 30, 40, 1, 1, 1, 1],
            ["BU00010002", 5, 5, 5, 5, 2, 2, 2, 2],
            ["Eindtotaal", 15, 25, 35, 45, 3, 3, 3, 3]]
    return pd.DataFrame(rows, columns=cols)


def test_parse_selects_year_and_drops_totals():
    jobs = parse_ikob_jobs(_raw())
    assert list(jobs.index) == ["BU00010001", "BU00010002"]
    assert jobs.loc["BU00010001"].tolist() == [10, 20, 30, 40]
    assert parse_ikob_jobs(_raw(), "2030H").loc["BU00010002"].tolist() == [2] * 4


def test_parse_errors():
    with pytest.raises(KeyError, match="arb_2040L_laag"):
        parse_ikob_jobs(_raw(), "2040L")
    dup = pd.concat([_raw(), _raw().iloc[:1]])
    with pytest.raises(ValueError, match="duplicate"):
        parse_ikob_jobs(dup)
    neg = _raw()
    neg.iloc[0, 1] = -1
    with pytest.raises(ValueError, match="negative"):
        parse_ikob_jobs(neg)


# ── IKOB table drives the sector-pool engine path ──────────────────

def test_pools_drive_hansen_with_income_matched_jobs():
    n = 3
    rng = np.random.default_rng(0)
    time = rng.uniform(5, 40, (n, n)).astype(DTYPE)
    np.fill_diagonal(time, 0.0)
    codes = ["BU00010001", "BU00010002", "BU00010003"]
    sectors = pd.DataFrame(
        {"cheap": [10., 0., 5.], "mid": [0., 10., 5.], "rich": [5., 5., 20.]},
        index=codes)
    wage = pd.Series({"cheap": 10.0, "mid": 20.0, "rich": 30.0})
    W = sector_income_weights(wage, sectors.sum())
    pools = sector_pools(sectors, codes, W)
    state = ModelState.create(
        generalized_cost=time, population=np.ones(n),
        opportunities=sectors.sum(axis=1).to_numpy(), decay_type="exponential",
        decay_beta=0.05, decay_epsilon=0.0)
    segs = build_segments(CurveSpec("weibull", (2.0, 45.0)),
                          pool_by="income_class",
                          only=["single_D1", "single_D10"])
    out = SegmentedRunner(decay_epsilon=None).run_hansen(
        state, segs, cost_matrices={"time": time}, opportunities=pools)
    w = np.exp(-(time / 45.0) ** 2)
    # national shares: cheap 0.25, mid 0.25... D1 sits inside the cheapest
    # sector, D10 inside the richest
    np.testing.assert_allclose(out["single_D1"], w @ pools["D1"], rtol=1e-4)
    assert pools["D1"].sum() < sectors["cheap"].sum() + 1e-6
    assert pools["D10"].sum() <= sectors["rich"].sum() + 1e-6
    assert not np.allclose(out["single_D1"], out["single_D10"])


# ── The real IKOB file (optional) ──────────────────────────────────

IKOB_FILE = os.environ.get("IKOB_JOBS_ALLZONES", "")


@pytest.mark.skipif(not os.path.exists(IKOB_FILE),
                    reason="set IKOB_JOBS_ALLZONES to Alle_Zones_2030_2040.xlsx")
def test_real_ikob_jobs_shape_and_totals():
    jobs = load_ikob_jobs(IKOB_FILE, "2018")
    assert len(jobs) == 14327 and jobs.notna().all().all()
    assert jobs.sum().sum() == pytest.approx(8_657_025, rel=1e-3)
