"""Weibull time margins from the fitted table."""

import math

import numpy as np
import pandas as pd
import pytest

from ikob2.core import families as fam
from ikob2.core.numerics import DTYPE
from ikob2.engine.runner import evaluate_marginal
from ikob2.segments.bridge import build_segments
from ikob2.segments.time_margins import (
    MODE_NAMES,
    WFH_NAMES,
    load_time_margins,
    time_margin,
)

TABLE = "data/margins/S_T_work.csv"


def test_repo_table_loads_all_six_cells():
    m = load_time_margins(TABLE)
    assert sorted(m) == sorted((mode, wfh) for mode in ("bike", "pt", "car")
                               for wfh in ("no_wfh", "wfh_possible"))
    k, eta = m[("car", "no_wfh")].params
    assert (k, eta) == pytest.approx((3.15426647164561, 44.8183592615287))
    assert m[("bike", "no_wfh")].curve == "weibull"


def test_medians_and_hazard_class_follow_from_the_parameters():
    m = load_time_margins(TABLE)
    for r in pd.read_csv(TABLE).itertuples(index=False):
        key = (MODE_NAMES[r.mode.lower()], WFH_NAMES[r.wfh.lower()])
        med = fam.survival("weibull", m[key].params, np.array([r.median]))
        assert med[0] == pytest.approx(0.5, abs=1e-6)
        assert fam.hazard_shape("weibull", m[key].params) == "increasing"


def test_working_from_home_widens_acceptable_times_for_every_mode():
    m = load_time_margins(TABLE)
    for mode in ("bike", "pt", "car"):
        assert fam.mean_threshold("weibull", m[(mode, "wfh_possible")].params) \
            > fam.mean_threshold("weibull", m[(mode, "no_wfh")].params)


def test_acceptance_at_a_travel_time_is_plausible():
    m = load_time_margins(TABLE)
    car = m[("car", "no_wfh")].params
    s30, s60 = fam.survival("weibull", car, np.array([30.0, 60.0]))
    # the paper's Figure 3: about 75% of professionals accept 30 minutes
    # by car (no home working) and almost none 60
    assert 0.70 < s30 < 0.80 and s60 < 0.1


def test_margin_drives_segments_and_the_marginal():
    m = load_time_margins(TABLE)
    spec = time_margin(m, "pt", "wfh_possible")
    segs = build_segments(spec, only=["single_D5"])
    assert segs[0].class_filter.time == spec
    t = np.array([[0.0, 51.9643233189623]], dtype=DTYPE)      # the median
    out = evaluate_marginal(t, spec)
    assert out[0, 0] == 1.0 and out[0, 1] == pytest.approx(0.5, abs=1e-4)
    with pytest.raises(KeyError, match="No time margin"):
        time_margin(m, "walk")


def _write(tmp_path, **override):
    row = {"wfh": "No home working", "mode": "Car", "eta": 44.8, "k": 3.15,
           "median": 44.8 * math.log(2) ** (1 / 3.15), "class": "IFR"}
    row.update(override)
    p = tmp_path / "t.csv"
    pd.DataFrame([row]).to_csv(p, index=False)
    return p


def test_validation(tmp_path):
    assert load_time_margins(_write(tmp_path))
    with pytest.raises(ValueError, match="median"):
        load_time_margins(_write(tmp_path, median=30.0))
    with pytest.raises(ValueError, match="class"):
        load_time_margins(_write(tmp_path, **{"class": "DFR"}))
    with pytest.raises(ValueError, match="positive"):
        load_time_margins(_write(tmp_path, eta=-1.0))
    with pytest.raises(ValueError, match="Unknown"):
        load_time_margins(_write(tmp_path, mode="Boat"))
    p = _write(tmp_path)
    df = pd.read_csv(p)
    pd.concat([df, df]).to_csv(p, index=False)
    with pytest.raises(ValueError, match="Duplicate"):
        load_time_margins(p)
    df.drop(columns="k").to_csv(p, index=False)
    with pytest.raises(KeyError, match="missing column"):
        load_time_margins(p)
