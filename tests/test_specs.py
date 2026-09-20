import numpy as np
import pandas as pd
import pytest

from ikob2.core.families import mean_threshold
from ikob2.domain.filter_config import CurveSpec
from ikob2.segments import specs
from ikob2.segments.bridge import build_segments

W = CurveSpec("weibull", (3.0, 45.0))


def row(low, high, atom=0.0):
    return pd.Series({"low": low, "high": high, "atom": atom}).pipe(
        lambda s: type("R", (), s.to_dict())())


def test_exponential_time_has_weibull_mean():
    e = specs.exponential_time(W)
    assert e.curve == "exponential"
    assert 1.0 / e.params[0] == pytest.approx(mean_threshold("weibull", W.params))


def test_mean_cost_uniform_and_atom():
    assert specs.mean_cost(row(4.0, 24.0)) == pytest.approx(14.0)
    assert specs.mean_cost(row(4.0, 24.0, 0.5)) == pytest.approx(7.0)
    assert specs.mean_cost(row(np.nan, np.nan, 1.0)) == 0.0


def test_m1_shared_cost_curve_uses_vot_and_beta():
    f = specs.cost_curve_factory("m1", W, 12.0)
    beta = mean_threshold("weibull", W.params)
    assert 1.0 / f(row(1, 2)).params[0] == pytest.approx(beta * 12.0 / 60.0)
    with pytest.raises(ValueError):
        specs.cost_curve_factory("m1", W, None)


def test_m1p_matches_mean_envelope_and_m2_keeps_uniform():
    f = specs.cost_curve_factory("m1p", W, None)
    assert 1.0 / f(row(4.0, 24.0)).params[0] == pytest.approx(14.0)
    assert specs.cost_curve_factory("m2", W, None) is None
    assert specs.cost_curve_factory("m3", W, None) is None


def test_spec_copula():
    assert specs.spec_copula("m2", 2.0).family == "independence"
    assert specs.spec_copula("m3", 2.0).family == "gumbel"
    assert specs.spec_copula("m3", float("inf")).family == "comonotone"
    with pytest.raises(ValueError):
        specs.spec_copula("m3", None)


def test_build_segments_uses_cost_curve_override():
    env = pd.DataFrame({"household_type": ["single"] * 2,
                        "income_class": ["D2", "D3"],
                        "low": [4.0, 8.0], "high": [24.0, 70.0],
                        "atom": [0.0, 0.0]})
    segs = build_segments(specs.exponential_time(W), envelope=env,
                          money_cost_id="c", household_types=["single"],
                          income_classes=["D2", "D3"],
                          cost_curve=specs.cost_curve_factory("m1p", W, None))
    means = [1.0 / s.class_filter.cost.params[0] for s in segs]
    assert means == pytest.approx([14.0, 39.0])


def test_vot_weighted_cost_extremes_and_generalised_cost_identity():
    c = np.array([[10.0, 10.0, 10.0]], dtype=np.float32)
    share = np.array([[1.0, 0.0, 0.5]], dtype=np.float32)
    w = specs.vot_weighted_cost(c, share, 15.0, 10.0)
    assert w[0, 0] == pytest.approx(10.0)                    # all rail: unchanged
    assert w[0, 1] == pytest.approx(10.0 * 15.0 / 10.0)      # all other
    # gate exp(-w/(beta*15/60)) equals exp(-c/(beta*VoT_ij/60))
    beta, vot_ij = 40.0, 0.5 * 15.0 + 0.5 * 10.0
    assert np.exp(-w[0, 2] / (beta * 15.0 / 60)) == pytest.approx(
        np.exp(-10.0 / (beta * vot_ij / 60)), rel=1e-5)
    nan = specs.vot_weighted_cost(c, np.array([[np.nan] * 3]), 15.0, 10.0)
    assert nan == pytest.approx(c)                           # unknown: rail value
