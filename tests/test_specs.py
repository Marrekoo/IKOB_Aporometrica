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


# ── dual cut-offs (M0u, M0s) ─────────────────────────────────────────

def test_cutoff_cost_of_one_segment_is_its_quantile():
    assert specs.cutoff_cost([row(4.0, 24.0)], 0.5) == pytest.approx(14.0)
    # S(c) = (24 - c)/20 = 1/4 at c = 19
    assert specs.cutoff_cost([row(4.0, 24.0)], 0.25) == pytest.approx(19.0)
    # the atom moves the median down: 0.8 (24 - c)/20 = 1/2 at c = 11.5
    assert specs.cutoff_cost([row(4.0, 24.0, 0.2)], 0.5) == pytest.approx(11.5)
    assert specs.cutoff_cost([row(68.11, 68.11)], 0.5) == 68.11   # no spread
    assert specs.cutoff_cost([row(np.nan, np.nan, 1.0)], 0.5) == 0.0
    assert specs.cutoff_cost([row(4.0, 24.0, 0.6)], 0.5) == 0.0   # S(0+) = 0.4
    with pytest.raises(ValueError):
        specs.cutoff_cost([row(4.0, 24.0)], 1.0)


def test_cutoff_cost_of_a_mixture_by_hand():
    a, b = row(4.0, 24.0), row(30.0, 30.0)
    # S(c) = (S_a + 1{c <= 30})/2: 1/2 on (24, 30], 0 above
    assert specs.cutoff_cost([a, b], 0.5) == pytest.approx(30.0)
    # 0.75 = ((24 - c)/20 + 1)/2 at c = 14
    assert specs.cutoff_cost([a, b], 0.75) == pytest.approx(14.0)
    # weights: S = 0.25 S_a + 0.75 1{c <= 30} >= 0.8 only while S_a >= 0.2
    assert specs.cutoff_cost([a, b], 0.8, [1, 3]) == pytest.approx(20.0)
    # a censored segment (atom 1) weighs in with S = 0
    d1 = row(np.nan, np.nan, 1.0)
    assert specs.cutoff_cost([d1, a], 0.5, [0.6, 0.4]) == 0.0
    assert specs.cutoff_cost([d1, a], 0.25, [0.6, 0.4]) == pytest.approx(
        24.0 - 0.25 / 0.4 * 20.0)
    with pytest.raises(ValueError):
        specs.cutoff_cost([a, b], 0.5, [1.0])


def test_dual_cutoff_margins_are_steps_at_the_quantiles():
    k, eta = W.params
    assert specs.cutoff_time(W, 0.5) == pytest.approx(specs.median_time(W))
    assert specs.cutoff_time(W, 0.25) == pytest.approx(
        eta * np.log(4.0) ** (1 / k), rel=1e-6)
    for spec in ("m0", "m0u", "m0s"):
        t = specs.time_margin_for(spec, W)
        assert t.curve == "step"
        assert t.params[0] == pytest.approx(specs.median_time(W))
    assert specs.time_margin_for("m0s", W, 0.25).params[0] == pytest.approx(
        eta * np.log(4.0) ** (1 / k), rel=1e-6)
    seg = specs.cost_curve_factory("m0s", W, None)
    assert seg(row(4.0, 24.0)) == CurveSpec("step", (14.0,))
    assert seg(row(np.nan, np.nan, 1.0)) == CurveSpec("step", (0.0,))
    one = specs.cost_curve_factory("m0u", W, None, cost_cutoff=9.5)
    assert one(row(4.0, 24.0)) == one(row(1.0, 2.0)) == CurveSpec("step", (9.5,))
    with pytest.raises(ValueError, match="cut-off"):
        specs.cost_curve_factory("m0u", W, None)
    assert specs.cost_curve_factory("m0", W, 12.0) is None


def test_reported_atoms():
    env = pd.DataFrame({"household_type": ["single"] * 3,
                        "income_class": ["D1", "D2", "D3"],
                        "low": [np.nan, 4.0, 4.0], "high": [np.nan, 24.0, 24.0],
                        "atom": [1.0, 0.6, 0.2]})
    m0s = specs.reported_atoms("m0s", env, 0.5)
    assert m0s["atom"].tolist() == [1.0, 1.0, 0.0]   # cut-off 0, 0, 11.5
    assert specs.reported_atoms("m2", env, 0.5) is env
    assert specs.atom_reported("m0s") and not specs.atom_reported("m0u")
