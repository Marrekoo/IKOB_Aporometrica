"""Weibull margin and Gumbel-Hougaard copula (threshold-gate model)."""

import json

import numpy as np
import pytest

from ikob2.core.compose import (
    comonotone,
    compose_filters,
    frechet_violation,
    gumbel_hougaard,
)
from ikob2.core.decay_curves import apply_decay, exponential, weibull
from ikob2.core.numerics import DTYPE, ensure_dense
from ikob2.domain.filter_config import (
    CopulaSpec,
    FilterConfigError,
    load_filter_config,
)


# ── Weibull ──────────────────────────────────────────────────────────

def test_weibull_matches_closed_form():
    t = np.array([0.0, 10.0, 30.0, 60.0, 120.0], dtype=DTYPE)
    k, eta = 2.5, 40.0
    np.testing.assert_allclose(
        weibull(t, k, eta), np.exp(-(t / eta) ** k), rtol=1e-5)


def test_weibull_is_a_survival_function():
    t = np.linspace(0, 300, 500, dtype=DTYPE)
    s = weibull(t, 2.0, 45.0)
    assert s[0] == 1.0
    assert np.all(np.diff(s) <= 0)
    assert s.min() >= 0.0 and s.max() <= 1.0
    assert s[-1] < 1e-6


def test_weibull_shape_one_is_exponential():
    t = np.linspace(0, 100, 50, dtype=DTYPE)
    eta = 30.0
    np.testing.assert_allclose(
        weibull(t, 1.0, eta), exponential(t, 1.0 / eta), rtol=1e-5)


def test_weibull_median_is_scale_times_ln2_root():
    k, eta = 3.0, 50.0
    median = eta * np.log(2.0) ** (1.0 / k)
    assert float(weibull(np.array([median]), k, eta)[0]) == pytest.approx(0.5, abs=1e-6)


def test_weibull_extreme_inputs_do_not_overflow():
    t = np.array([0.0, 1e6, 1e12], dtype=DTYPE)
    with np.errstate(all="raise"):
        s = weibull(t, 8.0, 5.0)
    assert np.all(np.isfinite(s))
    assert s[0] == 1.0 and s[1] < 1e-30 and s[2] < 1e-30


@pytest.mark.parametrize("shape,scale", [(0, 10), (-1, 10), (2, 0), (2, -5),
                                          (np.nan, 10), (2, np.inf)])
def test_weibull_rejects_invalid_params(shape, scale):
    with pytest.raises(ValueError):
        weibull(np.ones(3, dtype=DTYPE), shape, scale)


def test_weibull_through_apply_decay():
    cost = np.array([[0.0, 30.0], [30.0, 0.0]], dtype=DTYPE)
    got = ensure_dense(apply_decay(cost, "weibull", (2.0, 30.0), epsilon=None))
    np.testing.assert_allclose(got, [[1, np.exp(-1)], [np.exp(-1), 1]], rtol=1e-5)


# ── Gumbel-Hougaard ──────────────────────────────────────────────────

def _grid(n=41):
    g = np.linspace(0.0, 1.0, n, dtype=DTYPE)
    u, v = np.meshgrid(g, g)
    return u, v


def test_gumbel_matches_paper_formula():
    u = np.array([0.8, 0.5, 0.3], dtype=DTYPE)
    v = np.array([0.6, 0.9, 0.2], dtype=DTYPE)
    for theta in (1.25, 1.5, 2.0, 4.0):
        lt, lm = -np.log(u.astype(float)), -np.log(v.astype(float))
        expected = np.exp(-(lt ** theta + lm ** theta) ** (1.0 / theta))
        np.testing.assert_allclose(
            gumbel_hougaard(u, v, theta), expected, rtol=1e-5)


def test_gumbel_theta_one_is_independence():
    u, v = _grid()
    np.testing.assert_allclose(gumbel_hougaard(u, v, 1.0), u * v, atol=1e-7)


def test_gumbel_large_theta_approaches_comonotone_and_is_stable():
    u, v = _grid()
    with np.errstate(all="raise"):
        big = gumbel_hougaard(u, v, 1e4)
    assert np.all(np.isfinite(big))
    np.testing.assert_allclose(big, comonotone(u, v), atol=2e-3)


def test_gumbel_boundary_conditions():
    g = np.linspace(0.0, 1.0, 11, dtype=DTYPE)
    one = np.ones_like(g)
    zero = np.zeros_like(g)
    for theta in (1.5, 4.0):
        np.testing.assert_allclose(gumbel_hougaard(g, one, theta), g, atol=1e-6)
        np.testing.assert_allclose(gumbel_hougaard(one, g, theta), g, atol=1e-6)
        np.testing.assert_array_equal(gumbel_hougaard(g, zero, theta), zero)
        np.testing.assert_array_equal(gumbel_hougaard(zero, zero, theta), zero)


def test_gumbel_is_symmetric_and_increasing_in_theta():
    u, v = _grid()
    for theta in (1.25, 2.0):
        np.testing.assert_allclose(
            gumbel_hougaard(u, v, theta), gumbel_hougaard(v, u, theta), atol=1e-7)
    # positive dependence raises the joint survival (paper Sec. 2.4)
    prev = u * v
    for theta in (1.25, 1.5, 2.0, 4.0):
        cur = gumbel_hougaard(u, v, theta)
        assert np.all(cur >= prev - 1e-6)
        prev = cur


@pytest.mark.parametrize("theta", [1.0, 1.25, 2.0, 4.0, 50.0])
def test_gumbel_respects_frechet_bounds(theta):
    u, v = _grid()
    assert frechet_violation(u, v, gumbel_hougaard(u, v, theta)) <= 5e-6


@pytest.mark.parametrize("theta", [0.99, 0.0, -1.0, np.inf, np.nan])
def test_gumbel_rejects_invalid_theta(theta):
    with pytest.raises(ValueError):
        gumbel_hougaard(np.ones(2, dtype=DTYPE), np.ones(2, dtype=DTYPE), theta)


def test_compose_filters_gumbel_and_theta_rules():
    u = np.full((3, 3), 0.7, dtype=DTYPE)
    v = np.full((3, 3), 0.4, dtype=DTYPE)
    out = ensure_dense(compose_filters(u, v, family="gumbel", theta=2.0,
                                       epsilon=None))
    np.testing.assert_allclose(out, gumbel_hougaard(u, v, 2.0), rtol=1e-6)

    with pytest.raises(ValueError, match="requires theta"):
        compose_filters(u, v, family="gumbel", epsilon=None)
    with pytest.raises(ValueError, match="takes no parameter"):
        compose_filters(u, v, family="independence", theta=2.0, epsilon=None)


# ── Config integration ───────────────────────────────────────────────

def _config(tmp_path, curve_time, copula):
    cls = {"time": curve_time,
           "cost": {"curve": "exponential", "beta": 0.5}}
    body = {"copula": copula,
            "modes": {"fiets": {"classes": {
                c: cls for c in ("laag", "middellaag", "middelhoog", "hoog")}}}}
    path = tmp_path / "filters.json"
    path.write_text(json.dumps(body))
    return path


def test_filter_config_loads_weibull_and_gumbel(tmp_path):
    path = _config(tmp_path,
                   {"curve": "weibull", "shape": 2.5, "scale": 40.0},
                   {"family": "gumbel", "theta": 1.5})
    fc = load_filter_config(path)
    cf = fc.filters["fiets"]["laag"]
    assert cf.time.curve == "weibull" and cf.time.params == (2.5, 40.0)
    assert cf.copula == CopulaSpec("gumbel", 1.5)


def test_filter_config_rejects_bad_weibull_and_gumbel(tmp_path):
    with pytest.raises(FilterConfigError, match="missing parameter"):
        load_filter_config(_config(
            tmp_path, {"curve": "weibull", "shape": 2.0},
            {"family": "independence"}))
    with pytest.raises(FilterConfigError, match="theta >= 1"):
        load_filter_config(_config(
            tmp_path, {"curve": "weibull", "shape": 2.0, "scale": 30.0},
            {"family": "gumbel", "theta": 0.5}))
    with pytest.raises(FilterConfigError, match="requires theta"):
        load_filter_config(_config(
            tmp_path, {"curve": "weibull", "shape": 2.0, "scale": 30.0},
            {"family": "gumbel"}))
