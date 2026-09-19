"""Car time and money, and the crow-fly detour model."""

import numpy as np
import pytest

from ikob2.skims.car import (
    ELECTRIC_CAR,
    FOSSIL_CAR,
    SHARED_CAR,
    TAXI,
    CarCostModel,
    DetourModel,
    calibrate_detour,
    car_time_and_cost,
    crowfly_km,
    parking_times,
)


def test_parking_times_by_urbanisation_class():
    arr, dep = parking_times([1, 2, 3, 4, 5, np.nan, 9])
    np.testing.assert_array_equal(arr, [12, 8, 4, 0, 0, 0, 0])
    np.testing.assert_array_equal(dep, arr / 4)


def test_legacy_rates():
    assert FOSSIL_CAR.variable_eur_per_km == 0.16
    assert ELECTRIC_CAR.variable_eur_per_km == 0.05
    assert (SHARED_CAR.variable_eur_per_km, SHARED_CAR.per_minute_eur) == (0.33, 0.05)
    assert (TAXI.variable_eur_per_km, TAXI.per_minute_eur) == (2.40, 0.40)
    assert FOSSIL_CAR.matrix_id != TAXI.matrix_id
    with pytest.raises(ValueError, match="non-negative"):
        CarCostModel(-0.1)


def test_time_and_cost_hand_computed():
    drive = np.array([[10.0, 30.0], [20.0, np.nan]])
    dist = np.array([[5.0, 25.0], [12.0, 40.0]])
    t, c = car_time_and_cost(
        drive, dist, CarCostModel(0.16, road_pricing_eur_per_km=0.04),
        origin_urbanisation=[1, 5], dest_urbanisation=[2, 1],
        dest_parking_cost_eur=[1.0, 3.0])
    # origin 0 (class 1) departs in 3 min; destination 1 (class 1) 12 min
    assert t[0, 0] == pytest.approx(10 + 3 + 8)
    assert t[0, 1] == pytest.approx(30 + 3 + 12)
    assert t[1, 0] == pytest.approx(20 + 0 + 8)
    assert np.isnan(t[1, 1])
    assert c[0, 0] == pytest.approx(0.20 * 5 + 1.0)
    assert c[0, 1] == pytest.approx(0.20 * 25 + 3.0)


def test_taxi_and_shared_car_charge_per_minute_of_total_time():
    drive = np.array([[20.0]])
    dist = np.array([[10.0]])
    _, c = car_time_and_cost(drive, dist, TAXI)
    assert c[0, 0] == pytest.approx(2.40 * 10 + 0.40 * 20)
    _, c = car_time_and_cost(drive, dist, SHARED_CAR,
                             dest_urbanisation=[1])
    assert c[0, 0] == pytest.approx(0.33 * 10 + 0.05 * (20 + 12))


def test_shape_mismatch():
    with pytest.raises(ValueError, match="differ in shape"):
        car_time_and_cost(np.ones((2, 2)), np.ones((2, 3)), FOSSIL_CAR)


def test_crowfly_and_detour_model():
    d = crowfly_km([[0, 0]], [[3000, 4000], [0, 0]])
    np.testing.assert_allclose(d, [[5.0, 0.0]])
    m = DetourModel((1.0, 11.0), (1.5, 1.2))
    np.testing.assert_allclose(m.route_km([0.5, 1.0, 6.0, 11.0, 50.0]),
                               [0.75, 1.5, 6 * 1.35, 13.2, 60.0])
    assert DetourModel.constant(1.3).route_km([10.0])[0] == pytest.approx(13.0)
    with pytest.raises(ValueError, match=">= 1"):
        DetourModel((1.0,), (0.9,))
    with pytest.raises(ValueError, match="increasing"):
        DetourModel((5.0, 1.0), (1.2, 1.3))


def test_calibration_recovers_band_medians_and_round_trips(tmp_path):
    rng = np.random.default_rng(0)
    crow = rng.uniform(0.5, 60, 4000)
    true = np.where(crow < 3, 1.6, np.where(crow < 20, 1.35, 1.2))
    route = crow * true * rng.uniform(0.97, 1.03, crow.size)
    m = calibrate_detour(crow, route)
    assert m.route_km([1.5])[0] / 1.5 == pytest.approx(1.6, rel=0.05)
    assert m.route_km([10.0])[0] / 10 == pytest.approx(1.35, rel=0.05)
    assert m.route_km([40.0])[0] / 40 == pytest.approx(1.2, rel=0.05)
    p = tmp_path / "d.json"
    m.save(p)
    assert DetourModel.load(p) == m
    with pytest.raises(ValueError, match="enough"):
        calibrate_detour(crow[:5], route[:5])
