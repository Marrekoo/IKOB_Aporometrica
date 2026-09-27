import pandas as pd
import pytest

from ikob2.segments.car_availability import car_availability


def persons(rows):
    return pd.DataFrame(rows, columns=["HHSam", "HHGestInkG", "HHAuto",
                                       "OPRijbewijsAu", "FactorP", "WoGem"])


def test_weighted_share_and_shrinkage_to_national():
    rows = ([(1, 1, 1, 1, 2.0, 999)] * 3 + [(1, 1, 0, 0, 1.0, 999)]
            + [(1, 1, 0, 0, 1.0, 344)] * 10)
    t = car_availability(persons(rows), 344, prior=10.0)
    r = t[(t.household_type == "single") & (t.income_class == "D1")].iloc[0]
    nat = 6.0 / (6.0 + 1.0 + 10.0)                     # weighted, all rows
    assert r.share_national == pytest.approx(6 / 17)
    assert r.share_local == 0.0 and r.n_local == 10
    assert r.share == pytest.approx((10 * 0.0 + 10 * nat) / 20)
    assert len(t) == 40


def test_licence_basis_and_unmodelled_rows_dropped():
    rows = [(2, 5, 1, 0, 1.0, 1), (2, 5, 1, 1, 1.0, 1),
            (8, 5, 1, 1, 1.0, 1), (2, 11, 1, 1, 1.0, 1)]   # 'other', unknown income
    hh = car_availability(persons(rows), None, basis="household_car")
    lic = car_availability(persons(rows), None, basis="car_and_licence")
    def pick(t):
        return t[(t.household_type == "couple") & (t.income_class == "D5")].iloc[0]
    assert pick(hh).share == 1.0 and pick(lic).share == 0.5
    assert pick(hh).n_national == 2
    with pytest.raises(ValueError):
        car_availability(persons(rows), None, basis="x")
