import pytest

from ikob2 import params
from ikob2.run.costs import (annual_rentals, annuity_factor, hub_annual_cost,
                             hubs_for_budget, rides_per_bike_per_day,
                             s1_compensation)

C = params.DEFAULTS.costs


def test_hub_cost_public_and_municipal():
    h = hub_annual_cost(C, 1.0)
    # 75,000 / 10 hubs = 7,500 over 5 years; 25,000 / 10 running per hub
    assert h["capex_per_year_public"] == pytest.approx(1500.0)
    assert h["running_per_year"] == pytest.approx(2500.0)   # no tariff compensation
    assert h["public"] == pytest.approx(4000.0)
    assert h["municipal"] == pytest.approx(750.0 + 2500.0)  # 50% provincial share
    assert hub_annual_cost(C, 27)["public"] == pytest.approx(108000.0)


def test_annuity_and_budget_in_hubs():
    assert annuity_factor(5, 0.0) == 5
    assert annuity_factor(5, 0.04) == pytest.approx(4.4518, abs=1e-3)
    assert hubs_for_budget(C, 40000.0) == pytest.approx(10.0)
    with pytest.raises(ValueError):
        annuity_factor(0, 0.0)


def test_volume_from_the_observed_rides():
    assert annual_rentals(C) == pytest.approx(1_875_000.0)   # 1.25 M in 8 months
    assert rides_per_bike_per_day(C) == pytest.approx(2.57, abs=0.01)


def test_compensation_is_the_revenue_foregone_at_baseline_volume():
    r = s1_compensation(C, revenue_base=300.0, revenue_new_at_base_volume=150.0,
                        rentals_base=100.0)
    assert r["mean_price_eur"] == 3.0 and r["share_of_revenue"] == 0.5
    assert r["annual_revenue_eur"] == pytest.approx(1_875_000 * 3.0)
    assert r["compensation_eur_year"] == pytest.approx(1_875_000 * 1.5)
    with pytest.raises(ValueError):
        s1_compensation(C, 1.0, 1.0, 0.0)
