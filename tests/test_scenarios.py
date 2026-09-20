"""Flat-price calibration (S4) and the fastest-acceptable choice rule."""

import numpy as np
import pytest

from ikob2.run.accessibility import MixedMode, ModeMatrices, OptionSet
from ikob2.run.scenarios import Usage, calibrate_flat, choice_terms, lime_usage
from ikob2.run.shared_bike import SharedBikeTariffs
from tests.test_price_scale import opts
from tests.test_run_accessibility import ENV, MARGINS, DESTS, ORIGINS, world


def test_choice_terms_sorts_by_time_and_tracks_the_cheaper_faster_option():
    t = np.array([30.0, 15.0, 20.0]).reshape(3, 1, 1)
    c = np.array([5.0, 9.0, 4.0]).reshape(3, 1, 1)
    order, ts, cs, hi = choice_terms(t, c)
    assert list(order[:, 0, 0]) == [1, 2, 0]
    assert list(ts[:, 0, 0]) == [15.0, 20.0, 30.0]
    # hi: inf for the fastest; then max(c, cheapest faster)
    assert np.isinf(hi[0, 0, 0]) or hi[0, 0, 0] >= 9.0
    assert hi[1, 0, 0] == 9.0 and hi[2, 0, 0] == 5.0    # 4 vs min(9, 4)=4 -> 5


def usage(lime_cost, scale=None):
    pop, jobs, wfh, wage, *_ = world()
    plain, fast = opts(lime_cost)
    return lime_usage(origins=ORIGINS, destinations=DESTS, populations=pop,
                      sector_jobs=jobs, wfh_share=wfh, sector_wage=wage,
                      envelope=ENV, time_margins=MARGINS,
                      mode=OptionSet((plain, fast), "pt"), price_scale=scale)


def test_revenue_is_price_times_rentals_and_scales_with_concessions():
    u = usage(6.0)
    assert u.rentals > 0 and u.revenue == pytest.approx(6.0 * u.rentals,
                                                        rel=1e-4)
    half = usage(6.0, {n: 0.5 for n in u.by_segment})
    # with half the price fewer people are priced out: rentals do not fall,
    # revenue per rental halves, scaled rentals are half of the rentals
    assert half.rentals >= u.rentals * (1 - 1e-6)
    assert half.revenue / half.rentals == pytest.approx(3.0, rel=1e-4)
    assert half.rentals_scaled == pytest.approx(half.rentals / 2, rel=1e-4)


def test_no_lime_option_means_no_usage():
    pop, jobs, wfh, wage, *_ = world()
    plain, _ = opts(6.0)
    u = lime_usage(origins=ORIGINS, destinations=DESTS, populations=pop,
                   sector_jobs=jobs, wfh_share=wfh, sector_wage=wage,
                   envelope=ENV, time_margins=MARGINS,
                   mode=OptionSet((plain,), "pt"))
    assert u.revenue == 0 and u.rentals == 0


def test_calibration_recovers_the_price_when_volume_does_not_react():
    tiers = SharedBikeTariffs()

    def usage_for(p):
        price = 4.0 if p is None else p
        return Usage(revenue=4.0 * 100.0 if p is None else p * 100.0,
                     rentals=100.0, rentals_scaled=100.0)
    for method in ("weighted_mean", "fixed_point"):
        p, info = calibrate_flat(usage_for, method=method)
        assert p == pytest.approx(4.0, abs=0.01), (method, info)
    assert tiers.lime_tiers


def test_fixed_point_charges_more_when_a_higher_price_loses_rentals():
    def usage_for(p):
        if p is None:
            return Usage(400.0, 100.0, 100.0)        # baseline: 4.0 x 100
        return Usage(p * 100 * np.exp(-0.1 * (p - 4.0)),
                     100 * np.exp(-0.1 * (p - 4.0)),
                     100 * np.exp(-0.1 * (p - 4.0)))
    pm, _ = calibrate_flat(usage_for, method="weighted_mean")
    pf, info = calibrate_flat(usage_for, method="fixed_point")
    # p exp(-0.1 (p - 4)) = 4 has its lower root at p = 4 by construction;
    # a baseline volume above the flat one pushes the fixed point up
    assert pm == pytest.approx(4.0) and pf == pytest.approx(4.0, abs=0.01)

    def worse(p):
        if p is None:
            return Usage(400.0, 100.0, 100.0)
        v = 90.0 * np.exp(-0.1 * (p - 4.0))          # flat loses 10% volume
        return Usage(p * v, v, v)
    pf2, _ = calibrate_flat(worse, method="fixed_point")
    assert pf2 > 4.0 * 1.05                          # compensates the loss


def test_effectiveness_compares_gain_and_cost_shares():
    import pandas as pd
    from ikob2.run.scenarios import effectiveness

    def table(vals):
        rows = []
        for seg, ic, a in vals:
            rows.append({"buurtcode": "O", "mode": "pt_v2", "segment": seg,
                         "income_class": ic, "household_type": "single",
                         "population": 10.0, "accessibility": a})
        return pd.DataFrame(rows)
    base = table([("single_D2", "D2", 100.0), ("single_D9", "D9", 500.0)])
    scen = table([("single_D2", "D2", 130.0), ("single_D9", "D9", 501.0)])
    ub = {"single_D2": {"revenue": 10.0}, "single_D9": {"revenue": 90.0}}
    us = {"single_D2": {"revenue": 5.0}, "single_D9": {"revenue": 45.0}}
    e = effectiveness(base, scen, ub, us)
    assert e.loc["D2", "gain"] == 300.0 and e.loc["D9", "gain"] == 10.0
    assert e.loc["D2", "cost"] == 5.0 and e.loc["D9", "cost"] == 45.0
    # a blanket half-price cut: the poor get 97% of the gain for 10% of the cost
    assert e.loc["D2", "share_ratio"] > 1 > e.loc["D9", "share_ratio"]
    assert e["gain_share"].sum() == pytest.approx(1.0)


def test_price_from_prices_the_baseline_choices_at_new_prices():
    """Compensation at baseline volume: half the Lime price on the same
    choices collects half the revenue, whatever the volume response."""
    pop, jobs, wfh, wage, *_ = world()
    kw = dict(origins=ORIGINS, destinations=DESTS, populations=pop,
              sector_jobs=jobs, wfh_share=wfh, sector_wage=wage,
              envelope=ENV, time_margins=MARGINS)
    plain, fast = opts(6.0)
    base = OptionSet((plain, fast), "pt")
    plain2, cheap = opts(3.0)
    cheap_set = OptionSet((plain2, cheap), "pt")
    u0 = lime_usage(mode=base, **kw)
    u_new = lime_usage(mode=cheap_set, **kw)               # volumes react
    u_at0 = lime_usage(mode=base, price_from=cheap_set, **kw)
    assert u_at0.rentals == pytest.approx(u0.rentals)      # same choices
    assert u_at0.revenue == pytest.approx(u0.revenue / 2, rel=1e-4)
    assert u_new.rentals >= u0.rentals                      # cheaper: more accept
    with pytest.raises(ValueError, match="options"):
        lime_usage(mode=base, price_from=OptionSet((plain2,), "pt"), **kw)
