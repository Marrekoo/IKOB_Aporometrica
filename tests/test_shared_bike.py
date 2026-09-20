import numpy as np
import pytest

from ikob2.run.accessibility import MixedMode, OptionSet
from ikob2.run.shared_bike import SharedBikeTariffs, shared_bike_modes

SHAPE = (3, 4)


def chains():
    t = {m: np.full(SHAPE, v, dtype=np.float32)
         for m, v in (("pt", 60.0), ("pt_wb", 50.0), ("pt_bw", 45.0),
                      ("pt_bb", 35.0))}
    c = {m: {"time": t[m]} for m in t}
    c["pt_bw"]["access_min"] = np.full(SHAPE, 10.0, dtype=np.float32)
    c["pt_bb"]["access_min"] = np.full(SHAPE, 12.0, dtype=np.float32)
    fares = {m: np.full(SHAPE, 8.0, dtype=np.float32) for m in t}
    return c, fares


def test_v1_prices_ovfiets_egress_only():
    c, f = chains()
    out = shared_bike_modes(c, f, np.array([.9, .8, .5]), variants=("v1",))
    v1 = out["pt_v1"]
    assert isinstance(v1, OptionSet) and len(v1.options) == 2
    plain, egress = v1.options
    assert plain.cost[0, 0] == 8.0 and egress.cost[0, 0] == pytest.approx(12.8)
    assert egress.time[0, 0] == 50.0


def test_v2_owners_ride_free_others_pay_dockless_and_weights_sum_to_one():
    c, f = chains()
    p = np.array([.9, .8, .5])
    v2 = shared_bike_modes(c, f, p, SharedBikeTariffs(4.8, 1.0, 0.2,
                                             dockless_model="unlock_per_minute"),
                           variants=("v2",))["pt_v2"]
    assert isinstance(v2, MixedMode)
    (w_o, owners), (w_n, others) = v2.parts
    np.testing.assert_allclose(w_o + w_n, 1.0)
    costs_o = sorted(o.cost[0, 0] for o in owners.options)
    costs_n = sorted(o.cost[0, 0] for o in others.options)
    # owners: plain 8, own access 8, own access + OV-fiets 12.8, OV-fiets 12.8
    assert costs_o == pytest.approx([8.0, 8.0, 12.8, 12.8])
    # others: plain 8; dockless access 8+1+0.2*10; +OV-fiets 8+1+0.2*12+4.8; OV-fiets
    assert costs_n == pytest.approx([8.0, 11.0, 12.8, 16.2])


def test_v0_and_validation():
    c, f = chains()
    v0 = shared_bike_modes(c, f, np.array([.9, .8, .5]), variants=("v0",))["pt_v0"]
    (_, owners), (_, others) = v0.parts
    assert len(owners.options) == 2 and len(others.options) == 1
    with pytest.raises(ValueError):
        shared_bike_modes(c, f, np.array([1.5, 0, 0]))
    with pytest.raises(ValueError):
        shared_bike_modes(c, f, np.array([.5, .5, .5]), variants=("v9",))
    with pytest.raises(ValueError):
        SharedBikeTariffs(ovfiets_eur=-1)


def test_v3_splits_times_into_legs_and_v4_prices_dockless():
    from ikob2.run.accessibility import LegOptionSet
    from ikob2.run.shared_bike import dockless_mode
    c, f = chains()
    c["pt_bb"]["egress_min"] = np.full(SHAPE, 6.0, dtype=np.float32)
    c["pt_wb"]["egress_min"] = np.full(SHAPE, 6.0, dtype=np.float32)
    v3 = shared_bike_modes(c, f, np.array([.9, .8, .5]),
                           variants=("v3",), bike_fixed_min=1.0)["pt_v3"]
    (_, owners), _ = v3.parts
    assert isinstance(owners, LegOptionSet)
    bb = owners.options[2]                      # pt_bb: access 12+1, egress 6+1
    a, b, pt = (x[0, 0] for x in bb.times)
    assert (a, b, pt) == pytest.approx((13.0, 7.0, 35.0 - 13.0 - 7.0))
    assert owners.options[0].times[2][0, 0] == 60.0          # plain: all PT
    bike_t = np.full(SHAPE, 20.0, dtype=np.float32)
    v4 = dockless_mode(bike_t, np.array([.9, .8, .5]),
                       SharedBikeTariffs(4.8, 1.0, 0.2,
                                         dockless_model="unlock_per_minute"),
                       1.0)
    (_, own), (w_n, rent) = v4.parts
    assert own.options[0].cost is None
    assert rent.options[0].time[0, 0] == 21.0
    assert rent.options[0].cost[0, 0] == pytest.approx(1.0 + 0.2 * 20.0)
    np.testing.assert_allclose(w_n, [.1, .2, .5])
    with pytest.raises(ValueError, match="egress_min"):
        c2, f2 = chains()
        shared_bike_modes(c2, f2, np.array([.9, .8, .5]), variants=("v3",))


def kind_chains():
    """Egress chains per hub kind: Lime is faster but a different price."""
    c, f = chains()
    for kind, wb, bb in (("lime", 48.0, 33.0), ("ovfiets", 50.0, 35.0)):
        for mode, tm in ((f"pt_wb_{kind}", wb), (f"pt_bb_{kind}", bb)):
            c[mode] = {"time": np.full(SHAPE, tm, dtype=np.float32),
                       "egress_min": np.full(SHAPE, 10.0, dtype=np.float32)}
            f[mode] = np.full(SHAPE, 8.0, dtype=np.float32)
        c[f"pt_bb_{kind}"]["access_min"] = np.full(SHAPE, 12.0, np.float32)
    return c, f


def test_each_hub_kind_is_its_own_option_with_its_own_price():
    c, f = kind_chains()
    out = shared_bike_modes(c, f, np.array([.9, .8, .5]), variants=("v1",))
    plain, *egress = out["pt_v1"].options
    assert len(egress) == 2
    by_time = {o.time[0, 0]: o.cost[0, 0] for o in egress}
    assert by_time[48.0] == pytest.approx(8.0 + 3.0)      # Lime, 11 min rental
    assert by_time[50.0] == pytest.approx(8.0 + 4.8)      # OV-fiets flat


def test_a_slower_cheaper_hub_can_be_the_one_that_passes():
    """OV-fiets is faster (48 min, EUR 12.8) but Lime is slower (50 min) and
    cheaper (EUR 11.0). A person with a budget of EUR 12 can only use the
    Lime option: choosing the fastest hub would lose the pair."""
    from ikob2.run.accessibility import union_terms
    c, f = kind_chains()
    c["pt_wb_ovfiets"]["time"][:] = 48.0
    c["pt_wb_lime"]["time"][:] = 50.0
    out = shared_bike_modes(c, f, np.array([1.0, 1.0, 1.0]), variants=("v1",))
    opts = out["pt_v1"].options
    ts = np.stack([o.time for o in opts]); cs = np.stack([o.cost for o in opts])
    kept = {(float(t[0, 0]), round(float(cc[0, 0]), 2))
            for t, cc, _ in union_terms(ts, cs)}
    assert (50.0, 11.0) in kept and (48.0, 12.8) in kept    # both survive
    # a budget of EUR 12 (threshold) accepts (50 min, 11.0) but not (48, 12.8)
    budget = 12.0
    acceptable = [(t, cc) for t, cc in ((50.0, 11.0), (48.0, 12.8))
                  if cc <= budget]
    assert acceptable == [(50.0, 11.0)]


def test_lime_tiers_price_dockless_access_as_the_baseline():
    c, f = chains()
    p = np.array([.9, .8, .5])
    v2 = shared_bike_modes(c, f, p, variants=("v2",))["pt_v2"]   # default
    _, (_, others) = v2.parts
    costs = sorted(o.cost[0, 0] for o in others.options)
    # dockless access: 10 min ride + 1 min fixed = 11 min -> EUR 3 -> 8 + 3;
    # with OV-fiets: 12 + 1 = 13 min -> EUR 3, + 4.8 -> 8 + 3 + 4.8
    assert costs == pytest.approx([8.0, 11.0, 12.8, 15.8])
    with pytest.raises(ValueError, match="dockless_model"):
        SharedBikeTariffs(dockless_model="free")


def test_dockless_door_to_door_uses_the_same_model():
    from ikob2.run.shared_bike import dockless_mode
    bike_t = np.full(SHAPE, 20.0, dtype=np.float32)
    lime = dockless_mode(bike_t, np.array([.9, .8, .5]), SharedBikeTariffs(), 1.0)
    (_, _), (_, rent) = lime.parts
    assert rent.options[0].cost[0, 0] == 4.0          # 21 min rented
