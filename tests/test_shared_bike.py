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
    v2 = shared_bike_modes(c, f, p, SharedBikeTariffs(4.8, 1.0, 0.2),
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
