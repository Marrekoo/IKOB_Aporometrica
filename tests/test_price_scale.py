"""Concessionary prices: a multiplier on the Lime part of a journey, by
segment."""

import numpy as np
import pytest

from ikob2.run.accessibility import (LegOption, LegOptionSet, ModeMatrices,
                                     OptionSet)
from ikob2.run.shared_bike import (SharedBikeTariffs, load_price_scales,
                                   segment_price_scales)
from tests.test_run_accessibility import NAMES, run, world

SEG = "single_D3"


def opts(lime_cost):
    """Plain (30 min, EUR 5) or a Lime option (15 min, EUR 5 + Lime)."""
    shape = (3, 4)
    plain = ModeMatrices(np.full(shape, 30.0, np.float32),
                         np.full(shape, 5.0, np.float32), "c")
    lime = np.full(shape, lime_cost, np.float32)
    fast = ModeMatrices(np.full(shape, 15.0, np.float32), 5.0 + lime, "c",
                        lime=lime, lime_rentals=1)
    return plain, fast


def acc(res, mode):
    t = res.table[res.table["mode"] == mode]
    return t.groupby("segment")["accessibility"].sum()


def test_a_segment_scale_changes_only_that_segment():
    pop, jobs, wfh, wage, *_ = world()
    plain, fast = opts(12.0)
    base = run(pop, jobs, wfh, wage, {"m": OptionSet((plain, fast), "pt")})
    cheap = run(pop, jobs, wfh, wage, {"m": OptionSet((plain, fast), "pt")},
                price_scale={SEG: 0.0})
    a0, a1 = acc(base, "m"), acc(cheap, "m")
    assert a1[SEG] > a0[SEG]                       # free Lime opens the pair
    others = [n for n in NAMES if n != SEG]
    np.testing.assert_allclose(a1[others], a0[others])
    # scale 0 equals an option that was free of the Lime price to begin with
    plain2, free = opts(0.0)
    ref = run(pop, jobs, wfh, wage, {"m": OptionSet((plain2, free), "pt")})
    assert a1[SEG] == pytest.approx(acc(ref, "m")[SEG], rel=1e-5)


def test_scale_one_and_no_scale_agree_and_lime_free_option_is_unaffected():
    pop, jobs, wfh, wage, *_ = world()
    plain, fast = opts(12.0)
    base = run(pop, jobs, wfh, wage, {"m": OptionSet((plain, fast), "pt")})
    ones = run(pop, jobs, wfh, wage, {"m": OptionSet((plain, fast), "pt")},
               price_scale={n: 1.0 for n in NAMES})
    np.testing.assert_allclose(acc(base, "m"), acc(ones, "m"))
    only_plain = run(pop, jobs, wfh, wage, {"m": OptionSet((plain,), "pt")},
                     price_scale={SEG: 0.0})
    np.testing.assert_allclose(acc(only_plain, "m"),
                               acc(run(pop, jobs, wfh, wage,
                                       {"m": OptionSet((plain,), "pt")}), "m"))


def test_legwise_options_scale_too():
    pop, jobs, wfh, wage, *_ = world()
    shape = (3, 4)
    half = np.full(shape, 15.0, np.float32)
    lime = np.full(shape, 12.0, np.float32)
    opt = LegOption((half, half), 5.0 + lime, "c", lime=lime, lime_rentals=1)
    mk = lambda: {"m": LegOptionSet((opt,), ("car", "car"))}  # noqa: E731
    base = acc(run(pop, jobs, wfh, wage, mk()), "m")
    cheap = acc(run(pop, jobs, wfh, wage, mk(), price_scale={SEG: 0.0}), "m")
    assert cheap[SEG] > base[SEG]
    assert cheap["couple_D5"] == pytest.approx(base["couple_D5"])


def test_price_scale_table(tmp_path):
    f = tmp_path / "t.csv"
    f.write_text("household_type,income_class,scale\n"
                 "*,D2,0.5\n*,D3,0.5\nsingle,D3,0.25\ncouple,*,0.9\n")
    t = load_price_scales(f)
    s = segment_price_scales(["single_D2", "single_D3", "couple_D3",
                              "couple_D7", "single_D7"], t)
    # later rows override: couple_D3 -> 0.9 (last row), single_D3 -> 0.25
    assert s == {"single_D2": 0.5, "single_D3": 0.25, "couple_D3": 0.9,
                 "couple_D7": 0.9, "single_D7": 1.0}
    assert segment_price_scales(["single_D3"], None) == {"single_D3": 1.0}
    (tmp_path / "bad.csv").write_text("household_type,income_class,scale\n"
                                      "*,D99,0.5\n")
    with pytest.raises(ValueError, match="unknown income_class"):
        load_price_scales(tmp_path / "bad.csv")
    (tmp_path / "neg.csv").write_text("household_type,income_class,scale\n"
                                      "*,D2,-1\n")
    with pytest.raises(ValueError, match="scale"):
        load_price_scales(tmp_path / "neg.csv")


def test_flat_model_charges_the_same_per_rental_at_any_duration():
    t = SharedBikeTariffs(dockless_model="flat", flat_eur=3.5, lime_scale=0.5)
    assert list(t.dockless_eur([2.0, 15.0, 39.0], 1.0)) == [1.75] * 3
    assert t.egress_eur([5.0], "lime", 1.0)[0] == pytest.approx(1.75)
    assert t.egress_eur([5.0], "ovfiets", 1.0)[0] == t.ovfiets_eur


# ── fare concessions: a multiplier on the public transport fare part ───

def test_a_fare_scale_reprices_only_that_segment_and_works_for_plain_modes():
    pop, jobs, wfh, wage, time, cost = world()
    fare = (cost * 0.8).astype(np.float32)               # most of the cost is fare
    mm = ModeMatrices(time, cost, "c", fare=fare)
    base = run(pop, jobs, wfh, wage, {"car": mm})
    cheap = run(pop, jobs, wfh, wage, {"car": mm}, fare_scale={SEG: 0.0})
    a0, a1 = acc(base, "car"), acc(cheap, "car")
    assert a1[SEG] > a0[SEG]
    others = [n for n in NAMES if n != SEG]
    np.testing.assert_allclose(a1[others], a0[others])
    # scale 0 equals the same journey priced without its fare
    ref = run(pop, jobs, wfh, wage,
              {"car": ModeMatrices(time, (cost - fare).astype(np.float32), "c")})
    assert a1[SEG] == pytest.approx(acc(ref, "car")[SEG], rel=1e-5)


def test_fare_and_lime_scales_combine_in_an_option_set():
    pop, jobs, wfh, wage, *_ = world()
    shape = (3, 4)
    fare = np.full(shape, 30.0, np.float32)     # inside the budget interval of SEG
    lime = np.full(shape, 15.0, np.float32)
    opt = ModeMatrices(np.full(shape, 20.0, np.float32), fare + lime, "c",
                       fare=fare, lime=lime, lime_rentals=1)
    plain = run(pop, jobs, wfh, wage, {"m": OptionSet((opt,), "pt")})
    both = run(pop, jobs, wfh, wage, {"m": OptionSet((opt,), "pt")},
               price_scale={SEG: 0.0}, fare_scale={SEG: 0.0})
    only_fare = run(pop, jobs, wfh, wage, {"m": OptionSet((opt,), "pt")},
                    fare_scale={SEG: 0.0})
    a0, ab, af = acc(plain, "m")[SEG], acc(both, "m")[SEG], acc(only_fare, "m")[SEG]
    assert ab >= af >= a0 and ab > a0
    free = ModeMatrices(np.full(shape, 20.0, np.float32), np.zeros(shape, np.float32), "c")
    assert ab == pytest.approx(acc(run(pop, jobs, wfh, wage,
                                       {"m": OptionSet((free,), "pt")}), "m")[SEG], rel=1e-5)


def test_fare_scale_reaches_legwise_options():
    pop, jobs, wfh, wage, *_ = world()
    shape = (3, 4)
    half = np.full(shape, 15.0, np.float32)
    fare = np.full(shape, 30.0, np.float32)
    opt = LegOption((half, half), fare, "c", fare=fare)
    mk = lambda: {"m": LegOptionSet((opt,), ("car", "car"))}  # noqa: E731
    base = acc(run(pop, jobs, wfh, wage, mk()), "m")
    cheap = acc(run(pop, jobs, wfh, wage, mk(), fare_scale={SEG: 0.0}), "m")
    assert cheap[SEG] > base[SEG] and cheap["couple_D5"] == pytest.approx(base["couple_D5"])
