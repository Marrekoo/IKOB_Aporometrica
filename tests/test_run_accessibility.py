"""End-to-end accessibility glue on a synthetic world."""

import numpy as np
import pandas as pd
import pytest

from ikob2.core import families as fam
from ikob2.domain.filter_config import CopulaSpec, CurveSpec
from ikob2.run.accessibility import (
    WFH_TYPES,
    ModeMatrices,
    run_accessibility,
)
from ikob2.segments.bridge import envelope_segment_names, load_reference_budgets
from ikob2.segments.jobs import sector_income_weights, sector_pools
from ikob2.segments.lisa import SECTORS
from ikob2.segments.time_margins import load_time_margins
from ikob2.segments.wfh import split_jobs_by_wfh

ENV = load_reference_budgets("data/envelope/reference_budgets.csv")
MARGINS = load_time_margins("data/margins/S_T_work.csv")
NAMES = envelope_segment_names(ENV)
ORIGINS = ["O0", "O1", "O2"]
DESTS = ["D0", "D1", "D2", "D3"]


def world(seed=0):
    rng = np.random.default_rng(seed)
    pop = pd.DataFrame(rng.uniform(10, 200, (3, len(NAMES))),
                       index=ORIGINS, columns=NAMES)
    jobs = pd.DataFrame(rng.uniform(5, 500, (4, len(SECTORS))),
                        index=DESTS, columns=list(SECTORS))
    wfh = pd.Series(rng.uniform(0.2, 0.7, len(SECTORS)), index=list(SECTORS))
    wage = pd.Series(rng.uniform(15, 35, len(SECTORS)), index=list(SECTORS))
    time = rng.uniform(3, 80, (3, 4)).astype(np.float32)
    time[0, 0] = 0.0
    cost = rng.uniform(1, 30, (3, 4)).astype(np.float32)
    cost[0, 0] = 0.0
    return pop, jobs, wfh, wage, time, cost


def run(pop, jobs, wfh, wage, matrices, **kw):
    return run_accessibility(
        origins=ORIGINS, destinations=DESTS, populations=pop,
        sector_jobs=jobs, wfh_share=wfh, sector_wage=wage, envelope=ENV,
        time_margins=MARGINS, matrices=matrices, epsilon=None, **kw)


def test_table_shape_and_columns():
    pop, jobs, wfh, wage, time, cost = world()
    res = run(pop, jobs, wfh, wage,
              {"bike": ModeMatrices(time),
               "car": ModeMatrices(time * 0.5, cost, "carcost")})
    t = res.table
    assert len(t) == 3 * len(NAMES) * 2
    assert {"buurtcode", "mode", "segment", "household_type", "income_class",
            "population", "accessibility", "atom",
            "accessibility_normalised"} <= set(t.columns)
    assert set(t["mode"]) == {"bike", "car"}
    assert res.meta["segments"] == 40 and res.meta["modes"] == ["bike", "car"]
    assert (t["accessibility"] >= 0).all() and t["accessibility"].notna().all()


def test_bike_matches_a_hand_computation():
    pop, jobs, wfh, wage, time, _ = world()
    res = run(pop, jobs, wfh, wage, {"bike": ModeMatrices(time)})
    W = sector_income_weights(wage, jobs.sum())
    no, yes = split_jobs_by_wfh(jobs, wfh)
    expected = np.zeros(3)
    for wtype, j in (("no_wfh", no), ("wfh_possible", yes)):
        pool = sector_pools(j, DESTS, W)["D5"].astype(float)
        k, eta = MARGINS[("bike", wtype)].params
        expected += fam.survival("weibull", (k, eta),
                                 time.astype(float)) @ pool
    got = res.table[(res.table.segment == "couple_D5")].sort_values(
        "buurtcode")["accessibility"].to_numpy()
    np.testing.assert_allclose(got, expected, rtol=1e-4)


def test_censored_first_decile_reaches_only_free_trips_by_car():
    pop, jobs, wfh, wage, time, cost = world()
    res = run(pop, jobs, wfh, wage, {"car": ModeMatrices(time, cost, "c")})
    d1 = res.table[res.table.income_class == "D1"]
    a = d1.pivot_table(index="buurtcode", columns="household_type",
                       values="accessibility")
    # only O0 -> D0 has cost 0: origins O1, O2 reach nothing by car
    assert (a.loc[["O1", "O2"]] == 0).all().all()
    assert (a.loc["O0"] > 0).all()
    assert d1["accessibility_normalised"].isna().all()       # atom = 1
    d5 = res.table[res.table.income_class == "D5"]
    assert d5["accessibility_normalised"].notna().all()


def test_priced_mode_gates_by_segment_budget_free_mode_does_not():
    pop, jobs, wfh, wage, time, cost = world()
    cost[:] = 60.0                                     # dear trips everywhere
    cost[0, 0] = 0.0
    car = run(pop, jobs, wfh, wage, {"car": ModeMatrices(time, cost, "c")}).table
    hi = car[(car.segment == "couple_D10") & (car.buurtcode == "O1")]
    lo = car[(car.segment == "couple_D2") & (car.buurtcode == "O1")]
    assert hi["accessibility"].iloc[0] > lo["accessibility"].iloc[0]  # budget 224-691 vs 5-24
    # an unpriced mode with the same times ignores budgets: identical
    # filter for all segments of a class
    bike = run(pop, jobs, wfh, wage, {"bike": ModeMatrices(time)}).table
    a = bike[(bike.income_class == "D5") & (bike.buurtcode == "O1")]
    assert a["accessibility"].nunique() == 1


def test_identical_wfh_curves_reduce_to_a_single_job_pool():
    pop, jobs, wfh, wage, time, _ = world()
    same = {("bike", w): CurveSpec("weibull", (2.5, 30.0)) for w in WFH_TYPES}
    res = run_accessibility(
        origins=ORIGINS, destinations=DESTS, populations=pop,
        sector_jobs=jobs, wfh_share=wfh, sector_wage=wage, envelope=ENV,
        time_margins=same, matrices={"bike": ModeMatrices(time)},
        epsilon=None)
    W = sector_income_weights(wage, jobs.sum())
    pool = sector_pools(jobs, DESTS, W)["D7"].astype(float)
    exp = fam.survival("weibull", (2.5, 30.0), time.astype(float)) @ pool
    got = res.table[res.table.segment == "single_D7"].sort_values(
        "buurtcode")["accessibility"].to_numpy()
    np.testing.assert_allclose(got, exp, rtol=1e-4)


def test_gumbel_copula_raises_priced_accessibility_and_summary_weights():
    pop, jobs, wfh, wage, time, cost = world()
    mats = {"car": ModeMatrices(time, cost, "c")}
    ind = run(pop, jobs, wfh, wage, mats)
    dep = run(pop, jobs, wfh, wage, mats, copula=CopulaSpec("gumbel", 3.0))
    assert dep.table["accessibility"].sum() >= ind.table["accessibility"].sum()
    s = ind.summary("income_class")
    t = ind.table[(ind.table["mode"] == "car")
                  & (ind.table.income_class == "D5")]
    exp = (t.accessibility * t.population).sum() / t.population.sum()
    assert s.loc[("car", "D5"), "accessibility"] == pytest.approx(exp)


def test_input_validation():
    pop, jobs, wfh, wage, time, cost = world()
    with pytest.raises(ValueError, match="does not match"):
        run(pop, jobs, wfh, wage, {"bike": ModeMatrices(time[:2])})
    with pytest.raises(KeyError, match="No time margin"):
        run(pop, jobs, wfh, wage, {"walk": ModeMatrices(time)})
    with pytest.raises(KeyError, match="lack segment"):
        run(pop.drop(columns=NAMES[0]), jobs, wfh, wage,
            {"bike": ModeMatrices(time)})
    with pytest.raises(ValueError, match="same shape"):
        ModeMatrices(time, cost[:2], "c")
    with pytest.raises(ValueError, match="cost_id"):
        ModeMatrices(time, cost)


def test_availability_scales_expected_but_not_conditional_accessibility():
    from ikob2.segments.ownership import availability_frames
    pop, jobs, wfh, wage, time, cost = world()
    mats = {"bike": ModeMatrices(time),
            "car": ModeMatrices(time * 0.5, cost, "carcost")}
    base = run(pop, jobs, wfh, wage, mats).table
    bike = pd.Series([0.9, 0.8, 0.5], index=ORIGINS)
    car = pd.DataFrame([(h, f"D{i}", 0.5 if h == "single" else 1.0)
                        for h in ("single", "couple", "single_parent",
                                  "couple_children") for i in range(1, 11)],
                       columns=["household_type", "income_class", "share"])
    av = availability_frames(ORIGINS, NAMES, bike_share=bike, car_table=car)
    t = run(pop, jobs, wfh, wage, mats, availability=av).table
    np.testing.assert_allclose(t["accessibility"], base["accessibility"])
    assert (base["availability"] == 1.0).all()
    b = t[t["mode"] == "bike"]
    np.testing.assert_allclose(
        b["accessibility_expected"],
        b["accessibility"] * b["buurtcode"].map(bike).to_numpy())
    c = t[t["mode"] == "car"]
    single = c["household_type"] == "single"
    assert (c.loc[single, "availability"] == 0.5).all()
    assert (c.loc[~single, "availability"] == 1.0).all()


def test_availability_out_of_range_rejected():
    pop, jobs, wfh, wage, time, cost = world()
    bad = pd.DataFrame(2.0, index=ORIGINS, columns=NAMES)
    with pytest.raises(ValueError, match="within"):
        run(pop, jobs, wfh, wage, {"bike": ModeMatrices(time)},
            availability={"bike": bad})


def test_m1_public_transport_vot_weighted_by_rail_share():
    pop, jobs, wfh, wage, time, cost = world()
    share = np.full(time.shape, 0.5, dtype=np.float32)
    mm = ModeMatrices(time, cost, "ptfare", rail_share=share)
    mixed = {"pt": 15.0, "pt_other": 10.0}
    weighted = run(pop, jobs, wfh, wage, {"pt": mm}, spec="m1",
                   vot=mixed).table["accessibility"]
    # a uniform share of 0.5 is the same as one VoT of 12.5
    single = run(pop, jobs, wfh, wage,
                 {"pt": ModeMatrices(time, cost, "ptfare")}, spec="m1",
                 vot={"pt": 12.5}).table["accessibility"]
    np.testing.assert_allclose(weighted, single, rtol=1e-4)


# ── alternative journeys (option sets) ───────────────────────────────

from ikob2.run.accessibility import MixedMode, OptionSet  # noqa: E402


def acc(res, mode):
    t = res.table
    return t[t["mode"] == mode].sort_values(["segment", "buurtcode"]
                                            )["accessibility"].to_numpy()


def test_option_set_identical_or_dominated_options_add_nothing():
    pop, jobs, wfh, wage, time, cost = world()
    a = ModeMatrices(time, cost, "fare")
    slower_dearer = ModeMatrices(time + 5, cost + 3, "fare")
    res = run(pop, jobs, wfh, wage, {
        "single": OptionSet((a,), "car"),
        "twice": OptionSet((a, a), "car"),
        "dominated": OptionSet((a, slower_dearer), "car")})
    np.testing.assert_allclose(acc(res, "twice"), acc(res, "single"),
                               rtol=1e-4)
    np.testing.assert_allclose(acc(res, "dominated"), acc(res, "single"),
                               rtol=1e-4)


def test_option_set_tradeoff_is_inclusion_exclusion():
    pop, jobs, wfh, wage, time, cost = world()
    slow_cheap = ModeMatrices(time * 1.6, cost * 0.3, "fare")
    fast_dear = ModeMatrices(time, cost * 1.5, "fare")
    both_bad = ModeMatrices(time * 1.6, cost * 1.5, "fare")   # (t_slow, c_dear)
    res = run(pop, jobs, wfh, wage, {
        "slow_cheap": OptionSet((slow_cheap,), "car"),
        "fast_dear": OptionSet((fast_dear,), "car"),
        "both": OptionSet((both_bad,), "car"),
        "union": OptionSet((slow_cheap, fast_dear), "car")})
    union = acc(res, "union")
    expected = acc(res, "slow_cheap") + acc(res, "fast_dear") - acc(res, "both")
    np.testing.assert_allclose(union, expected, rtol=1e-4, atol=1e-3)
    assert (union >= np.maximum(acc(res, "slow_cheap"),
                                acc(res, "fast_dear")) - 1e-3).all()
    assert (union <= acc(res, "slow_cheap") + acc(res, "fast_dear") + 1e-3).all()


def test_mixed_mode_weights_the_option_sets_per_origin():
    pop, jobs, wfh, wage, time, cost = world()
    plain = ModeMatrices(time * 1.6, cost * 0.3, "fare")
    fast = ModeMatrices(time, cost * 1.5, "fare")
    only_plain = OptionSet((plain,), "car")
    both = OptionSet((plain, fast), "car")
    w = np.array([1.0, 0.0, 0.25])
    res = run(pop, jobs, wfh, wage, {
        "a": only_plain, "b": both,
        "mix": MixedMode(((1 - w, only_plain), (w, both)))})
    t = res.table
    piv = {m: t[t["mode"] == m].pivot(index="buurtcode", columns="segment",
                                      values="accessibility") for m in
           ("a", "b", "mix")}
    for i, o in enumerate(ORIGINS):
        np.testing.assert_allclose(
            piv["mix"].loc[o], (1 - w[i]) * piv["a"].loc[o]
            + w[i] * piv["b"].loc[o], rtol=1e-4, atol=1e-3)


@pytest.mark.parametrize("spec,extra", [
    ("m0u", {"cost_cutoff_eur": 12.0}), ("m0s", {}),
    ("m1", {"vot": {"car": 12.0}}), ("m1p", {}), ("m2", {}),
    ("m3", {"theta": 2.0})])
def test_one_option_equals_the_plain_mode_under_every_specification(spec, extra):
    pop, jobs, wfh, wage, time, cost = world()
    a = ModeMatrices(time, cost, "fare")
    res = run(pop, jobs, wfh, wage, {"car": a, "set": OptionSet((a,), "car")},
              spec=spec, **extra)
    np.testing.assert_allclose(acc(res, "set"), acc(res, "car"),
                               rtol=1e-4, atol=1e-2)


def test_option_set_bad_weights():
    pop, jobs, wfh, wage, time, cost = world()
    a = ModeMatrices(time, cost, "fare")
    with pytest.raises(ValueError, match="share"):
        run(pop, jobs, wfh, wage, {"x": MixedMode(
            ((np.array([2.0, 0, 0]), OptionSet((a,), "car")),))})


# ── leg-wise gates ───────────────────────────────────────────────────

from ikob2.run.accessibility import LegOption, LegOptionSet  # noqa: E402


def test_legwise_single_leg_equals_the_plain_gate():
    pop, jobs, wfh, wage, time, cost = world()
    res = run(pop, jobs, wfh, wage, {
        "car": ModeMatrices(time, cost, "fare"),
        "leg": LegOptionSet((LegOption((time,), cost, "fare"),), ("car",))})
    np.testing.assert_allclose(acc(res, "leg"), acc(res, "car"),
                               rtol=2e-4, atol=1e-2)


def test_legwise_is_more_lenient_than_one_gate_on_the_total():
    pop, jobs, wfh, wage, time, cost = world()
    half = time / 2
    res = run(pop, jobs, wfh, wage, {
        "car": ModeMatrices(time, cost, "fare"),
        "leg": LegOptionSet((LegOption((half, half), cost, "fare"),),
                            ("car", "car"))})
    assert (acc(res, "leg") >= acc(res, "car") - 1e-2).all()
    assert acc(res, "leg").sum() > acc(res, "car").sum()


def test_legwise_union_inclusion_exclusion_and_identical_options():
    pop, jobs, wfh, wage, time, cost = world()
    a = LegOption((time * 0.4, time * 0.6), cost * 0.3, "fare")
    b = LegOption((time * 0.2, time * 0.9), cost * 1.5, "fare")
    ab = LegOption((time * 0.4, time * 0.9), cost * 1.5, "fare")
    res = run(pop, jobs, wfh, wage, {
        "a": LegOptionSet((a,), ("car", "car")),
        "b": LegOptionSet((b,), ("car", "car")),
        "ab": LegOptionSet((ab,), ("car", "car")),
        "aa": LegOptionSet((a, a), ("car", "car")),
        "u": LegOptionSet((a, b), ("car", "car"))})
    np.testing.assert_allclose(acc(res, "aa"), acc(res, "a"), rtol=2e-4,
                               atol=1e-2)
    np.testing.assert_allclose(
        acc(res, "u"), acc(res, "a") + acc(res, "b") - acc(res, "ab"),
        rtol=2e-4, atol=1e-2)


@pytest.mark.parametrize("spec,extra", [
    ("m0u", {"cost_cutoff_eur": 12.0}), ("m0s", {}),
    ("m1", {"vot": {"car": 12.0}}), ("m1p", {}), ("m3", {"theta": 2.0}),
    ("m3", {"theta": float("inf")})])
def test_legwise_single_leg_equals_the_plain_gate_under_every_specification(
        spec, extra):
    pop, jobs, wfh, wage, time, cost = world()
    res = run(pop, jobs, wfh, wage, {
        "car": ModeMatrices(time, cost, "fare"),
        "leg": LegOptionSet((LegOption((time,), cost, "fare"),), ("car",))},
        spec=spec, **extra)
    np.testing.assert_allclose(acc(res, "leg"), acc(res, "car"),
                               rtol=2e-4, atol=1e-2)


# ── the ladder: M0 (isochrone in generalised time) and M1c ───────────

def test_m0_is_a_step_in_generalised_time_at_the_median_acceptable_time():
    from ikob2.segments.specs import median_time, time_margin_for
    pop, jobs, wfh, wage, time, cost = world()
    vot = {"car": 12.0}
    res = run(pop, jobs, wfh, wage, {"car": ModeMatrices(time, cost, "fare")},
              spec="m0", vot=vot)
    # hand computation: the income-matched jobs within the median acceptable
    # time T*_w of each job type, in generalised minutes t + 60 c / VoT
    med = {w: median_time(MARGINS[("car", w)]) for w in WFH_TYPES}
    assert med["no_wfh"] > 0
    g = time.astype(float) + cost.astype(float) * 60.0 / vot["car"]
    W = sector_income_weights(wage, jobs.sum())
    no, yes = split_jobs_by_wfh(jobs, wfh)
    expected = np.zeros(3)
    for wtype, j in (("no_wfh", no), ("wfh_possible", yes)):
        pool = sector_pools(j, DESTS, W)["D5"].astype(float)
        expected += fam.survival("step", (med[wtype],), g) @ pool
    got = res.table[(res.table["mode"] == "car")
                    & (res.table.segment == "couple_D5")].sort_values(
        "buurtcode")["accessibility"].to_numpy()
    np.testing.assert_allclose(got, expected, rtol=1e-4)
    assert 0 < expected.sum() < sum(
        sector_pools(j, DESTS, W)["D5"].sum() * 3 for j in (no, yes))
    # the same run with the cost far above the value of time removes access
    dear = run(pop, jobs, wfh, wage,
               {"car": ModeMatrices(time, cost * 1000, "fare")},
               spec="m0", vot=vot)
    assert dear.table["accessibility"].sum() < res.table["accessibility"].sum()
    assert time_margin_for("m0", MARGINS[("car", "no_wfh")]).curve == "step"


def test_m0_option_set_takes_the_least_generalised_time():
    pop, jobs, wfh, wage, time, cost = world()
    a = ModeMatrices(time, cost, "fare")
    fast_dear = ModeMatrices(time * 0.2, cost * 40, "fare")
    res = run(pop, jobs, wfh, wage, {
        "a": OptionSet((a,), "car"), "u": OptionSet((a, fast_dear), "car"),
        "b": OptionSet((fast_dear,), "car")}, spec="m0", vot={"car": 12.0})
    sa, su, sb = acc(res, "a"), acc(res, "u"), acc(res, "b")
    assert (su >= np.maximum(sa, sb) - 1e-6).all()      # the union is at least each


def test_m1c_uses_one_calibrated_exponential_cost_margin():
    from ikob2.segments.specs import (cost_curve_factory, implied_vot,
                                      median_segment_cost, median_time)
    curve = cost_curve_factory("m1c", MARGINS[("car", "no_wfh")], None, 40.0)(None)
    assert curve.curve == "exponential" and curve.params[0] == pytest.approx(1 / 40.0)
    with pytest.raises(ValueError, match="calibrated"):
        cost_curve_factory("m1c", MARGINS[("car", "no_wfh")], None, None)
    m = median_time(MARGINS[("car", "no_wfh")])
    k, eta = MARGINS[("car", "no_wfh")].params
    assert m == pytest.approx(eta * np.log(2) ** (1 / k), rel=1e-6)
    assert median_segment_cost(ENV) > 0
    assert implied_vot(30.0, MARGINS[("car", "no_wfh")]) > 0
    pop, jobs, wfh, wage, time, cost = world()
    res = run(pop, jobs, wfh, wage, {"car": ModeMatrices(time, cost, "fare")},
              spec="m1c", cost_mean_eur=40.0)
    tab = res.table
    assert (tab["atom"] == 0).all()        # no atom under the exponential specs
    # decile 1 keeps its jobs (a shared margin, no censoring)
    assert tab[tab.income_class == "D1"]["accessibility"].sum() > 0


# ── dual cut-offs: M0u (one cost cut-off) and M0s (one per segment) ──

def _dual_cutoff_by_hand(time_cost_pairs, seg, c_star, jobs, wfh, wage):
    """Income-matched jobs of segment `seg` (its class) for which some
    (time, cost) option is within the median acceptable time and the cost
    cut-off `c_star`."""
    from ikob2.segments.specs import median_time
    W = sector_income_weights(wage, jobs.sum())
    no, yes = split_jobs_by_wfh(jobs, wfh)
    expected = np.zeros(3)
    for wtype, j in (("no_wfh", no), ("wfh_possible", yes)):
        t_star = median_time(MARGINS[("car", wtype)])
        ok = np.zeros((3, 4), dtype=bool)
        for t, c in time_cost_pairs:
            ok |= (t <= t_star) & (c <= c_star)
        pool = sector_pools(j, DESTS, W)[seg.rsplit("_", 1)[1]].astype(float)
        expected += ok.astype(float) @ pool
    return expected


def _segment(res, mode, seg):
    t = res.table
    return t[(t["mode"] == mode) & (t.segment == seg)].sort_values(
        "buurtcode")


def test_m0s_is_a_step_on_each_margin_at_the_segment_median():
    from ikob2.segments.specs import cutoff_cost
    pop, jobs, wfh, wage, time, cost = world()
    res = run(pop, jobs, wfh, wage, {"car": ModeMatrices(time, cost, "fare")},
              spec="m0s")
    for seg in ("couple_D3", "single_D5", "couple_children_D10"):
        r = next(ENV[(ENV.household_type + "_" + ENV.income_class) == seg]
                 .itertuples())
        c_star = cutoff_cost([r], 0.5)
        assert c_star == pytest.approx(
            0.5 * (r.low + r.high) if r.atom == 0 else 0.0)
        expected = _dual_cutoff_by_hand([(time, cost)], seg, c_star,
                                        jobs, wfh, wage)
        got = _segment(res, "car", seg)
        np.testing.assert_allclose(got["accessibility"], expected, rtol=1e-4)
        assert (got["atom"] == 0.0).all()
    # decile 1 (atom 1): cut-off zero, only the free pair (O0, D0) counts
    d1 = _segment(res, "car", "single_D1")
    np.testing.assert_allclose(
        d1["accessibility"],
        _dual_cutoff_by_hand([(time, cost)], "single_D1", 0.0, jobs, wfh, wage),
        rtol=1e-4)
    assert d1["accessibility"].iloc[0] > 0 and (d1["accessibility"].iloc[1:] == 0).all()
    assert (d1["atom"] == 1.0).all() and d1["accessibility_normalised"].isna().all()


def test_m0u_one_cost_cutoff_for_every_segment():
    pop, jobs, wfh, wage, time, cost = world()
    res = run(pop, jobs, wfh, wage, {"car": ModeMatrices(time, cost, "fare")},
              spec="m0u", cost_cutoff_eur=12.0)
    for seg in ("single_D1", "couple_D3", "couple_children_D10"):
        got = _segment(res, "car", seg)
        np.testing.assert_allclose(
            got["accessibility"],
            _dual_cutoff_by_hand([(time, cost)], seg, 12.0, jobs, wfh, wage),
            rtol=1e-4)
    assert (res.table["atom"] == 0).all()        # no atom: f(0, 0) = 1


def test_dual_cutoff_union_and_generalised_time_differ_by_hand():
    """A fast dear and a slow cheap option: under M0s a pair counts when one
    of them passes both cut-offs; under M0 when the least generalised time
    is within T*. The fast dear option compensates its fare under M0 only."""
    from ikob2.segments.specs import cutoff_cost, median_time
    pop, jobs, wfh, wage, time, cost = world()
    seg = "couple_D3"
    r = next(ENV[(ENV.household_type + "_" + ENV.income_class) == seg]
             .itertuples())
    c_star = cutoff_cost([r], 0.5)
    t_star = median_time(MARGINS[("car", "no_wfh")])
    fast_dear = (np.full((3, 4), 0.2 * t_star, np.float32),
                 np.full((3, 4), 1.5 * c_star, np.float32))
    slow_cheap = (np.full((3, 4), 2.0 * t_star, np.float32),
                  np.full((3, 4), 0.5 * c_star, np.float32))
    opts = OptionSet(tuple(ModeMatrices(t, c, "fare")
                           for t, c in (fast_dear, slow_cheap)), "car")
    dual = _segment(run(pop, jobs, wfh, wage, {"u": opts}, spec="m0s"),
                    "u", seg)["accessibility"].to_numpy()
    # neither option passes both cut-offs (no-wfh jobs; the wfh margin is
    # longer, so the slow option may pass there)
    np.testing.assert_allclose(
        dual, _dual_cutoff_by_hand([fast_dear, slow_cheap], seg, c_star,
                                   jobs, wfh, wage), rtol=1e-4)
    # generalised time of the fast option at a VoT that makes its fare cost
    # 0.3 T*: 0.2 T* + 0.3 T* < T*, so every pair counts under M0
    vot = 1.5 * c_star * 60.0 / (0.3 * t_star)
    gc = _segment(run(pop, jobs, wfh, wage, {"u": opts}, spec="m0",
                      vot={"car": vot}), "u", seg)["accessibility"].to_numpy()
    W = sector_income_weights(wage, jobs.sum())
    all_jobs = sum(sector_pools(j, DESTS, W)["D3"].sum()
                   for j in split_jobs_by_wfh(jobs, wfh))
    np.testing.assert_allclose(gc, all_jobs, rtol=1e-4)
    assert (dual < gc - 1.0).all()
