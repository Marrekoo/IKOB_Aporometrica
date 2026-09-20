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
