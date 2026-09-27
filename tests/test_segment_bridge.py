"""Household-type x income segments in the accessibility engine."""

import numpy as np
import pandas as pd
import pytest

from ikob2.core.decay_curves import uniform, weibull
from ikob2.core.numerics import DTYPE
from ikob2.domain.filter_config import CopulaSpec, CurveSpec
from ikob2.domain.state import ModelState
from ikob2.engine.runner import SegmentedRunner
from ikob2.segments.bridge import (
    aggregate_by,
    build_segments,
    load_envelope,
    populations_for_zones,
    segment_name,
    validate_envelope,
)
from ikob2.segments.config import HOUSEHOLD_TYPES, INCOME_CLASSES, SegmentConfig
from ikob2.segments.pipeline import compute_segments

from test_segments import make_children_raw, make_income_raw, make_kwb

TIME = CurveSpec("weibull", (2.0, 45.0))
FARE = "fare(test)"


def full_envelope(low=5.0, high=15.0, atom=0.0):
    rows = [(t, c, low, high, atom) for c in INCOME_CLASSES
            for t in HOUSEHOLD_TYPES]
    return pd.DataFrame(rows, columns=["household_type", "income_class",
                                       "low", "high", "atom"])


# ── Envelope table ───────────────────────────────────────────────────

def test_envelope_defaults_atom_and_is_returned_as_copy():
    env = full_envelope().drop(columns="atom")
    out = validate_envelope(env)
    assert (out["atom"] == 0).all() and "atom" not in env.columns


@pytest.mark.parametrize("mutate,match", [
    (lambda d: d.drop(columns="high"), "lacks column"),
    (lambda d: d.assign(household_type="triple"), "unknown household"),
    (lambda d: d.assign(income_class="D99"), "unknown household type"),
    (lambda d: pd.concat([d, d.iloc[:1]]), "duplicate"),
    (lambda d: d.assign(low=np.where(d.index == 0, 20.0, d.low)),
     r"0 <= low <= high"),
    (lambda d: d.assign(low=-1.0), r"0 <= low <= high"),
    (lambda d: d.assign(atom=1.5), "atom must be"),
    (lambda d: d.assign(high=np.where(d.index == 0, np.nan, d.high)),
     "missing/non-finite"),
])
def test_envelope_validation_rejects(mutate, match):
    with pytest.raises(ValueError, match=match):
        validate_envelope(mutate(full_envelope()))


def test_load_envelope_roundtrip(tmp_path):
    path = tmp_path / "env.csv"
    full_envelope(2.0, 9.0, 0.1).to_csv(path, index=False)
    env = load_envelope(path)
    assert len(env) == 44 and (env.high == 9.0).all()


# ── Segment construction ─────────────────────────────────────────────

def test_time_only_segments_share_one_filter():
    segs = build_segments(TIME)
    assert len(segs) == 44
    assert [s.name for s in segs][:5] == [
        "single_D1", "couple_D1", "single_parent_D1",
        "couple_children_D1", "single_D2"]
    assert len({s.weight_key for s in segs}) == 1
    assert all(s.money_cost_id is None and s.pool == "default" for s in segs)
    s = segs[5]
    assert (s.household_type, s.income) == ("couple", "D2")


def test_segments_with_envelope_carry_their_own_cost_margin():
    env = full_envelope()
    env.loc[env.income_class == "D1", ["low", "high", "atom"]] = [1.0, 4.0, 0.3]
    segs = build_segments(TIME, envelope=env, money_cost_id=FARE,
                          copula=CopulaSpec("gumbel", 2.0))
    d1 = next(s for s in segs if s.name == "single_D1")
    d5 = next(s for s in segs if s.name == "single_D5")
    assert d1.class_filter.cost == CurveSpec("uniform", (1.0, 4.0), atom=0.3)
    assert d5.class_filter.cost == CurveSpec("uniform", (5.0, 15.0))
    assert d1.class_filter.copula == CopulaSpec("gumbel", 2.0)
    assert d1.money_cost_id == FARE
    # 4 D1 segments share one filter, the other 40 another
    assert len({s.weight_key for s in segs}) == 2


def test_build_segments_misconfiguration():
    with pytest.raises(ValueError, match="needs money_cost_id"):
        build_segments(TIME, envelope=full_envelope())
    with pytest.raises(ValueError, match="without an envelope"):
        build_segments(TIME, money_cost_id=FARE)
    with pytest.raises(KeyError, match="no row for"):
        build_segments(TIME, envelope=full_envelope().iloc[:-3],
                       money_cost_id=FARE)
    with pytest.raises(KeyError, match="Unknown segment"):
        build_segments(TIME, only=["nope_D1"])
    with pytest.raises(ValueError, match="Unknown pool_by"):
        build_segments(TIME, pool_by="bogus")


def test_only_restricts_and_missing_rows_then_allowed():
    env = full_envelope().iloc[:6]
    only = [segment_name(r.household_type, r.income_class)
            for r in env.itertuples()]
    segs = build_segments(TIME, envelope=env, money_cost_id=FARE, only=only)
    assert {s.name for s in segs} == set(only)
    assert len(segs) == 6


def test_pools():
    by_income = build_segments(TIME, pool_by="income_class")
    assert {s.pool for s in by_income} == set(INCOME_CLASSES)
    custom = build_segments(TIME, pool_by=lambda t, c: t)
    assert {s.pool for s in custom} == set(HOUSEHOLD_TYPES)


# ── Populations ──────────────────────────────────────────────────────

def _table():
    return pd.DataFrame({"buurtcode": ["BU1", "BU2", "BU3"],
                         "single_D1": [10.0, np.nan, 30.0],
                         "couple_D1": [1.0, 2.0, 3.0]})


def test_populations_align_to_engine_order_and_zero_fill():
    segs = build_segments(TIME, only=["single_D1", "couple_D1"])
    pops = populations_for_zones(_table(), ["BU3", "BU9", "BU1"], segs)
    np.testing.assert_array_equal(pops["single_D1"], [30.0, 0.0, 10.0])
    np.testing.assert_array_equal(pops["couple_D1"], [3.0, 0.0, 1.0])
    assert pops["single_D1"].dtype == np.float32
    nan_zone = populations_for_zones(_table(), ["BU2"], segs)
    assert nan_zone["single_D1"][0] == 0.0


def test_populations_errors():
    segs = build_segments(TIME, only=["single_D1", "single_D2"])
    with pytest.raises(KeyError, match="lacks segment"):
        populations_for_zones(_table(), ["BU1"], segs)
    segs = build_segments(TIME, only=["single_D1"])
    with pytest.raises(ValueError, match="duplicates"):
        populations_for_zones(_table(), ["BU1", "BU1"], segs)
    dup = pd.concat([_table(), _table().iloc[:1]])
    with pytest.raises(ValueError, match="duplicate buurtcodes"):
        populations_for_zones(dup, ["BU1"], segs)


# ── Hansen through the engine ────────────────────────────────────────

def small_state(n=4, seed=0):
    rng = np.random.default_rng(seed)
    time = rng.uniform(5, 80, (n, n)).astype(DTYPE)
    np.fill_diagonal(time, 0.0)
    money = rng.uniform(1, 20, (n, n)).astype(DTYPE)
    np.fill_diagonal(money, 0.0)
    jobs = rng.uniform(10, 100, n).astype(DTYPE)
    state = ModelState.create(
        generalized_cost=time, population=np.ones(n), opportunities=jobs,
        decay_type="exponential", decay_beta=0.05, decay_epsilon=0.0)
    return state, time, money, jobs


def test_hansen_equals_hand_computed_gate():
    state, time, money, jobs = small_state()
    env = full_envelope()
    env.loc[env.income_class == "D2", ["low", "high", "atom"]] = [2.0, 8.0, 0.25]
    segs = build_segments(TIME, envelope=env, money_cost_id=FARE,
                          only=["single_D1", "single_D2"])
    out = SegmentedRunner(decay_epsilon=None).run_hansen(
        state, segs, cost_matrices={"time": time, FARE: money})

    st = weibull(time, 2.0, 45.0)
    expected1 = (st * uniform(money, 5.0, 15.0)) @ jobs
    sm2 = uniform(money, 2.0, 8.0)
    sm2 = np.where(money > 0, sm2 * 0.75, sm2)
    expected2 = (st * sm2) @ jobs
    np.testing.assert_allclose(out["single_D1"], expected1, rtol=1e-5)
    np.testing.assert_allclose(out["single_D2"], expected2, rtol=1e-5)


def test_hansen_time_only_segments_all_equal():
    state, time, _, jobs = small_state()
    segs = build_segments(TIME)
    out = SegmentedRunner(decay_epsilon=None).run_hansen(
        state, segs, cost_matrices={"time": time})
    assert len(out) == 44
    expected = weibull(time, 2.0, 45.0) @ jobs
    for a in out.values():
        np.testing.assert_allclose(a, expected, rtol=1e-5)


def test_hansen_atom_one_leaves_only_free_trips():
    state, time, money, jobs = small_state()
    env = full_envelope(atom=1.0)
    segs = build_segments(TIME, envelope=env, money_cost_id=FARE,
                          only=["single_D1"])
    a = SegmentedRunner(decay_epsilon=None).run_hansen(
        state, segs, cost_matrices={"time": time, FARE: money})["single_D1"]
    # only intra-zonal (cost 0) pairs survive
    expected = np.diag(weibull(time, 2.0, 45.0)) * jobs
    np.testing.assert_allclose(a, expected, rtol=1e-5)


def test_hansen_pools_use_their_own_opportunities():
    state, time, _, jobs = small_state()
    segs = build_segments(TIME, pool_by="income_class",
                          only=["single_D1", "single_D2"])
    other = jobs[::-1].copy()
    out = SegmentedRunner(decay_epsilon=None).run_hansen(
        state, segs, cost_matrices={"time": time},
        opportunities={"D1": jobs, "D2": other})
    w = weibull(time, 2.0, 45.0)
    np.testing.assert_allclose(out["single_D1"], w @ jobs, rtol=1e-5)
    np.testing.assert_allclose(out["single_D2"], w @ other, rtol=1e-5)
    with pytest.raises(ValueError, match="validation failed"):
        SegmentedRunner(decay_epsilon=None).run_hansen(
            state, segs, cost_matrices={"time": time},
            opportunities={"D1": jobs})


def test_hansen_rerun_sees_new_cost_and_leaks_no_pins():
    state, time, money, _ = small_state()
    segs = build_segments(TIME, only=["single_D1"])
    runner = SegmentedRunner(decay_epsilon=None)
    base = runner.run_hansen(state, segs)["single_D1"]
    scaled = state.with_updates(
        generalized_cost=(state.generalized_cost * 2.0).astype(np.float32))
    varied = runner.run_hansen(scaled, segs)["single_D1"]
    assert not np.allclose(base, varied)
    fresh = SegmentedRunner(decay_epsilon=None).run_hansen(
        scaled, segs)["single_D1"]
    np.testing.assert_array_equal(varied, fresh)
    assert runner.registry.pinned_size_mb() == 0.0


def test_hansen_needs_money_matrix_for_cost_filters():
    state, time, _, _ = small_state()
    segs = build_segments(TIME, envelope=full_envelope(),
                          money_cost_id=FARE, only=["single_D1"])
    with pytest.raises(KeyError, match="money cost"):
        SegmentedRunner(decay_epsilon=None).run_hansen(
            state, segs, cost_matrices={"time": time})


# ── From pipeline output to engine ───────────────────────────────────

def test_pipeline_output_drives_the_shen_runner_end_to_end():
    kwb = make_kwb(n_gm=12, per_gm=3)
    result = compute_segments(kwb, make_income_raw(12), make_children_raw(12),
                              SegmentConfig())
    zone_codes = list(kwb.buurtcode)
    n = len(zone_codes)
    rng = np.random.default_rng(5)
    time = rng.uniform(5, 90, (n, n)).astype(DTYPE)
    np.fill_diagonal(time, 0.0)
    money = rng.uniform(0.5, 12, (n, n)).astype(DTYPE)
    np.fill_diagonal(money, 0.0)
    jobs = rng.uniform(10, 500, n).astype(DTYPE)
    state = ModelState.create(
        generalized_cost=time,
        population=kwb.inwoners.to_numpy(),
        opportunities=jobs, decay_type="exponential", decay_beta=0.05,
        decay_epsilon=0.0)

    segs = build_segments(TIME, envelope=full_envelope(),
                          money_cost_id=FARE)
    pops = populations_for_zones(result.population_scaled, zone_codes, segs)
    runner = SegmentedRunner(decay_epsilon=1e-9)
    res = runner.run(state, segs, pops,
                     cost_matrices={"time": time, FARE: money})

    assert len(res.per_segment) == 44
    assert all(np.all(np.isfinite(a)) for a in res.per_segment.values())
    total_pop = sum(p.astype(np.float64) for p in pops.values())
    np.testing.assert_allclose(res.total_population, total_pop, rtol=1e-5)
    np.testing.assert_allclose(total_pop, kwb.inwoners, rtol=1e-4)

    # Shen conservation across all segments (single pool): the
    # population-weighted accessibilities add up to the job total
    weighted = sum(res.per_segment[s.name].astype(np.float64)
                   * pops[s.name] for s in segs)
    # epsilon truncation makes this approximate
    assert weighted.sum() == pytest.approx(float(jobs.sum()), rel=1e-3)

    hansen = runner.run_hansen(state, segs,
                               cost_matrices={"time": time, FARE: money})
    assert set(hansen) == set(res.per_segment)
    # competition can only reduce what a resident gets vs potential access
    # in total (Shen shares each job among competing residents)
    tot_h = sum(hansen[s.name].astype(np.float64) * pops[s.name] for s in segs)
    assert tot_h.sum() > weighted.sum()


# ── Aggregation ──────────────────────────────────────────────────────

def test_aggregate_by_population_weighted_mean():
    segs = build_segments(TIME, only=["single_D1", "couple_D1", "single_D2"])
    per = {"single_D1": np.array([10.0, 0.0]),
           "couple_D1": np.array([20.0, 0.0]),
           "single_D2": np.array([5.0, 8.0])}
    pop = {"single_D1": np.array([1.0, 0.0]),
           "couple_D1": np.array([3.0, 0.0]),
           "single_D2": np.array([2.0, 4.0])}
    by_income = aggregate_by(per, pop, segs, "income_class")
    np.testing.assert_allclose(by_income["D1"][0], (10 * 1 + 20 * 3) / 4)
    assert np.isnan(by_income["D1"][1])          # no D1 population in zone 1
    np.testing.assert_allclose(by_income["D2"], [5.0, 8.0])
    by_hh = aggregate_by(per, pop, segs, "household_type")
    np.testing.assert_allclose(by_hh["single"][0], (10 * 1 + 5 * 2) / 3)
    allv = aggregate_by(per, pop, segs, "all")["all"]
    np.testing.assert_allclose(allv[0], (10 + 60 + 10) / 6)
    with pytest.raises(ValueError, match="Unknown key"):
        aggregate_by(per, pop, segs, "bogus")


# ── Rectangular Hansen (origins x destinations) ──────────────────────

def rect_problem(n_o=3, n_d=7, seed=2):
    rng = np.random.default_rng(seed)
    time = rng.uniform(3, 90, (n_o, n_d)).astype(DTYPE)
    money = rng.uniform(0.5, 18, (n_o, n_d)).astype(DTYPE)
    jobs = rng.uniform(5, 200, n_d).astype(DTYPE)
    return time, money, jobs


def test_rectangular_matches_hand_computation():
    time, money, jobs = rect_problem()
    env = full_envelope(4.0, 12.0, 0.2)
    segs = build_segments(TIME, envelope=env, money_cost_id=FARE,
                          only=["single_D1", "couple_D4"])
    out = SegmentedRunner(decay_epsilon=None).run_hansen(
        None, segs, cost_matrices={"time": time, FARE: money},
        opportunities={"default": jobs})
    sm = np.where(money > 0, uniform(money, 4.0, 12.0) * 0.8,
                  uniform(money, 4.0, 12.0))
    expected = (weibull(time, 2.0, 45.0) * sm) @ jobs
    assert out["single_D1"].shape == (3,)
    np.testing.assert_allclose(out["single_D1"], expected, rtol=1e-5)
    np.testing.assert_allclose(out["couple_D4"], expected, rtol=1e-5)


def test_rectangular_equals_rows_of_the_square_run():
    # origins = subset of the zones; result must be the corresponding
    # rows of the full square computation
    state, time, money, jobs = small_state(n=9, seed=4)
    segs = build_segments(TIME, envelope=full_envelope(),
                          money_cost_id=FARE, only=["single_D1"])
    runner = SegmentedRunner(decay_epsilon=None)
    full = runner.run_hansen(state, segs,
                             cost_matrices={"time": time, FARE: money})
    origins = np.array([1, 4, 7])
    rect = runner.run_hansen(
        None, segs,
        cost_matrices={"time": time[origins], FARE: money[origins]},
        opportunities={"default": jobs})
    np.testing.assert_allclose(rect["single_D1"], full["single_D1"][origins],
                               rtol=1e-6)


def test_rectangular_sparse_matrices_and_pools():
    from scipy import sparse
    time, money, jobs = rect_problem(n_o=4, n_d=12, seed=6)
    segs = build_segments(TIME, pool_by="income_class",
                          only=["single_D1", "single_D2"])
    other = jobs[::-1].copy()
    dense = SegmentedRunner(decay_epsilon=None).run_hansen(
        None, segs, cost_matrices={"time": time},
        opportunities={"D1": jobs, "D2": other})
    sp = SegmentedRunner(decay_epsilon=None).run_hansen(
        None, segs, cost_matrices={"time": sparse.csr_matrix(time)},
        opportunities={"D1": jobs, "D2": other})
    for name in ("single_D1", "single_D2"):
        np.testing.assert_allclose(sp[name], dense[name], rtol=1e-5)
    assert not np.allclose(dense["single_D1"], dense["single_D2"])


def test_rectangular_epsilon_sparsifies_far_destinations():
    time, _, jobs = rect_problem(n_o=2, n_d=6, seed=8)
    time[:, 3:] = 5000.0                       # unreachable destinations
    segs = build_segments(TIME, only=["single_D1"])
    a = SegmentedRunner(decay_epsilon=1e-6).run_hansen(
        None, segs, cost_matrices={"time": time},
        opportunities={"default": jobs})["single_D1"]
    near = (weibull(time[:, :3], 2.0, 45.0)) @ jobs[:3]
    np.testing.assert_allclose(a, near, rtol=1e-4)


def test_rectangular_misconfiguration():
    time, money, jobs = rect_problem()
    segs = build_segments(TIME, only=["single_D1"])
    runner = SegmentedRunner(decay_epsilon=None)
    with pytest.raises(ValueError, match="needs both"):
        runner.run_hansen(None, segs, cost_matrices={"time": time})
    with pytest.raises(ValueError, match="needs both"):
        runner.run_hansen(None, segs, opportunities={"default": jobs})
    with pytest.raises(ValueError, match="has shape \\(3, 7\\).*\\(3, 6\\)"):
        runner.run_hansen(None, segs, cost_matrices={"time": time},
                          opportunities={"default": jobs[:-1]})
    with pytest.raises(ValueError, match="disagree"):
        pooled = build_segments(TIME, pool_by="income_class",
                                only=["single_D1", "single_D2"])
        runner.run_hansen(None, pooled, cost_matrices={"time": time},
                          opportunities={"D1": jobs, "D2": jobs[:-1]})
    priced = build_segments(TIME, envelope=full_envelope(),
                            money_cost_id=FARE, only=["single_D1"])
    with pytest.raises(ValueError, match="has shape"):
        runner.run_hansen(None, priced,
                          cost_matrices={"time": time, FARE: money[:2]},
                          opportunities={"default": jobs})
    assert runner.registry.pinned_size_mb() == 0.0


def test_square_state_path_still_rejects_wrong_opportunity_length():
    state, time, _, jobs = small_state(n=4)
    segs = build_segments(TIME, only=["single_D1"])
    with pytest.raises(ValueError, match="expected"):
        SegmentedRunner(decay_epsilon=None).run_hansen(
            state, segs, cost_matrices={"time": time},
            opportunities={"default": jobs[:-1]})


# ── Reference budgets (Table 6) ──────────────────────────────────────

from ikob2.segments.bridge import (  # noqa: E402
    envelope_segment_names,
    load_reference_budgets,
)

BUDGETS = "data/envelope/reference_budgets_x_m_calc.csv"   # per home-based tour


def test_reference_budgets_match_the_published_table():
    env = load_reference_budgets(BUDGETS, legs_per_tour=1.0)
    assert len(env) == 40
    assert set(env.household_type) == {"single", "couple", "single_parent",
                                       "couple_children"}
    assert set(env.income_class) == {f"D{i}" for i in range(1, 11)}

    def row(t, c):
        return env[(env.household_type == t)
                   & (env.income_class == c)].iloc[0]

    # spot checks against the paper's Figure 5 excerpt and Table 6
    assert (row("couple", "D2").low, row("couple", "D2").high) == (4.87, 23.52)
    assert (row("couple_children", "D5").low,
            row("couple_children", "D5").high) == (43.32, 245.56)
    assert (row("single", "D2").low, row("single", "D2").high,
            row("single", "D2").km_low) == (1.85, 25.53, 0.0)
    assert (row("single_parent", "D10").low, row("single_parent", "D10").high,
            row("single_parent", "D10").km_high) == (191.79, 782.01, 4115.84)
    assert (row("couple", "D10").low, row("couple", "D10").high) == (224.30, 691.11)
    # the table is not monotone in income for single households (Q6 < Q5)
    assert row("single", "D6").low < row("single", "D5").low


def test_censored_first_decile_policies():
    atom = load_reference_budgets(BUDGETS, censored="atom")
    d1 = atom[atom.income_class == "D1"]
    assert len(d1) == 4 and (d1.atom == 1.0).all()
    assert (d1.low == 0).all() and (d1.high == 0).all()
    assert (atom[atom.income_class != "D1"].atom == 0.0).all()

    dropped = load_reference_budgets(BUDGETS, censored="drop")
    assert len(dropped) == 36 and "D1" not in set(dropped.income_class)
    with pytest.raises(ValueError, match="censored"):
        load_reference_budgets(BUDGETS, censored="error")
    with pytest.raises(ValueError, match="must be"):
        load_reference_budgets(BUDGETS, censored="maybe")


def test_reference_budgets_drive_segments_and_the_gate():
    env = load_reference_budgets(BUDGETS, legs_per_tour=1.0)
    names = envelope_segment_names(env)
    assert len(names) == 40 and "single_onbekend" not in names
    segs = build_segments(TIME, envelope=env, money_cost_id=FARE, only=names)
    assert len(segs) == 40
    q1 = next(s for s in segs if s.name == "single_D1")
    assert q1.class_filter.cost.atom == 1.0
    q5 = next(s for s in segs if s.name == "couple_D5")
    assert q5.class_filter.cost == CurveSpec("uniform", (23.16, 148.17))

    # a censored segment reaches only free (cost 0) trips; a middle
    # segment with budget 23-148 EUR clears a 6 EUR trip
    state, time, money, jobs = small_state()
    a = SegmentedRunner(decay_epsilon=None).run_hansen(
        state, [q1, q5], cost_matrices={"time": time, FARE: money})
    w = weibull(time, 2.0, 45.0)
    np.testing.assert_allclose(a["single_D1"], np.diag(w) * jobs, rtol=1e-5)
    np.testing.assert_allclose(a["couple_D5"], w @ jobs, rtol=1e-5)


# ── Tour versus trip basis ───────────────────────────────────────────

from ikob2.segments.bridge import rescale_budgets  # noqa: E402


def test_default_basis_converts_per_tour_to_per_journey():
    # the table is per home-based tour; the default divides by 2.2 journeys
    env = load_reference_budgets(BUDGETS)
    raw = pd.read_csv(BUDGETS)
    d2 = env[(env.household_type == "couple") & (env.income_class == "D2")].iloc[0]
    assert (d2.low, d2.high) == (pytest.approx(4.87 / 2.2), pytest.approx(23.52 / 2.2))
    assert env.attrs["legs_per_tour"] == 2.2
    assert len(env) == len(raw)
    one = load_reference_budgets(BUDGETS, legs_per_tour=1.0)
    d2 = one[(one.household_type == "couple") & (one.income_class == "D2")].iloc[0]
    assert (d2.low, d2.high) == (4.87, 23.52)          # 1: the table as it is


def test_legs_per_tour_divides_bounds_and_km_but_not_atom():
    base = load_reference_budgets(BUDGETS, legs_per_tour=1.0)
    two = load_reference_budgets(BUDGETS, legs_per_tour=2.0)
    for col in ("low", "high", "km_low", "km_high"):
        np.testing.assert_allclose(two[col], base[col] / 2.0)
    np.testing.assert_array_equal(two["atom"], base["atom"])
    assert two.attrs["legs_per_tour"] == 2.0
    assert (two[two.income_class == "D1"].atom == 1.0).all()


def test_legs_per_tour_per_household_type():
    legs = {"single": 1.5, "couple": 2.0, "single_parent": 2.5,
            "couple_children": 3.0}
    env = load_reference_budgets(BUDGETS, legs_per_tour=legs)
    base = load_reference_budgets(BUDGETS, legs_per_tour=1.0)
    for t, n in legs.items():
        sel = env.household_type == t
        np.testing.assert_allclose(env.loc[sel, "high"],
                                   base.loc[sel, "high"] / n)
    with pytest.raises(KeyError, match="couple_children"):
        load_reference_budgets(BUDGETS, legs_per_tour={
            "single": 1.0, "couple": 1.0, "single_parent": 1.0})


@pytest.mark.parametrize("bad", [0.0, -1.0, np.nan, np.inf])
def test_legs_per_tour_validation(bad):
    with pytest.raises(ValueError, match="positive and finite"):
        load_reference_budgets(BUDGETS, legs_per_tour=bad)


def test_rescaled_budget_changes_who_clears_a_fare():
    # a 40 EUR one-way trip: cleared by D5 couples (23-148 EUR) when the
    # table is read per trip, but only partly once it is per 3-leg tour
    env1 = load_reference_budgets(BUDGETS, legs_per_tour=1.0)
    env3 = rescale_budgets(env1, 3.0)
    s1 = build_segments(TIME, envelope=env1, money_cost_id=FARE,
                        only=["couple_D5"])[0].class_filter.cost
    s3 = build_segments(TIME, envelope=env3, money_cost_id=FARE,
                        only=["couple_D5"])[0].class_filter.cost
    c = np.array([[40.0]], dtype=DTYPE)
    assert uniform(c, *s1.params)[0, 0] > uniform(c, *s3.params)[0, 0]
    assert s3.params == pytest.approx((23.16 / 3, 148.17 / 3))


def test_budgets_per_journey_are_not_divided(tmp_path):
    t = pd.read_csv(BUDGETS)
    t["unit"] = "journey"
    p = tmp_path / "rb.csv"
    t.to_csv(p, index=False)
    env = load_reference_budgets(p)                    # default 2.2 is ignored
    raw = load_reference_budgets(BUDGETS, legs_per_tour=1.0)
    np.testing.assert_allclose(env["high"], raw["high"])
    t.loc[t.household_type == "single", "unit"] = "tour"
    t.to_csv(p, index=False)
    mixed = load_reference_budgets(p)
    s = mixed.household_type == "single"
    np.testing.assert_allclose(mixed.loc[s, "high"], raw.loc[s, "high"] / 2.2)
    np.testing.assert_allclose(mixed.loc[~s, "high"], raw.loc[~s, "high"])
