"""Run-versus-run comparison and the impedance-shape variants."""

import numpy as np
import pandas as pd
import pytest

from ikob2.cli.accessibility import time_curve
from ikob2.core import families as fam
from ikob2.run.accessibility import ModeMatrices, run_accessibility
from ikob2.run.compare import compare_runs

from test_run_accessibility import DESTS, ENV, NAMES, ORIGINS, world
from ikob2.segments.time_margins import load_time_margins


def _table(values, mode="car"):
    rows = []
    for i, o in enumerate(ORIGINS):
        for n in ("single_D2", "couple_D5"):
            rows.append({"mode": mode, "buurtcode": o, "segment": n,
                         "household_type": n.rsplit("_", 1)[0],
                         "income_class": n.rsplit("_", 1)[1],
                         "population": 10.0 + i,
                         "accessibility": values[(o, n)]})
    return pd.DataFrame(rows)


def test_identical_runs_agree_perfectly():
    vals = {(o, n): float(10 * i + j) for i, o in enumerate(ORIGINS)
            for j, n in enumerate(("single_D2", "couple_D5"))}
    t = _table(vals)
    res = compare_runs(t, t)
    c = res["cells"].iloc[0]
    assert c["pearson"] == pytest.approx(1) and c["spearman"] == pytest.approx(1)
    assert c["mean_ratio_b_over_a"] == pytest.approx(1.0)
    assert res["origins"].iloc[0]["spearman"] == pytest.approx(1)
    assert (res["income"]["ratio_b_over_a"] == 1).all()


def test_scaled_run_keeps_ranking_and_changes_levels():
    vals = {(o, n): float(3 + 7 * i + 2 * j) for i, o in enumerate(ORIGINS)
            for j, n in enumerate(("single_D2", "couple_D5"))}
    a = _table(vals)
    b = a.assign(accessibility=a.accessibility * 2.5)
    res = compare_runs(a, b)
    assert res["cells"].iloc[0]["spearman"] == pytest.approx(1)
    assert res["cells"].iloc[0]["mean_ratio_b_over_a"] == pytest.approx(2.5)
    np.testing.assert_allclose(res["income"]["ratio_b_over_a"], 2.5)


def test_reversed_ranking_is_detected_and_disjoint_runs_rejected():
    vals = {(o, n): float(i + 1) for i, o in enumerate(ORIGINS)
            for n in ("single_D2", "couple_D5")}
    a = _table(vals)
    b = a.assign(accessibility=1.0 / a.accessibility)
    assert compare_runs(a, b)["cells"].iloc[0]["spearman"] == pytest.approx(-1)
    with pytest.raises(ValueError, match="share no"):
        compare_runs(a, a.assign(buurtcode="X" + a.buurtcode))


def test_time_curves_for_the_shape_comparison():
    step = time_curve("step", 45)
    assert step.curve == "step" and step.params == (45.0,)
    mean = time_curve("exponential", 45)
    assert mean.params == (pytest.approx(1 / 45),)
    assert fam.mean_threshold("exponential", mean.params) == pytest.approx(45)
    half = time_curve("exponential", 45, "half")
    assert fam.survival("exponential", half.params,
                        np.array([45.0]))[0] == pytest.approx(0.5)
    assert fam.mean_threshold("step", step.params) == 45
    with pytest.raises(ValueError, match="Unknown"):
        time_curve("weibull", 45)


def test_time_only_variants_run_without_an_envelope_and_differ():
    pop, jobs, wfh, wage, time, cost = world()
    common = dict(origins=ORIGINS, destinations=DESTS, populations=pop,
                  sector_jobs=jobs, wfh_share=wfh, sector_wage=wage,
                  envelope=None, segment_names=NAMES, epsilon=None,
                  matrices={"car": ModeMatrices(time, cost, "c")})
    runs = {}
    for shape in ("step", "exponential"):
        c = time_curve(shape, 45)
        margins = {("car", w): c for w in ("no_wfh", "wfh_possible")}
        runs[shape] = run_accessibility(time_margins=margins, **common)
    step = runs["step"].table
    # cost matrices are ignored without an envelope
    assert (step["atom"] == 0).all()
    # hard cut-off: integer-valued combination of pooled jobs within 45 min
    from ikob2.segments.jobs import sector_income_weights, sector_pools
    from ikob2.segments.wfh import split_jobs_by_wfh
    W = sector_income_weights(wage, jobs.sum())
    exp = np.zeros(3)
    for j in split_jobs_by_wfh(jobs, wfh):
        pool = sector_pools(j, DESTS, W)["D5"].astype(float)
        exp += (time <= 45).astype(float) @ pool
    got = step[step.segment == "couple_D5"].sort_values("buurtcode")[
        "accessibility"].to_numpy()
    np.testing.assert_allclose(got, exp, rtol=1e-4)
    res = compare_runs(runs["step"].table, runs["exponential"].table)
    assert res["cells"].iloc[0]["cells"] == 3 * len(NAMES)
    with pytest.raises(ValueError, match="segment_names"):
        run_accessibility(**{**common, "segment_names": None},
                          time_margins=margins)
