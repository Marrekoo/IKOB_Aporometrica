"""Inputs drawn within the rounding of their sources (run.perturb)."""

import pandas as pd
import pytest

from ikob2.run.perturb import perturb_jobs, perturb_population


def jobs():
    return pd.DataFrame({"L01": [30.0, 10.0, 0.0, 50.0, 8.0], "L02": [0.0, 0.0, 5.0, 20.0, 0.0]},
                        index=["BU03440101", "BU03440102", "BU03440201", "BU00340101",
                               "BU00350101"])


def test_jobs_move_within_the_rounding_and_keep_their_proportions():
    j = jobs()
    p = perturb_jobs(j, seed=3, unit=10.0)
    gem = j.index.str[2:6]
    t0, t1 = j.groupby(gem).sum(), p.groupby(gem).sum()
    assert ((t1 - t0).abs() <= 5.0 + 1e-9).to_numpy().all()
    # within municipality 0344 and sector L01 the two buurten keep 3 : 1
    assert p.loc["BU03440101", "L01"] == pytest.approx(3 * p.loc["BU03440102", "L01"])
    # a municipality x sector published as zero (0035, L02) stays zero
    assert p.loc["BU00350101", "L02"] == 0.0 and p.loc["BU00350101", "L01"] != 8.0
    assert p.loc["BU00340101", "L01"] != 50.0                 # it does move
    pd.testing.assert_frame_equal(p, perturb_jobs(j, seed=3, unit=10.0))
    assert not p.equals(perturb_jobs(j, seed=4, unit=10.0))
    with pytest.raises(ValueError):
        perturb_jobs(j, seed=1, unit=0.0)


def population():
    cols = {f"{t}_{c}": v for (t, c), v in zip(
        [("single", "D2"), ("couple", "D2"), ("single", "D5"), ("couple", "D5")],
        [100.0, 300.0, 200.0, 400.0])}
    return pd.DataFrame([{"buurtcode": "BU1", "inwoners": 1000.0, **cols},
                         {"buurtcode": "BU2", "inwoners": 0.0,
                          **{k: 0.0 for k in cols}}])


def test_population_moves_within_the_rounding_of_counts_and_shares():
    pop = population()
    p = perturb_population(pop, seed=7, count_unit=5.0, share_unit=0.01)
    seg = [c for c in pop.columns if c.endswith(("D2", "D5"))]
    t0, t1 = pop.loc[0, seg].sum(), p.loc[0, seg].sum()
    assert abs(t1 - t0) <= 2.5 + 1e-9 and t1 != t0
    assert p.loc[0, "inwoners"] == pytest.approx(t1)
    # the income-class shares move by at most half a percentage point (plus
    # the renormalisation)
    d2_0 = pop.loc[0, ["single_D2", "couple_D2"]].sum() / t0
    d2_1 = p.loc[0, ["single_D2", "couple_D2"]].sum() / t1
    assert abs(d2_1 - d2_0) <= 0.0075 and d2_1 != d2_0
    assert (p.loc[0, seg] >= 0).all()
    assert (p.loc[1, seg] == 0).all()                          # an empty buurt stays empty
    pd.testing.assert_frame_equal(p, perturb_population(pop, 7, 5.0, 0.01))
