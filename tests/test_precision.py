"""Significant digits from the spread over perturbed runs (run.precision)."""

import numpy as np
import pandas as pd
import pytest

from ikob2.run.precision import (round_significant, significant_digits, statistics,
                                 summarise)


def test_significant_digits_follow_the_leading_digit_of_the_uncertainty():
    assert significant_digits(12347.0, 2500.0) == 2          # 12 000
    assert significant_digits(94512.0, 40.0) == 4            # 94 510
    assert significant_digits(0.4186, 0.03) == 2             # 0.42
    assert significant_digits(579.3, 600.0) == 1             # at least one digit
    assert np.isnan(significant_digits(5.0, 0.0))
    assert round_significant(12347.0, 2) == 12000.0
    assert round_significant(0.4186, 2) == pytest.approx(0.42)
    assert round_significant(7.0, np.nan) == 7.0


def runs(shift=0.0):
    rows = []
    for o, pop in (("A", 10.0), ("B", 30.0)):
        for s, ic in (("single_D2", "D2"), ("single_D7", "D7")):
            rows.append({"buurtcode": o, "segment": s, "mode": "pt_v2", "income_class": ic,
                         "population": pop, "accessibility": 100.0 + shift})
    base = pd.DataFrame(rows)
    cut = base.assign(accessibility=base["accessibility"] + np.where(
        base["income_class"] == "D2", 6.0, 0.0))
    return {"s0": base, "s1": cut}


def test_statistics_and_summary_by_hand():
    st = statistics(runs(), "pt_v2", {"s1": 120.0}).set_index(
        ["statistic", "scenario", "income_class"])["value"]
    assert st[("level", "s0", "D2")] == 100.0
    assert st[("gain_per_person", "s1", "D2")] == 6.0
    assert st[("gain_job_persons", "s1", "all")] == 6.0 * 40     # 40 persons in D2
    assert st[("gain_per_eur", "s1", "all")] == pytest.approx(2.0)
    ref = statistics(runs(), "pt_v2", {"s1": 120.0})
    draws = pd.concat([statistics(runs(d), "pt_v2", {"s1": 120.0}).assign(draw=i)
                       for i, d in enumerate((-2.0, 0.0, 2.0))])
    s = summarise(ref, draws).set_index(["statistic", "scenario", "income_class"])
    # the levels move by +-2 (sd 2), the gains not at all: shared inputs cancel
    assert s.loc[("level", "s0", "D2"), "sd"] == pytest.approx(2.0)
    assert s.loc[("level", "s0", "D2"), "digits"] == 3       # 100, sd 2
    assert s.loc[("gain_per_person", "s1", "D2"), "sd"] == 0.0
    assert np.isnan(s.loc[("gain_per_person", "s1", "D2"), "digits"])
