import pandas as pd
import pytest

from ikob2.run.gap import reachability_gap, summarise


def table(a, atom=0.0):
    return pd.DataFrame({
        "buurtcode": ["O1", "O1", "O2", "O2"], "mode": "pt",
        "segment": ["single_D2", "single_D9"] * 2,
        "household_type": "single", "income_class": ["D2", "D9"] * 2,
        "population": [10.0, 30.0, 20.0, 20.0], "accessibility": a,
        "atom": atom})


def test_gap_is_time_only_minus_gated_and_never_negative_for_a_gate():
    g = reachability_gap(table([60.0, 90.0, 40.0, 80.0]),
                         table([100.0, 100.0, 100.0, 100.0]), "pt")
    assert list(g["gap"]) == [40.0, 10.0, 60.0, 20.0]
    assert g["gap_share"].iloc[0] == pytest.approx(0.4)
    assert (g["gap"] >= 0).all()


def test_summary_weights_by_population_and_reports_the_atom():
    gated = table([60.0, 90.0, 40.0, 80.0], atom=[0.5, 0.0, 0.5, 0.0])
    g = reachability_gap(gated, table([100.0] * 4), "pt")
    s = summarise(g)
    assert s.loc["D2", "gap"] == pytest.approx((40 * 10 + 60 * 20) / 30)
    assert s.loc["D9", "gap_share"] == pytest.approx(
        (10 * 30 + 20 * 20) / (100 * 50))
    assert s.loc["D2", "atom"] == 0.5 and s.loc["D9", "atom"] == 0.0
