import numpy as np
import pandas as pd
import pytest

from ikob2.outputs.diagnostics import (_area_over_diagonal, money_gate,
                                       segment_survival, ttt)


def env(rows):
    return pd.DataFrame(rows, columns=["household_type", "income_class", "low",
                                       "high", "atom"])


def test_uniform_segment_is_increasing_hazard_and_ttt_above_the_diagonal():
    g = np.arange(0.0, 100.01, 0.5)
    s = segment_survival(0.0, 100.0, 0.0, g)
    u, phi = ttt(s, g)
    assert _area_over_diagonal(u, phi) == pytest.approx(1 / 6, abs=2e-3)


def test_exponential_ttt_is_the_diagonal():
    g = np.arange(0.0, 200.0, 0.05)
    u, phi = ttt(np.exp(-g / 10.0), g)
    assert _area_over_diagonal(u, phi) == pytest.approx(0.0, abs=5e-3)


def test_one_segment_has_no_aggregation_shift():
    e = env([("single", "D3", 10.0, 50.0, 0.0)])
    pop = pd.DataFrame({"single_D3": [100.0]}, index=["O1"])
    s = money_gate(e, pop)["summary"].iloc[0]
    assert s["ttt_shift"] == pytest.approx(0.0, abs=1e-9)
    assert s["hazard_class"] == "increasing" and s["atom"] == 0.0


def test_mixing_segments_of_different_scale_moves_toward_decreasing_hazard():
    e = env([("single", "D2", 0.0, 10.0, 0.0), ("single", "D9", 0.0, 400.0, 0.0)])
    pop = pd.DataFrame({"single_D2": [50.0], "single_D9": [50.0]}, index=["O1"])
    s = money_gate(e, pop, step=0.5)["summary"].iloc[0]
    assert s["ttt_shift"] < -0.02                 # mixture is less IFR than parts
    assert s["cv"] > 1.0                          # more dispersed than a uniform


def test_atom_mean_and_zero_population_origins():
    e = env([("single", "D1", 0.0, 0.0, 1.0), ("single", "D5", 10.0, 30.0, 0.0)])
    pop = pd.DataFrame({"single_D1": [30.0, 0.0], "single_D5": [70.0, 0.0]},
                       index=["O1", "O2"])
    r = money_gate(e, pop)
    s = r["summary"].set_index("buurtcode")
    assert list(s.index) == ["O1"]                # empty origin skipped
    assert s.loc["O1", "atom"] == pytest.approx(0.3)
    assert s.loc["O1", "mean_threshold"] == pytest.approx(0.7 * 20.0, rel=0.02)
    c = r["curves"]
    assert c.loc[c.c == 0, "survival"].iloc[0] == 1.0    # free trips clear
