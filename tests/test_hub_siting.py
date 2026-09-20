import numpy as np
import pandas as pd
import pytest

from ikob2.skims.hub_siting import propose_hubs, score_candidates


def cands():
    return pd.DataFrame({
        "code": list("ABCDE"),
        "x": [0.0, 100.0, 1000.0, 2000.0, 3000.0], "y": 0.0,
        "access": [10.0, 20.0, 30.0, 90.0, 100.0],
        "bike_share": [0.2, 0.3, 0.4, 0.9, 0.95]})


def test_low_access_and_low_ownership_score_lowest():
    s = score_candidates(cands())
    assert list(s.sort_values().index) == [0, 1, 2, 3, 4]
    only_access = score_candidates(cands().assign(
        bike_share=[0.9, 0.8, 0.1, 0.5, 0.6]), access_weight=1.0)
    assert list(only_access.sort_values().index)[:2] == [0, 1]


def test_hubs_respect_the_spacing_to_existing_and_chosen_hubs():
    # B is 100 m from A (chosen first): skipped; C is 1 km from A: taken
    got = propose_hubs(cands(), np.empty((0, 2)), 2, min_spacing_m=400)
    assert list(got["code"]) == ["A", "C"]
    # an existing hub at C's place blocks C
    got = propose_hubs(cands(), np.array([[1000.0, 0.0]]), 2, min_spacing_m=400)
    assert list(got["code"]) == ["A", "D"]


def test_too_many_hubs_for_the_spacing_is_an_error():
    with pytest.raises(ValueError, match="fit"):
        propose_hubs(cands(), np.empty((0, 2)), 5, min_spacing_m=5000)
    with pytest.raises(ValueError):
        score_candidates(cands(), access_weight=1.5)
