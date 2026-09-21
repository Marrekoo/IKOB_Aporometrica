import numpy as np
import pandas as pd
import pytest

from ikob2.segments.pt_spend import pt_spend_by_decile, pt_trips
from ikob2.skims.pt_fare import PtFareModel


def legs():
    rows = []
    def tour(vid, opid, inc, gem, w, legs_):
        for i, (rvm, km) in enumerate(legs_):
            rows.append({"OPID": opid, "HHGestInkG": inc, "WoGem": gem, "FactorP": 100.0,
                         "VerplID": vid, "Verpl": 1, "RitID": f"{vid}-{i}", "Rvm": rvm,
                         "AfstR": km * 10, "FactorV": w})
    tour("a", "p1", 2, 344, 100.0, [(9, 0.4), (2, 20.0), (9, 0.3)])       # walk-train-walk
    tour("b", "p1", 2, 344, 100.0, [(3, 5.0)])                             # bus
    tour("c", "p1", 2, 344, 100.0, [(1, 12.0)])                            # car: not PT
    tour("d", "p2", 2, 999, 100.0, [(2, 20.0)])                            # national only
    for opid, inc in (("p1", 2), ("p2", 2)):                               # persons without trips
        pass
    return pd.DataFrame(rows)


def test_a_pt_tour_is_priced_by_the_model_fare_rules():
    fm = PtFareModel()
    t = pt_trips(legs(), fm)
    assert set(t.index) == {"a", "b", "d"}            # the car tour is not a public transport tour
    assert t.loc["a", "fare"] == pytest.approx(float(fm.rail_fare([20.0])[0]), rel=1e-6)
    assert t.loc["b", "fare"] == pytest.approx(1.08 + 0.18 * 5.0, rel=1e-6)


def test_spend_is_trips_times_fare_with_local_shrinkage_towards_national():
    out = pt_spend_by_decile(legs(), municipality=344, prior=100.0).set_index("income_class")
    d2 = out.loc["D2"]
    assert d2["n_national"] == 2 and d2["n_local"] == 1
    assert d2["spend_eur_year"] == pytest.approx(d2["trips_per_year"] * d2["mean_fare_eur"])
    # the shrunk rate lies between the local and the national rate
    lo, hi = sorted([d2["trips_local"], d2["trips_national"]])
    assert lo <= d2["trips_per_year"] <= hi
    assert out.loc["D5", "n_national"] == 0 or np.isnan(out.loc["D5", "spend_eur_year"]) \
        or out.loc["D5", "spend_eur_year"] == 0
