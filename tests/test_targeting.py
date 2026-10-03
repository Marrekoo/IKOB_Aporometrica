"""Person-based against location-based price cuts (run.targeting) and the
paper tables over scenarios and pairs (cli.paper_tables)."""

import json

import numpy as np
import pandas as pd
import pytest

from ikob2.run.targeting import targeting

# two origins (Z in the zone, R not), two segments (D2 in the target group,
# D7 not); population and baseline accessibility per cell
CELLS = [("Z", "single_D2", 10.0, 100.0), ("Z", "single_D7", 30.0, 300.0),
         ("R", "single_D2", 20.0, 200.0), ("R", "single_D7", 40.0, 400.0)]


def table(gains=(0.0, 0.0, 0.0, 0.0)):
    return pd.DataFrame([{"buurtcode": o, "segment": s, "mode": "pt_v2",
                          "income_class": s.split("_")[1], "population": p,
                          "accessibility": a + g}
                         for (o, s, p, a), g in zip(CELLS, gains)])


def test_targeting_by_hand():
    runs = {
        # by income: D2 everywhere; gains 5 (Z) and 2 (R) per D2 person
        "person": (table((5.0, 0.0, 2.0, 0.0)), {
            "lime_price_scales": {"single_D2": 0.5, "single_D7": 1.0},
            "scenario": {"cost": {"compensation_eur_year": 100.0,
                                  "by_segment_eur_year": {"single_D2": 100.0,
                                                          "single_D7": 0.0},
                                  "by_origin_eur_year": {"Z": 40.0, "R": 60.0}}}}),
        # by address: everyone in Z; D2 gains 5, D7 gains 1
        "place": (table((5.0, 1.0, 0.0, 0.0)), {
            "lime_price_scales": {"single_D2": 0.5, "single_D7": 0.5},
            "lime_price_zones": ["Z"],
            "scenario": {"cost": {"compensation_eur_year": 80.0,
                                  "by_segment_eur_year": {"single_D2": 20.0,
                                                          "single_D7": 60.0},
                                  "by_origin_eur_year": {"Z": 80.0, "R": 0.0}}}}),
    }
    out = targeting(table(), runs, ["Z"], target_classes=["D2"])
    s = out["summary"].set_index("scenario")
    # person-based: all 30 D2 persons eligible, nobody else
    assert s.loc["person", "target_eligible_share"] == 1.0
    assert s.loc["person", "eligible_outside_target_share"] == 0.0
    assert s.loc["person", "gain_job_persons"] == 5 * 10 + 2 * 20
    assert s.loc["person", "gain_share_zone"] == pytest.approx(50 / 90)
    assert s.loc["person", "cost_share_zone"] == pytest.approx(0.4)
    assert s.loc["person", "gain_per_eur"] == pytest.approx(0.9)
    # place-based: 10 of the 30 D2 persons; 30 of the 40 eligible are D7
    assert s.loc["place", "target_eligible_share"] == pytest.approx(10 / 30)
    assert s.loc["place", "eligible_outside_target_share"] == pytest.approx(30 / 40)
    assert s.loc["place", "cost_share_target"] == pytest.approx(0.25)
    assert s.loc["place", "gain_share_target"] == pytest.approx(50 / 80)
    c = out["cells"].set_index(["scenario", "income_class", "place"])
    assert c.loc[("place", "D7", "zone"), "gain_per_person"] == 1.0
    assert c.loc[("person", "D2", "rest"), "gain"] == 40.0


def test_paper_tables_read_the_scenarios_and_pairs(tmp_path):
    from ikob2.cli.paper_tables import tables
    from ikob2.utils.paths import DataLayout

    lay = DataLayout(tmp_path)
    rng = np.random.default_rng(1)
    origins = [f"O{i}" for i in range(4)]
    segs = [f"single_D{k}" for k in range(2, 7)]
    rows = [{"buurtcode": o, "segment": s, "mode": "pt_v2",
             "household_type": "single", "income_class": s.split("_")[1],
             "population": 10.0, "atom": 0.0} for o in origins for s in segs]

    def write(name, acc):
        d = lay.run_dir(name)
        d.mkdir(parents=True)
        t = pd.DataFrame(rows).assign(accessibility=acc,
                                      accessibility_normalised=acc)
        t.to_csv(d / "accessibility.csv", index=False)

    base = rng.uniform(100, 200, len(rows))
    write("sp_m0s_s0", base)
    write("sp_m0s_s4", base + rng.uniform(1, 5, len(rows)))
    write("sp_m0s_s2t", base + rng.uniform(1, 5, len(rows)))
    write("spt_m0_s0", base + 50)            # M0s takes M0's time-only run
    res = tables(lay, ["m0s"], "pt_v2", scenarios=("s4", "s2t"),
                 pairs=(("s4", "s2t"),), gap_scenarios=("s4",))
    inc = res["incidence_by_spec"]
    assert set(inc["scenario"]) == {"s4", "s2t"}
    assert res["interchange_by_spec"]["pair"].tolist() == ["s4/s2t"]
    assert set(res["gap_by_spec"]["scenario"]) == {"s0", "s4"}
    g0 = res["gap_by_spec"].query("scenario == 's0'")
    assert np.allclose(g0["gap"], 50.0)      # time-only minus gated
    assert res["correlation_by_spec"].empty   # one specification: nothing to correlate
    json.dumps(res["interchange_by_spec"].to_dict())   # plain values
