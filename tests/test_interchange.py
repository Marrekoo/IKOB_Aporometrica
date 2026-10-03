import numpy as np
import pandas as pd
import pytest

from ikob2.run.interchange import interchange_ratio


def table(gains, mode="pt_v2", base=100.0):
    rows = []
    for (o, s), g in gains.items():
        ht, ic = s.rsplit("_", 1)
        rows.append({"buurtcode": o, "mode": mode, "segment": s,
                     "household_type": ht, "income_class": ic,
                     "population": 10.0, "accessibility": base + g})
    return pd.DataFrame(rows)


SEGS = ["single_D2", "single_D3", "single_D4", "single_D5"]


def world(ga, gb):
    keys = [(o, s) for o in ("O1", "O2") for s in SEGS]
    base = table({k: 0.0 for k in keys})
    a = table({k: ga(k) for k in keys})
    b = table({k: gb(k) for k in keys})
    return base, a, b


def test_a_common_ratio_has_no_dispersion():
    """Generalised cost with common inputs: R is the same for every segment."""
    base, a, b = world(lambda k: 2.0 * (1 + SEGS.index(k[1])),
                       lambda k: 1.0 * (1 + SEGS.index(k[1])))
    res = interchange_ratio(base, a, b)
    assert np.allclose(res["pairs"]["R"], 2.0)
    assert res["summary"]["pooled_cv"].iloc[0] == pytest.approx(0.0)
    assert res["summary"]["pooled_iqr_ratio"].iloc[0] == pytest.approx(1.0)


def test_segment_varying_ratio_gives_dispersion_and_undefined_pairs_are_kept():
    gate_a = {"single_D2": 8.0, "single_D3": 4.0, "single_D4": 1.0,
              "single_D5": 0.0}
    gate_b = {"single_D2": 2.0, "single_D3": 2.0, "single_D4": 2.0,
              "single_D5": 0.0}                       # D5: no gain from B
    base, a, b = world(lambda k: gate_a[k[1]], lambda k: gate_b[k[1]])
    res = interchange_ratio(base, a, b, min_segments=3)
    p = res["pairs"]
    assert list(p.loc[p.segment == "single_D5", "defined"]) == [False, False]
    assert p.loc[p.segment == "single_D2", "R"].iloc[0] == pytest.approx(4.0)
    o = res["origins"]
    assert (o["defined"] == 3).all() and (o["cv"] > 0.5).all()
    s = res["summary"].iloc[0]
    assert s["pairs_undefined"] == 2 and s["origins_undefined"] == 0
    assert s["pooled_cv"] > 0.5 and s["pooled_median_R"] == pytest.approx(2.0)


def test_origins_without_gain_from_b_are_reported_undefined():
    base, a, b = world(lambda k: 1.0, lambda k: 0.0 if k[0] == "O2" else 1.0)
    res = interchange_ratio(base, a, b)
    o = res["origins"].set_index("buurtcode")
    assert o.loc["O2", "defined"] == 0 and np.isnan(o.loc["O2", "cv"])
    assert res["summary"]["origins_undefined"].iloc[0] == 1


def test_dispersion_is_within_origin_not_across_origins():
    """Two origins with different but internally constant ratios: no
    between-segment dispersion, whatever the between-origin difference."""
    base, a, b = world(lambda k: 4.0 if k[0] == "O1" else 1.0, lambda k: 1.0)
    s = interchange_ratio(base, a, b)["summary"].iloc[0]
    assert s["pooled_cv"] == pytest.approx(0.0)


def test_correlations_between_specifications_agree_in_levels_but_not_in_gains():
    from ikob2.cli.paper_tables import correlations

    def tab(levels, mode="pt_v2"):
        rows = []
        for i, a in enumerate(levels):
            rows.append({"buurtcode": f"O{i // 3}", "mode": mode,
                         "segment": f"single_D{2 + i % 3}", "household_type": "single",
                         "income_class": f"D{2 + i % 3}", "population": 1.0,
                         "accessibility": a})
        return pd.DataFrame(rows)
    lv = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0]
    a0, b0 = tab(lv), tab([2 * v for v in lv])              # same ranking
    a1 = tab([v + g for v, g in zip(lv, [5, 0, 0, 5, 0, 0])])
    b1 = tab([2 * v + g for v, g in zip(lv, [0, 5, 0, 0, 5, 0])])   # other gains
    c = correlations({"a": {"s0": a0, "s1": a1}, "b": {"s0": b0, "s1": b1}}, "pt_v2")
    get = lambda m, x, y: c[(c.measure == m) & (c.method == "pearson")  # noqa: E731
                            & (c.spec_a == x) & (c.spec_b == y)]["correlation"].iloc[0]
    assert get("levels", "a", "b") == pytest.approx(1.0)
    assert get("gain_s1", "a", "b") < 0.5
