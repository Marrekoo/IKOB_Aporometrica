"""Segment pipeline (GSPREE port): unit tests on synthetic data."""

import os
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from ikob2.segments.config import LOW_INCOME_CLASSES, SegmentConfig
from ikob2.segments.ipf import ipf_batch
from ikob2.segments.kwb import clean_pct, read_kwb
from ikob2.segments.marginals import (
    build_gemeente_seed,
    decile_marginals,
    household_marginals,
    single_parent_shares,
)
from ikob2.segments.pipeline import compute_segments
from ikob2.segments.structure import (
    fit_structure_model,
    gemeente_covariates,
    std_vec,
)

CFG = SegmentConfig()


# ── IPF ──────────────────────────────────────────────────────────────

def test_ipf_reproduces_both_marginals():
    rng = np.random.default_rng(0)
    seed = rng.uniform(0.1, 5, size=(6, 4, 11))
    row = rng.uniform(10, 100, size=(6, 4))
    col = rng.uniform(1, 9, size=(6, 11))
    res = ipf_batch(seed, row, col, tol=1e-10, max_iter=500)
    assert res.converged.all()
    np.testing.assert_allclose(res.table.sum(2), row, atol=1e-8)
    scaled_col = col * (row.sum(1) / col.sum(1))[:, None]
    np.testing.assert_allclose(res.table.sum(1), scaled_col, atol=1e-8)


def test_ipf_uniform_seed_gives_independence():
    row = np.array([[30.0, 70.0]])
    col = np.array([[20.0, 80.0]])
    res = ipf_batch(np.ones((1, 2, 2)), row, col)
    np.testing.assert_allclose(res.table[0], np.outer(row[0], col[0]) / 100)


def test_ipf_preserves_odds_ratio_of_seed():
    seed = np.array([[[4.0, 1.0], [1.0, 4.0]]])
    res = ipf_batch(seed, np.array([[50.0, 50.0]]), np.array([[70.0, 30.0]]))
    t = res.table[0]
    assert (t[0, 0] * t[1, 1]) / (t[0, 1] * t[1, 0]) == pytest.approx(16.0)


def test_ipf_zero_row_target_gives_zero_row():
    seed = np.ones((1, 3, 2))
    res = ipf_batch(seed, np.array([[10.0, 0.0, 10.0]]),
                    np.array([[12.0, 8.0]]))
    np.testing.assert_allclose(res.table[0, 1], 0.0, atol=1e-6)
    assert res.table.sum() == pytest.approx(20.0)


def test_ipf_tables_converge_independently():
    easy = np.ones((2, 2))
    hard = np.array([[1e6, 1.0], [1.0, 1e6]])
    seed = np.stack([easy, hard])
    res = ipf_batch(seed, np.array([[5.0, 5.0]] * 2),
                    np.array([[7.0, 3.0]] * 2), tol=1e-12, max_iter=500)
    assert res.iters[0] < res.iters[1]


def test_ipf_input_validation():
    with pytest.raises(ValueError, match="do not match"):
        ipf_batch(np.ones((1, 2, 2)), np.ones((1, 3)), np.ones((1, 2)))
    with pytest.raises(ValueError, match="positive totals"):
        ipf_batch(np.ones((1, 2, 2)), np.zeros((1, 2)), np.ones((1, 2)))


# ── Cleaning / helpers ───────────────────────────────────────────────

def test_clean_pct_drops_sentinels_and_out_of_range():
    x = pd.Series([0, 50, 100, -99999999, 101, np.nan, "12,5x"])
    out = clean_pct(x)
    assert out.iloc[:3].tolist() == [0, 50, 100]
    assert out.iloc[3:].isna().all()


def test_std_vec_matches_sample_sd_and_handles_constant():
    x = pd.Series([1.0, 2.0, 3.0, np.nan])
    z = std_vec(x)
    assert z.iloc[:3].tolist() == pytest.approx([-1.0, 0.0, 1.0])
    assert np.isnan(z.iloc[3])
    assert (std_vec(pd.Series([5.0, 5.0, 5.0])) == 0).all()


def test_segment_columns_are_class_major_like_r():
    cols = CFG.segment_columns
    assert len(cols) == 44 and len(set(cols)) == 44
    assert cols[:5] == ["single_D1", "couple_D1", "single_parent_D1",
                        "couple_children_D1", "single_D2"]
    assert cols[-1] == "couple_children_onbekend"


# ── Synthetic inputs ─────────────────────────────────────────────────

def make_income_raw(n_gm=14, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for g in range(n_gm):
        for t, key in CFG.household_key_map.items():
            pct = rng.dirichlet(np.ones(10) * 8) * 100
            pct = np.round(pct)
            row = {"Populatie": "1050010", "KenmerkenVanHuishoudens": key,
                   "RegioS": f"GM{1000 + g:04d}  ", "Perioden": "2022JJ00",
                   CFG.income_total_col: float(rng.uniform(2, 60))}
            for (cls, col), p in zip(CFG.income_decile_cols.items(), pct):
                row[col] = p
            rows.append(row)
    return pd.DataFrame(rows)


def make_children_raw(n_gm=14):
    rows = [{"LeeftijdKindEren": "1017000", "Perioden": "2022JJ00",
             "RegioS": f"GM{1000 + g:04d}  ",
             "TotaalHuishoudensMetKinderen_1": 1000.0 + 10 * g,
             "TotaalEenouderhuishoudensMetKinderen_13": 200.0 + g}
            for g in range(n_gm)]
    # other age categories must be ignored
    rows.append({"LeeftijdKindEren": "9999999", "Perioden": "2022JJ00",
                 "RegioS": "GM1000", "TotaalHuishoudensMetKinderen_1": 5.0,
                 "TotaalEenouderhuishoudensMetKinderen_13": 5.0})
    return pd.DataFrame(rows)


def make_kwb(n_gm=14, per_gm=5, seed=1):
    rng = np.random.default_rng(seed)
    recs = []
    for g in range(n_gm):
        for b in range(per_gm):
            hh = float(rng.integers(200, 900))
            p_single = float(rng.integers(20, 50))
            p_no = float(rng.integers(20, 40))
            recs.append({
                "buurtcode": f"BU{1000 + g:04d}{b:04d}",
                "buurtnaam": f"b{g}-{b}", "gemeentecode": f"GM{1000 + g:04d}",
                "gemeentenaam": f"g{g}",
                "inwoners": hh * 2.2, "huishoudens": hh,
                "p_hh_single": p_single, "p_hh_no_child": p_no,
                "p_hh_with_child": 100 - p_single - p_no,
                "p_laag40": float(rng.integers(10, 60)),
                "stedelijkheid": float(rng.integers(1, 6)),
                "gem_woz": float(rng.integers(150, 600)),
            })
    kwb = pd.DataFrame(recs)
    kwb["hh_eenpersoons"] = kwb.huishoudens * kwb.p_hh_single / 100
    kwb["hh_zonder_kind"] = kwb.huishoudens * kwb.p_hh_no_child / 100
    kwb["hh_met_kind"] = kwb.huishoudens * kwb.p_hh_with_child / 100
    return kwb


# ── Seed / shares ────────────────────────────────────────────────────

def test_seed_converts_percentages_to_counts_with_remainder_unknown():
    raw = make_income_raw(n_gm=1)
    seed = build_gemeente_seed(raw, CFG)
    assert len(seed) == 1 * 4 * 11
    row = raw[raw.KenmerkenVanHuishoudens == "1050015"].iloc[0]
    total = row[CFG.income_total_col] * 1000
    d1 = seed[(seed.household_type == "single") & (seed.income_class == "D1")]
    assert d1.n_hh.iloc[0] == pytest.approx(
        total * row[CFG.income_decile_cols["D1"]] / 100)
    single = seed[seed.household_type == "single"]
    assert single.n_hh.sum() == pytest.approx(
        max(total, single[single.income_class != "onbekend"].n_hh.sum()),
        rel=1e-6)


def test_seed_floors_empty_cells_and_strips_keys():
    raw = make_income_raw(n_gm=2)
    col = CFG.income_decile_cols["D5"]
    raw.loc[0, col] = np.nan
    seed = build_gemeente_seed(raw, CFG)
    assert (seed.n_hh > 0).all()
    assert seed.gemeentecode.str.match(r"^GM\d{4}$").all()
    cell = seed[(seed.gemeentecode == "GM1000") & (seed.income_class == "D5")
                & (seed.household_type == "single")]
    assert cell.n_hh.iloc[0] == 1e-6


def test_single_parent_shares_use_total_age_only_and_fallback():
    raw = make_children_raw(3)
    raw.loc[raw.RegioS.str.strip() == "GM1001", "TotaalHuishoudensMetKinderen_1"] = 0.0
    s = single_parent_shares(raw, CFG)
    assert s["GM1000"] == pytest.approx(200 / 1000)
    assert s["GM1001"] == CFG.fallback_single_parent_share_of_with_children


# ── Household marginals ──────────────────────────────────────────────

def test_household_marginals_sum_to_households():
    kwb = make_kwb()
    sp = single_parent_shares(make_children_raw(), CFG)
    hh = household_marginals(kwb, sp, CFG)
    np.testing.assert_allclose(hh[list(CFG.household_types)].sum(axis=1),
                               kwb.huishoudens)
    assert not hh.use_fallback.any()
    g0 = hh.iloc[0]
    with_kids = kwb.hh_met_kind.iloc[0]
    share = sp["GM1000"]
    assert g0.single_parent == pytest.approx(with_kids * share)
    assert g0.couple == pytest.approx(kwb.hh_zonder_kind.iloc[0])


def test_household_marginals_fallback_mix_and_no_child_includes_single():
    kwb = make_kwb(n_gm=2, per_gm=2)
    kwb.loc[0, ["hh_eenpersoons", "hh_zonder_kind", "hh_met_kind"]] = np.nan
    sp = single_parent_shares(make_children_raw(2), CFG)
    hh = household_marginals(kwb, sp, CFG)
    assert hh.use_fallback.tolist() == [True, False, False, False]
    r = hh.iloc[0]
    assert r.single == pytest.approx(kwb.huishoudens[0] * 0.40)
    assert r.couple_children == pytest.approx(kwb.huishoudens[0] * 0.24)

    cfg2 = replace(CFG, kwb_no_child_includes_single=True)
    hh2 = household_marginals(kwb, sp, cfg2)
    raw_couple = max(kwb.hh_zonder_kind[1] - kwb.hh_eenpersoons[1], 0)
    total = (kwb.hh_eenpersoons[1] + raw_couple + kwb.hh_met_kind[1])
    assert hh2.couple[1] == pytest.approx(
        raw_couple * kwb.huishoudens[1] / total)


def test_household_marginals_missing_households_stays_missing():
    kwb = make_kwb(n_gm=1, per_gm=2)
    kwb.loc[0, "huishoudens"] = np.nan
    hh = household_marginals(kwb, single_parent_shares(make_children_raw(1), CFG), CFG)
    assert hh.loc[0, list(CFG.household_types)].isna().all()
    assert not hh.use_fallback[0]


# ── Income marginals ─────────────────────────────────────────────────

def test_decile_marginals_hit_local_low_income_target():
    kwb = make_kwb()
    seed = build_gemeente_seed(make_income_raw(), CFG)
    dec = decile_marginals(kwb, seed, CFG)
    classes = list(CFG.income_classes)
    np.testing.assert_allclose(dec[classes].sum(axis=1), 1.0)
    low = dec[list(LOW_INCOME_CLASSES)].sum(axis=1)
    np.testing.assert_allclose(low, kwb.p_laag40 / 100, atol=1e-9)
    assert (dec.calibration_source == "p_laag40").all()


def test_decile_marginals_fall_back_to_gemeente_median_then_shape():
    kwb = make_kwb(n_gm=3, per_gm=3)
    seed = build_gemeente_seed(make_income_raw(3), CFG)
    kwb.loc[0, "p_laag40"] = np.nan            # gm0 has other valid buurten
    kwb.loc[3:5, "p_laag40"] = np.nan          # gm1: none valid
    dec = decile_marginals(kwb, seed, CFG)
    assert dec.calibration_source[0] == "gemeente_median_p_laag40"
    assert dec.calibration_source[3] == "gemeente_shape"
    exp = kwb.loc[1:2, "p_laag40"].median() / 100
    assert dec.low_target_used[0] == pytest.approx(exp)


def test_decile_marginals_uniform_shape_for_unknown_gemeente():
    kwb = make_kwb(n_gm=2, per_gm=1)
    seed = build_gemeente_seed(make_income_raw(1), CFG)   # gm1 missing
    dec = decile_marginals(kwb, seed, CFG)
    assert dec.calibration_source[1] == "p_laag40"
    low = dec.loc[1, list(LOW_INCOME_CLASSES)].sum()
    assert low == pytest.approx(kwb.p_laag40[1] / 100)


def test_calibration_none_keeps_municipal_shape():
    kwb = make_kwb(n_gm=2, per_gm=2)
    seed = build_gemeente_seed(make_income_raw(2), CFG)
    dec = decile_marginals(kwb, seed, replace(CFG, local_income_calibration="none"))
    assert (dec.calibration_source == "none").all()
    assert dec.groupby(kwb.gemeentecode).low_target_used.nunique().eq(1).all()


# ── Structure model ──────────────────────────────────────────────────

@pytest.mark.filterwarnings("ignore:Perfect separation")
def test_structure_model_recovers_known_coefficients():
    rng = np.random.default_rng(3)
    G = 30
    cov = pd.DataFrame({
        "gemeentecode": [f"GM{i:04d}" for i in range(G)],
        "stedelijkheid_std": rng.normal(size=G),
        "gem_woz_std": rng.normal(size=G),
    })
    cells = [(t, c) for t in CFG.household_types for c in CFG.income_classes]
    a = rng.normal(1.0, 0.5, len(cells))
    b1 = rng.normal(0, 0.2, len(cells))
    b2 = rng.normal(0, 0.2, len(cells))
    rows = []
    for g in range(G):
        for k, (t, c) in enumerate(cells):
            mu = np.exp(a[k] + b1[k] * cov.stedelijkheid_std[g]
                        + b2[k] * cov.gem_woz_std[g])
            rows.append((cov.gemeentecode[g], t, c, mu * 1000))
    seed = pd.DataFrame(rows, columns=["gemeentecode", "household_type",
                                       "income_class", "n_hh"])
    model = fit_structure_model(seed, cov, CFG)
    assert model.with_covariates
    np.testing.assert_allclose(model.alpha, a, atol=1e-4)
    np.testing.assert_allclose(model.beta_sted, b1, atol=1e-4)
    np.testing.assert_allclose(model.beta_woz, b2, atol=1e-4)
    pred = model.predict([0.3], [-0.7], CFG)[0]
    k = cells.index(("couple", "D3"))
    assert pred[1, 2] == pytest.approx(np.exp(a[k] + b1[k] * 0.3 - b2[k] * 0.7))


def test_structure_model_without_enough_covariates_is_cell_mean():
    seed = build_gemeente_seed(make_income_raw(4), CFG)
    cov = pd.DataFrame({"gemeentecode": ["GM1000"],
                        "stedelijkheid_std": [0.0], "gem_woz_std": [0.0]})
    model = fit_structure_model(seed, cov, CFG)
    assert not model.with_covariates
    cell = seed[(seed.household_type == "couple")
                & (seed.income_class == "D2")].n_hh.mean() / 1000
    got = model.predict([0.0], [0.0], CFG)[0]
    assert got[1, 1] == pytest.approx(cell)


def test_structure_predict_missing_covariate_gives_floor_seed():
    cov = pd.DataFrame({"gemeentecode": [f"GM{i:04d}" for i in range(12)],
                        "stedelijkheid_std": np.linspace(-1, 1, 12),
                        "gem_woz_std": np.cos(np.arange(12))})
    seed = build_gemeente_seed(make_income_raw(12), CFG)
    seed["gemeentecode"] = seed["gemeentecode"].map(
        dict(zip(sorted(seed.gemeentecode.unique()), cov.gemeentecode)))
    model = fit_structure_model(seed, cov, CFG)
    assert model.with_covariates
    pred = model.predict([np.nan, 0.0], [0.0, 0.0], CFG)
    assert (pred[0] == 1e-8).all()          # R: NA prediction -> 1e-8
    assert (pred[1] > 1e-8).all()


# ── End to end ───────────────────────────────────────────────────────

def test_compute_segments_end_to_end_invariants():
    kwb = make_kwb()
    kwb.loc[3, ["huishoudens", "inwoners"]] = 0.0     # empty buurt
    kwb.loc[3, ["hh_eenpersoons", "hh_zonder_kind", "hh_met_kind"]] = 0.0
    res = compute_segments(kwb, make_income_raw(), make_children_raw(), CFG)

    hb, sc = res.household_based, res.population_scaled
    seg = CFG.segment_columns
    assert list(hb.columns[-44:]) == seg and len(hb) == len(kwb)
    assert (hb[seg].to_numpy() >= 0).all()

    # zero-household buurt is skipped and all zeros
    assert res.diagnostics.status[3] == "skipped_zero_households"
    assert (hb.loc[3, seg] == 0).all() and (sc.loc[3, seg] == 0).all()

    ok = res.diagnostics.status == "ok"
    assert ok.sum() == len(kwb) - 1

    # household-type marginals are reproduced (persons / hh_size summed
    # over income classes == household marginal)
    for t in CFG.household_types:
        cols = [f"{t}_{c}" for c in CFG.income_classes]
        hh_t = hb.loc[ok, cols].sum(axis=1) / CFG.hh_size[t]
        np.testing.assert_allclose(
            hh_t, res.hh_marginals.loc[ok, t], rtol=1e-6, atol=1e-6)

    # income marginals are reproduced
    for c in CFG.income_classes:
        cols = [f"{t}_{c}" for t in CFG.household_types]
        hh_c = sum(hb.loc[ok, f"{t}_{c}"] / CFG.hh_size[t]
                   for t in CFG.household_types)
        np.testing.assert_allclose(
            hh_c, res.dec_marginals.loc[ok, c] * kwb.huishoudens[ok],
            rtol=1e-5, atol=1e-5)

    # population-scaled variant sums to inwoners
    np.testing.assert_allclose(sc.loc[ok, seg].sum(axis=1),
                               kwb.inwoners[ok], rtol=1e-9)
    assert res.report["not_converged"] == 0
    assert res.report["hh_marginal_max_abs_diff"] < 1e-9


def test_compute_segments_zero_population_scaled_to_zero():
    kwb = make_kwb(n_gm=12, per_gm=2)
    kwb.loc[1, "inwoners"] = np.nan
    res = compute_segments(kwb, make_income_raw(12), make_children_raw(12), CFG)
    seg = CFG.segment_columns
    assert (res.population_scaled.loc[1, seg] == 0).all()
    assert res.household_based.loc[1, seg].sum() > 0


# ── KWB reader ───────────────────────────────────────────────────────

def _write_kwb(path):
    geopandas = pytest.importorskip("geopandas")
    from shapely.geometry import Point
    v = CFG.kwb_vars
    df = pd.DataFrame({
        v["zonecode"]: ["BU00010001", "BU00010002", "WK000100", "BU00020001"],
        v["zonename"]: list("abcd"),
        v["citycode"]: ["GM0001", "GM0001", "GM0001", "GM0002"],
        v["cityname"]: ["x", "x", "x", "y"],
        v["inhabitants"]: [100, -99999999, 50, 80],
        v["households"]: [40, 30, 20, -99999999],
        v["p_hh_single"]: [30, 40, 20, 10],
        v["p_hh_no_child"]: [30, 30, 30, 30],
        v["p_hh_with_child"]: [40, 30, 50, -99999999],
        v["stedelijkheid"]: [2, -99999999, 3, 4],
        v["avg_house_value"]: [300, 250, -99999999, 400],
        v["p_hh_low_income"]: [25, 30, 40, 101],
    })
    gdf = geopandas.GeoDataFrame(df, geometry=[Point(i, i) for i in range(4)],
                                 crs="EPSG:28992")
    gdf.to_file(path, layer="buurten", driver="GPKG")


def test_read_kwb_filters_and_cleans(tmp_path):
    p = tmp_path / "k.gpkg"
    _write_kwb(p)
    kwb = read_kwb(p, CFG)
    assert kwb.buurtcode.tolist() == ["BU00010001", "BU00010002", "BU00020001"]
    assert np.isnan(kwb.inwoners[1]) and np.isnan(kwb.huishoudens[2])
    assert kwb.hh_eenpersoons[0] == pytest.approx(40 * 30 / 100)
    assert np.isnan(kwb.p_laag40[2])           # 101 invalid
    # covariate sentinels: NA by default, kept for R parity
    assert np.isnan(kwb.stedelijkheid[1])
    assert kwb.stedelijkheid[[0, 2]].tolist() == [2, 4]
    kept = read_kwb(p, replace(CFG, covariate_sentinels="keep"))
    assert kept.stedelijkheid[1] == -99999999


def test_read_kwb_study_area_and_missing_columns(tmp_path):
    p = tmp_path / "k.gpkg"
    _write_kwb(p)
    kwb = read_kwb(p, replace(CFG, gemeente_codes=("GM0002",)))
    assert kwb.buurtcode.tolist() == ["BU00020001"]
    with pytest.raises(ValueError, match="No buurten left"):
        read_kwb(p, replace(CFG, gemeente_codes=("GM9999",)))
    bad = replace(CFG, kwb_vars={**CFG.kwb_vars, "p_hh_low_income": "nope"})
    with pytest.raises(KeyError, match="nope"):
        read_kwb(p, bad)


# ── Parity with the R script (optional) ──────────────────────────────

# Set IKOB_R_SEGMENTS_GPKG (the R output) and IKOB_R_KWB_GPKG (the KWB
# file it was built from) to run the parity check; skipped otherwise.
R_GPKG = os.environ.get("IKOB_R_SEGMENTS_GPKG",
                        "data/r_reference/nl_segments.gpkg")
R_KWB = os.environ.get("IKOB_R_KWB_GPKG", "data/wijkenbuurten_2022_v3.gpkg")


@pytest.mark.skipif(not (os.path.exists(R_GPKG) and os.path.exists(R_KWB)
                         and os.path.exists("data/statline")),
                    reason="R reference output / inputs not available")
def test_parity_with_r_household_based():
    import geopandas as gpd
    from ikob2.segments.pipeline import run_pipeline

    cfg = replace(CFG, covariate_sentinels="keep")
    res = run_pipeline(R_KWB, "data/statline", cfg)
    ref = gpd.read_file(R_GPKG, layer="buurt_segments_household_based",
                        ignore_geometry=True).set_index("buurtcode")
    got = res.household_based.set_index("buurtcode").loc[ref.index]
    seg = cfg.segment_columns
    np.testing.assert_allclose(got[seg].to_numpy(), ref[seg].to_numpy(),
                               rtol=1e-4, atol=1e-4)
