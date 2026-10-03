"""Jobs matched to income deciles through sector x occupation cells."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from ikob2.segments import occupations as occ
from ikob2.segments.jobs import sector_income_weights
from ikob2.segments.lisa import SECTOR_TO_NACE, SECTORS

TYPES = ("no_wfh", "wfh_possible")
DATA = Path(__file__).resolve().parents[1] / "data" / "occupations"


def tables(**kw):
    base = dict(
        employment=pd.DataFrame({
            "nace_r2": ["B", "B", "C", "C", "F", "F"],
            "isco08": ["OC2", "OC7", "OC2", "OC7", "OC2", "OC7"],
            "employed_thousands": [10.0, 30.0, 30.0, 10.0, 5.0, 15.0]}),
        wages=pd.DataFrame({
            "nace_r2": ["B", "B", "C", "C", "F"],
            "isco08": ["OC2", "OC7", "OC2", "OC7", "OC2"],
            "mean_hourly_eur": [40.0, 20.0, 30.0, 18.0, 32.0]}),
        employment_isco2=None, quartiles=None, crosswalk=None, telework=None)
    base.update(kw)
    return occ.OccupationTables(**base)


def test_sector_cells_weight_wages_by_employment_and_fill_from_the_occupation():
    cells = occ.sector_cells(tables(), {"S1": ("B", "C"), "S2": ("F",)})
    c = cells.set_index(["sector", "isco08"])
    assert c.loc[("S1", "OC2"), "employment"] == 40.0
    # (10 x 40 + 30 x 30) / 40
    assert c.loc[("S1", "OC2"), "mean_wage"] == pytest.approx(32.5)
    assert c.loc[("S1", "OC7"), "mean_wage"] == pytest.approx((30 * 20 + 10 * 18) / 40)
    assert c.loc[("S2", "OC2"), "mean_wage"] == 32.0
    # F publishes no OC7 wage: the employment-weighted OC7 wage over B and C
    assert c.loc[("S2", "OC7"), "mean_wage"] == pytest.approx(19.5)
    with pytest.raises(KeyError):
        occ.sector_cells(tables(), {"S3": ("Z",)})


def test_within_cell_sd_by_hand():
    z = stats.norm.ppf(0.75)
    q = pd.DataFrame({"brc": ["01", "0111", "0112"],
                      "employees_thousands": [30.0, 10.0, 20.0],
                      "p25": [10.0, 10.0, np.nan], "p50": [15.0, 15.0, np.nan],
                      "p75": [22.0, 20.0, np.nan]})
    # 0111 consists of one OC2 unit; 0112 of two OC2 units (no quartiles:
    # those of its class 01 apply)
    xw = pd.DataFrame({"isco08_unit": ["2310", "2320", "2330"],
                       "brc_group": ["0111", "0112", "0112"]})
    sd1, sd2 = np.log(20 / 10) / (2 * z), np.log(22 / 10) / (2 * z)
    mu = np.log(15.0)                          # both medians 15: no between part
    expected = np.sqrt((10 * sd1 ** 2 + 20 * sd2 ** 2) / 30)
    assert occ.within_cell_sd(q, xw) == pytest.approx(expected)
    # different medians add the between-group variance of the log medians
    q2 = q.copy()
    q2.loc[q2.brc == "0111", "p50"] = 15.0 * np.e ** 0.3
    m = (10 * (mu + 0.3) + 20 * mu) / 30
    between = (10 * (mu + 0.3 - m) ** 2 + 20 * (mu - m) ** 2) / 30
    assert occ.within_cell_sd(q2, xw) == pytest.approx(np.sqrt(expected ** 2 + between))


def test_teleworkability_by_major_by_hand():
    tw = pd.DataFrame({"isco08": ["211", "212", "221", "711"],
                       "physical_interaction": [1.0, 0.5, 0.2, 0.0]})
    e2 = pd.DataFrame({"isco08": ["OC21", "OC22", "OC71"],
                       "employed_thousands": [30.0, 10.0, 5.0]})
    with pytest.raises(KeyError):                 # no OC1 rows
        occ.teleworkability_by_major(tw, e2)
    tw_all = pd.concat([tw, pd.DataFrame({
        "isco08": [f"{m}11" for m in "1345689"], "physical_interaction": 0.3})])
    e2_all = pd.concat([e2, pd.DataFrame({
        "isco08": [f"OC{m}1" for m in "1345689"], "employed_thousands": 1.0})])
    t = occ.teleworkability_by_major(tw_all, e2_all)
    # OC21 = mean(1.0, 0.5) = 0.75, OC22 = 0.2: (30 x 0.75 + 10 x 0.2) / 40
    assert t["OC2"] == pytest.approx((30 * 0.75 + 10 * 0.2) / 40)
    assert t["OC7"] == 0.0 and t["OC1"] == pytest.approx(0.3)


def cells_two_sectors():
    return pd.DataFrame({"sector": ["S1", "S1", "S2"],
                         "isco08": ["OC2", "OC9", "OC2"],
                         "employment": [3.0, 1.0, 2.0],
                         "mean_wage": [30.0, 12.0, 25.0]})


def test_weights_partition_the_jobs_and_split_home_working_by_occupation():
    tw = pd.Series({"OC2": 0.8, "OC9": 0.1})
    total = pd.Series({"S1": 400.0, "S2": 100.0})
    m = occ.occupation_job_weights(cells_two_sectors(), 0.4, tw, total, TYPES)
    W = m.by_type["no_wfh"] + m.by_type["wfh_possible"]
    deciles = [f"D{k}" for k in range(1, 11)]
    np.testing.assert_allclose(W.loc[deciles].sum(), 1.0, atol=1e-12)
    np.testing.assert_allclose(W.loc["onbekend"], 1.0, atol=1e-12)
    # home working of S1: its cells' teleworkability by their job shares
    wfh_s1 = m.by_type["wfh_possible"].loc[deciles, "S1"].sum()
    assert wfh_s1 == pytest.approx(0.75 * 0.8 + 0.25 * 0.1)
    # each decile holds a tenth of all jobs
    pools = (W.loc[deciles] * total).sum(axis=1)
    np.testing.assert_allclose(pools, total.sum() / 10, rtol=1e-8)
    # the decile edges are the quantiles of the job-weighted mixture
    jobs = np.array([300.0, 100.0, 100.0])
    mu = np.log([30.0, 12.0, 25.0]) - 0.4 ** 2 / 2
    for k, b in enumerate(m.meta["decile_edges_eur_per_hour"], start=1):
        assert np.sum(jobs / jobs.sum() * stats.norm.cdf((np.log(b) - mu) / 0.4)) \
            == pytest.approx(k / 10, abs=1e-4)   # edges rounded


def test_one_occupation_per_sector_and_no_spread_is_the_sector_method():
    wage = pd.Series({"S1": 15.0, "S2": 22.0, "S3": 31.0})
    total = pd.Series({"S1": 250.0, "S2": 450.0, "S3": 300.0})
    cells = pd.DataFrame({"sector": list(wage.index), "isco08": "OC2",
                          "employment": 1.0, "mean_wage": wage.values})
    m = occ.occupation_job_weights(cells, 1e-6, pd.Series({"OC2": 0.0}),
                                   total, TYPES)
    W0 = sector_income_weights(wage, total)
    np.testing.assert_allclose(m.by_type["no_wfh"].loc[W0.index, W0.columns],
                               W0, atol=1e-6)


def test_sector_job_weights_equal_the_sector_split():
    wage = pd.Series({"S1": 15.0, "S2": 22.0})
    total = pd.Series({"S1": 250.0, "S2": 750.0})
    share = pd.Series({"S1": 0.2, "S2": 0.6})
    m = occ.sector_job_weights(wage, total, share, TYPES)
    W0 = sector_income_weights(wage, total)
    pd.testing.assert_frame_equal(m.by_type["wfh_possible"], W0 * share)
    pd.testing.assert_frame_equal(m.by_type["no_wfh"] + m.by_type["wfh_possible"], W0)
    with pytest.raises(ValueError):
        occ.sector_job_weights(wage, total, share.drop("S2"), TYPES)


def test_jsonstat_frame():
    payload = {"id": ["geo", "isco08"], "size": [1, 2],
               "dimension": {"geo": {"category": {"index": {"NL": 0}}},
                             "isco08": {"category": {"index": {"OC2": 1, "OC1": 0}}}},
               "value": {"1": 7.5}}
    df = occ.jsonstat_frame(payload)
    assert df.isco08.tolist() == ["OC1", "OC2"]
    assert np.isnan(df.value[0]) and df.value[1] == 7.5


def test_the_repository_tables():
    t = occ.load_tables(DATA)
    cells = occ.sector_cells(t, SECTOR_TO_NACE)
    assert set(cells.sector) == set(SECTORS)
    assert (cells.mean_wage > 5).all() and (cells.mean_wage < 100).all()
    sd = occ.within_cell_sd(t.quartiles, t.crosswalk)
    assert 0.2 < sd < 0.6
    tw = occ.teleworkability_by_major(t.telework, t.employment_isco2)
    assert list(tw.index) == list(occ.ISCO_MAJORS)
    assert ((tw >= 0) & (tw <= 1)).all()
    assert tw["OC2"] > 0.5 > tw["OC9"]          # professionals vs elementary
    m = occ.occupation_job_weights(cells, sd, tw,
                                   pd.Series(1000.0, index=list(SECTORS)), TYPES)
    edges = m.meta["decile_edges_eur_per_hour"]
    assert edges == sorted(edges) and 5 < edges[0] < edges[-1] < 60
    with pytest.raises(FileNotFoundError):
        occ.load_tables(DATA / "nowhere")


def test_fetch_tables_writes_every_file_and_a_manifest(tmp_path, monkeypatch):
    """The fetch without network: each URL answers a small fixture."""
    import hashlib
    import io
    import json

    from ikob2.segments import statline

    def js(dims, values):
        ids = list(dims)
        return json.dumps({
            "id": ids, "size": [len(v) for v in dims.values()],
            "dimension": {k: {"category": {"index": {c: i for i, c in enumerate(v)}}}
                          for k, v in dims.items()},
            "value": values}).encode()

    xls = io.BytesIO()
    pd.DataFrame({"ISCO2008unitgroup": ["2310"], "ISCO2008unitgrouplabel": ["x"],
                  "BRC2014beroepsgroep": ["0111"], "BRC2014beroepsgroep_label": ["y"],
                  "BRC2014beroepsklasse": ["01"]}).to_csv(xls, index=False)
    answers = {
        "lfsa_eisn2": js({"nace_r2": ["TOTAL", "C"], "isco08": ["OC2", "OC21"]},
                         {"2": 5.0}),
        "earn_ses22_47": js({"nace_r2": ["C"], "isco08": ["OC2"]}, {"0": 30.5}),
        "lfsa_egai2d": js({"isco08": ["OC2", "OC21"]}, {"1": 7.0}),
        ".xls": b"xls", "zenodo": b"ISCO08,Occupation Title,Physical interaction,"
                                  b"Social interaction\n211,Sci,1.0,0.5\n"}

    def fake_get(url):
        return next(v for k, v in answers.items() if k in url)

    monkeypatch.setattr(occ, "_get", fake_get)
    monkeypatch.setattr(pd, "read_excel", lambda b, **kw: pd.read_csv(xls.seek(0) or xls,
                                                                     dtype=str))
    monkeypatch.setattr(statline, "_odata_get", lambda *a, **k: pd.DataFrame({
        "Beroep": ["A1"], "Perioden": ["2022JJ00"], "Werknemer_1": [9.0],
        "k_25ePercentiel_2": [1.0], "k_50ePercentielMediaan_3": [2.0],
        "k_75ePercentiel_4": [3.0]}))
    monkeypatch.setattr(statline, "_odata_get_dimension",
                        lambda *a, **k: {"A1": "0111 Docenten"})
    manifest = occ.fetch_tables(tmp_path)
    assert set(manifest) == set(occ.FILES.values())
    for name, m in manifest.items():
        assert m["sha256"] == hashlib.sha256((tmp_path / name).read_bytes()).hexdigest()
    assert json.loads((tmp_path / "sources.json").read_text()) == manifest
    assert "lfsa_eisn2?geo=NL" in manifest["lfs_nace_isco_nl.csv"]["url"]
    emp = pd.read_csv(tmp_path / "lfs_nace_isco_nl.csv")
    assert emp.to_dict("records") == [{"nace_r2": "C", "isco08": "OC2",
                                       "employed_thousands": 5.0}]
    q = pd.read_csv(tmp_path / "brc_wage_quartiles.csv", dtype={"brc": str})
    assert q.brc.tolist() == ["0111"] and q.p50.tolist() == [2.0]
    tw = pd.read_csv(tmp_path / "teleworkability_isco3.csv", dtype={"isco08": str})
    assert tw.columns.tolist() == ["isco08", "title", "physical_interaction",
                                   "social_interaction"]
