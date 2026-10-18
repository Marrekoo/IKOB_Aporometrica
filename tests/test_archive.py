"""The data deposit of the paper runs (cli.archive)."""

import hashlib
import tarfile

import pytest

from ikob2.cli.archive import ODIN_AGGREGATES, build, contents
from ikob2.params import DEFAULTS
from ikob2.utils.paths import DataLayout

PLAN = """
[specs]
m2 = ["--spec", "m2"]
[scenarios]
s0 = []
s1 = ["--price-scales", "half.csv"]
[[sets]]
prefix = "sp"
"""


def folder(tmp_path):
    """A data folder with one small file for every item of the deposit."""
    lay = DataLayout(tmp_path / "root")
    p = DEFAULTS.paths
    files = [lay.inputs / "gtfs" / "gtfs-nl.zip", lay.inputs / "osm" / p.osm_national,
             *[lay.inputs / "osm" / f"{r}.osm.pbf" for r in p.osm_regions],
             lay.kwb(DEFAULTS.accessibility.kwb_year, p.kwb_version), lay.lisa(),
             lay.intermediate / "skims" / p.study / "manifest.json",
             lay.intermediate / "skims" / f"{p.study}_peak" / "all" / "t.npy",
             lay.sector_jobs(DEFAULTS.accessibility.jobs_year),
             *[lay.intermediate / f for f in ODIN_AGGREGATES],
             lay.run_dir("sp_m2_s0") / "accessibility.csv",
             lay.run_dir("sp_m2_s1") / "run.json",
             lay.comparison_dir() / "specs" / "baseline_by_spec.csv",
             lay.comparison_dir() / "targeting" / "targeting_summary.csv",
             lay.comparison_dir() / "precision" / "precision.csv",
             lay.comparison_dir() / "other" / "not_in_the_deposit.csv"]
    for i, f in enumerate(files):
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(f"file {i}\n")
    plan = tmp_path / "plan.toml"
    plan.write_text(PLAN)
    return lay, plan


def test_contents_cover_inputs_skims_runs_and_tables(tmp_path):
    lay, plan = folder(tmp_path)
    c = contents(lay, DEFAULTS, [plan])
    rel = {k: {f.relative_to(lay.root).as_posix() for f in v} for k, v in c.items()}
    assert len(rel["inputs.tar"]) == 4 + len(DEFAULTS.paths.osm_regions)
    # both skim stores of the study area, the jobs; no ODiN aggregates by default
    assert {"intermediate/skims/utrecht_nl/manifest.json",
            "intermediate/skims/utrecht_nl_peak/all/t.npy",
            "intermediate/jobs/sector_jobs_2022.csv"} == rel["intermediate.tar"]
    # the runs of the plan and the three paper tables, nothing else
    assert rel["outputs.tar.gz"] == {
        "outputs/runs/sp_m2_s0/accessibility.csv", "outputs/runs/sp_m2_s1/run.json",
        "outputs/comparisons/specs/baseline_by_spec.csv",
        "outputs/comparisons/targeting/targeting_summary.csv",
        "outputs/comparisons/precision/precision.csv"}
    with_odin = contents(lay, DEFAULTS, [plan], odin_aggregates=True)
    assert len(with_odin["intermediate.tar"]) == 3 + len(ODIN_AGGREGATES)


def test_build_writes_archives_and_a_checkable_manifest(tmp_path):
    lay, plan = folder(tmp_path)
    out = tmp_path / "deposit"
    counts = build(lay, DEFAULTS, [plan], out)
    lines = (out / "MANIFEST.sha256").read_text().splitlines()
    # 4 + 4 provincial extracts inputs, 3 intermediate, 5 outputs
    assert counts == {"inputs.tar": 8, "intermediate.tar": 3, "outputs.tar.gz": 5}
    assert len(lines) == 16
    for line in lines:
        digest, rel = line.split("  ")
        assert digest == hashlib.sha256((lay.root / rel).read_bytes()).hexdigest()
    with tarfile.open(out / "outputs.tar.gz") as tar:
        names = tar.getnames()
        member = tar.extractfile("outputs/runs/sp_m2_s0/accessibility.csv").read()
    assert len(names) == counts["outputs.tar.gz"]
    assert member == (lay.run_dir("sp_m2_s0") / "accessibility.csv").read_bytes()
    assert "ODiN microdata (DANS)" in (out / "README.md").read_text()


def test_missing_input_is_an_error(tmp_path):
    lay, plan = folder(tmp_path)
    (lay.inputs / "gtfs" / "gtfs-nl.zip").unlink()
    with pytest.raises(FileNotFoundError, match="gtfs-nl.zip"):
        contents(lay, DEFAULTS, [plan])


def _fake_zenodo(monkeypatch, deposition):
    """Record the API calls of `upload` against a fake Zenodo."""
    from ikob2.cli import archive

    calls = []

    def call(method, url, token, data=None, json_body=None, size=None):
        calls.append((method, url, json_body))
        if method == "GET":
            return deposition
        if method == "POST":
            return {"id": 9, "links": {"bucket": "B", "html": "new"}}
        return {}

    monkeypatch.setattr(archive, "_call", call)
    return calls


def test_upload_to_an_existing_draft_replaces_files_and_keeps_its_metadata(
        tmp_path, monkeypatch):
    from ikob2.cli.archive import upload

    (tmp_path / "outputs.tar.gz").write_bytes(b"x")
    draft = {"id": 7, "submitted": False,
             "links": {"bucket": "B7", "html": "draft"},
             "files": [{"links": {"self": "F1"}}, {"links": {"self": "F2"}}],
             "metadata": {"prereserve_doi": {"doi": "10.5281/zenodo.7"},
                          "title": "old"}}
    calls = _fake_zenodo(monkeypatch, draft)
    assert upload(tmp_path, {"title": "new"}, "t", deposition=7) == "draft"
    assert [(m, u) for m, u, _ in calls] == [
        ("GET", "https://zenodo.org/api/deposit/depositions/7"),
        ("DELETE", "F1"), ("DELETE", "F2"), ("PUT", "B7/outputs.tar.gz"),
        ("PUT", "https://zenodo.org/api/deposit/depositions/7")]
    # the reserved DOI stays, the metadata file wins where both have a key
    assert calls[-1][2] == {"metadata": {
        "prereserve_doi": {"doi": "10.5281/zenodo.7"}, "title": "new"}}


def test_upload_refuses_a_published_deposition(tmp_path, monkeypatch):
    from ikob2.cli.archive import upload

    calls = _fake_zenodo(monkeypatch, {"id": 7, "submitted": True})
    with pytest.raises(SystemExit, match="published"):
        upload(tmp_path, {}, "t", deposition=7)
    assert [m for m, _, _ in calls] == ["GET"]


def test_upload_without_a_deposition_creates_a_new_draft(tmp_path, monkeypatch):
    from ikob2.cli.archive import upload

    (tmp_path / "README.md").write_text("r")
    calls = _fake_zenodo(monkeypatch, None)
    assert upload(tmp_path, {"title": "t"}, "t") == "new"
    assert [m for m, _, _ in calls] == ["POST", "PUT", "PUT"]
