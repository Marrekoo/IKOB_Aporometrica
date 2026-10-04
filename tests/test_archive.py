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
