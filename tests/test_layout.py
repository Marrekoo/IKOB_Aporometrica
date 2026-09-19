"""Data folder layout."""

import pytest

from ikob2.cli.layout import link_input
from ikob2.utils.paths import INPUT_DIRS, DataLayout


def test_ensure_creates_folders_readme_and_is_idempotent(tmp_path):
    lay = DataLayout(tmp_path / "data")
    created = lay.ensure()
    assert len(created) == len(lay.all_dirs())
    for sub in INPUT_DIRS:
        assert (lay.inputs / sub).is_dir()
    assert (lay.statline()).is_dir() and (lay.outputs / "runs").is_dir()
    assert "inputs/" in (lay.root / "README.md").read_text()
    assert lay.ensure() == []                    # nothing new the second time


def test_ensure_never_overwrites_existing_readme_or_files(tmp_path):
    root = tmp_path / "data"
    root.mkdir()
    (root / "README.md").write_text("mine")
    (root / "old.gpkg").write_text("x")
    DataLayout(root).ensure()
    assert (root / "README.md").read_text() == "mine"
    assert (root / "old.gpkg").read_text() == "x"


def test_conventional_paths(tmp_path):
    lay = DataLayout(tmp_path)
    assert lay.kwb(2022).name == "wijkenbuurten_2022_v3.gpkg"
    assert lay.skim_dir("utrecht_nl") == tmp_path / "intermediate/skims/utrecht_nl"
    assert lay.sector_jobs(2022).name == "sector_jobs_2022.csv"
    assert lay.run_dir("s0") == tmp_path / "outputs/runs/s0"
    assert lay.detour_model().parent.name == "calibration"


def test_link_input(tmp_path):
    lay = DataLayout(tmp_path / "data")
    lay.ensure()
    src = tmp_path / "elsewhere.gpkg"
    src.write_text("g")
    dest = link_input(lay, src, "kwb", "wijkenbuurten_2022_v3.gpkg")
    assert dest.is_symlink() and dest.read_text() == "g"
    assert link_input(lay, src, "kwb", "wijkenbuurten_2022_v3.gpkg") == dest
    other = tmp_path / "other.gpkg"
    other.write_text("h")
    with pytest.raises(FileExistsError):
        link_input(lay, other, "kwb", "wijkenbuurten_2022_v3.gpkg")
    with pytest.raises(ValueError, match="one of"):
        link_input(lay, src, "nope")
    with pytest.raises(FileNotFoundError):
        link_input(lay, tmp_path / "missing", "kwb")
