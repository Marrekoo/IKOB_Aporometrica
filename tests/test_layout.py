"""Data folder layout."""

from pathlib import Path

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
    assert lay.s2_hubs() == tmp_path / "intermediate/hubs/utrecht_hubs_s2.csv"
    assert lay.odin().parent == tmp_path / "inputs/odin"
    assert lay.pt_spend().parent == lay.car_availability().parent


def test_every_conventional_folder_is_created(tmp_path):
    lay = DataLayout(tmp_path)
    lay.ensure()
    for path in (lay.bike_ownership(), lay.s2_hubs(), lay.pt_spend(),
                 lay.lisa(), lay.ikob_jobs(), lay.odin(), lay.segments_gpkg(),
                 lay.osm_walk_dir() / "x"):
        assert path.parent.is_dir(), path
    for sub in ("hubs", "ovfiets", "veh_owners"):
        assert (lay.inputs / sub).is_dir()


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


def test_seed_copies_missing_reference_files_and_never_overwrites(tmp_path):
    from ikob2.utils.paths import REPO_DATA
    lay = DataLayout(tmp_path / "data")
    lay.ensure()
    (lay.statline() / "86161NED_2022JJ00.csv").write_text("mine")
    copied = lay.seed(REPO_DATA)
    names = {p.relative_to(lay.root).as_posix() for p in copied}
    assert "inputs/envelope/reference_budgets.csv" in names
    assert "inputs/survey/S_T_work.csv" in names
    assert "inputs/tariffs/lime_price_scales.csv" in names
    assert "intermediate/calibration/car_detour.json" in names
    assert "cache/statline/86161NED_2022JJ00.csv" not in names
    assert (lay.statline() / "86161NED_2022JJ00.csv").read_text() == "mine"
    assert lay.budgets().exists() and lay.survey_margins().exists()
    assert lay.seed(REPO_DATA) == []                 # nothing new the second time
    with pytest.raises(FileNotFoundError):
        lay.seed(tmp_path / "nowhere")


def test_reference_names_resolve_in_their_folder(tmp_path, monkeypatch):
    from ikob2.utils.paths import resolve_input
    lay = DataLayout(tmp_path)
    assert lay.budgets() == tmp_path / "inputs/envelope/reference_budgets.csv"
    assert lay.tariff("x.csv") == tmp_path / "inputs/tariffs/x.csv"
    own = tmp_path / "elsewhere.csv"
    own.write_text("a")
    assert resolve_input(own, lay.inputs / "envelope") == own     # exists as given
    monkeypatch.chdir(tmp_path)                                  # relative to cwd
    assert resolve_input("elsewhere.csv", lay.inputs / "envelope") == \
        Path("elsewhere.csv")
    assert resolve_input("absent.csv", lay.inputs / "envelope") == \
        tmp_path / "inputs/envelope/absent.csv"
    assert resolve_input("b.csv", None) == Path("b.csv")
