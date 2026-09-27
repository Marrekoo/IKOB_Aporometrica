"""Path resolution of the accessibility command line."""

import argparse

import pytest

from ikob2 import params as params_mod
from ikob2.cli import accessibility as cli
from ikob2.utils.paths import REPO_DATA, DataLayout


def _args(**over):
    base = dict(data_root=None, kwb=None, skims=None, sector_jobs=None,
                out=None, statline=None, bike_ownership=None,
                car_availability=None, detour=None, margins=None,
                budgets="reference_budgets.csv",
                price_scales="lime_price_scales.csv", pt_fare_scales="",
                study="utrecht_nl", run="r", kwb_year=2022, jobs_year=2022)
    base.update(over)
    return argparse.Namespace(**base)


@pytest.fixture(autouse=True)
def no_env_root(monkeypatch):
    monkeypatch.delenv(params_mod.ENV_ROOT, raising=False)


def test_reference_files_come_from_the_seeded_data_folder(tmp_path):
    lay = DataLayout(tmp_path)
    lay.ensure()
    lay.seed(REPO_DATA)
    args = _args(data_root=str(tmp_path))
    cli.resolve_paths(args, params_mod.DEFAULTS)
    assert args.budgets == str(lay.budgets())
    assert args.margins == str(lay.survey_margins())
    assert args.price_scales == str(lay.tariff("lime_price_scales.csv"))
    assert args.statline == str(lay.statline())
    assert args.out == str(lay.run_dir("r"))


def test_unseeded_folder_names_the_missing_reference_files(tmp_path):
    DataLayout(tmp_path).ensure()
    with pytest.raises(SystemExit, match="--budgets .*reference_budgets.csv"):
        cli.resolve_paths(_args(data_root=str(tmp_path)), params_mod.DEFAULTS)


def test_without_a_root_paths_must_be_explicit():
    with pytest.raises(SystemExit, match="--kwb.*--statline, --margins"):
        cli.resolve_paths(_args(), params_mod.DEFAULTS)
