"""Run plans (cli.batch)."""

import tomllib
from pathlib import Path

import pytest

from ikob2.cli.batch import command, expand

PLAN = {
    "specs": {"m1": ["--spec", "m1"], "m3t2": ["--spec", "m3", "--theta", "2"]},
    "scenarios": {"s0": [], "s1": ["--price-scales", "half.csv"]},
    "sets": [
        {"prefix": "sp", "args": ["--modes", "pt"]},
        {"prefix": "scen", "name": "{prefix}_{scenario}", "specs": ["m1"],
         "scenarios": ["s1"], "args": ["--report-usage"]},
    ],
}


def test_expand_crosses_specs_and_scenarios_in_order():
    runs = expand(PLAN)
    assert [n for n, _ in runs] == ["sp_m1_s0", "sp_m1_s1", "sp_m3t2_s0",
                                    "sp_m3t2_s1", "scen_s1"]
    # set arguments, then the specification's, then the scenario's
    assert dict(runs)["sp_m3t2_s1"] == ["--modes", "pt", "--spec", "m3", "--theta",
                                        "2", "--price-scales", "half.csv"]
    assert dict(runs)["scen_s1"] == ["--report-usage", "--spec", "m1",
                                     "--price-scales", "half.csv"]


def test_expand_rejects_unknown_names_and_duplicate_runs():
    with pytest.raises(ValueError, match="unknown"):
        expand({**PLAN, "sets": [{"specs": ["m9"]}]})
    twice = {**PLAN, "sets": [{"name": "x", "specs": ["m1"], "scenarios": ["s0"]}] * 2}
    with pytest.raises(ValueError, match="twice"):
        expand(twice)


def test_command_puts_the_data_folder_in_place_of_root():
    cmd = command(Path("/d"), "r", ["--bike-ownership", "{root}/inputs/x.csv"])
    assert cmd[-4:] == ["--run", "r", "--bike-ownership", "/d/inputs/x.csv"]


def test_the_paper_plan_expands_to_unique_runs():
    plan = tomllib.loads((Path(__file__).resolve().parents[1] / "paper"
                          / "runs.toml").read_text())
    runs = expand(plan)
    names = [n for n, _ in runs]
    assert len(names) == len(set(names)) == 11 * 9 + 11 * 7 + 4 * 3 + 9 + 7
    assert "scen_s4a_nobike" in names and "spt_m2_s2t" in names
