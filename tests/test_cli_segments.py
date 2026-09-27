"""Parameters and paths of the segments command line."""

import pytest

from ikob2.cli import segments as cli


@pytest.fixture
def parse(monkeypatch):
    """Parse a command line like main() does, without running the command."""
    monkeypatch.delenv("IKOB_DATA_ROOT", raising=False)
    seen = []
    for name in ("cmd_fetch", "cmd_run", "cmd_jobs", "cmd_car", "cmd_pt_spend"):
        monkeypatch.setattr(cli, name, seen.append)

    def run(argv):
        cli.main(argv)
        return seen[-1]
    return run


def test_values_come_from_the_parameters(parse):
    args = parse(["jobs"])
    prm = cli.resolve(args)
    assert args.establishment_weight == prm.jobs.establishment_weight == 0.25
    assert args.year == prm.accessibility.jobs_year
    args = parse(["--set", "jobs.establishment_weight=0.5", "jobs"])
    cli.resolve(args)
    assert args.establishment_weight == 0.5
    args = parse(["--set", "jobs.establishment_weight=0.5", "jobs",
                  "--establishment-weight", "0.1"])
    cli.resolve(args)
    assert args.establishment_weight == 0.1          # the flag wins
    args = parse(["car-availability", "--prior", "12"])
    cli.resolve(args)
    assert (args.municipality, args.car_prior) == (344, 12.0)


def test_paths_come_from_the_layout(tmp_path, parse):
    args = parse(["--data-root", str(tmp_path), "pt-spend"])
    prm = cli.resolve(args)
    cli.fill(args, cli.layout(args, prm), odin=cli.DataLayout.odin,
             out=lambda lay: lay.pt_spend())
    assert args.odin == str(tmp_path / "inputs/odin/ODIN_22_23_clean.csv")
    assert args.out == str(tmp_path / "intermediate/ownership/pt_spend_utrecht.csv")


def test_missing_paths_without_a_root_are_named(parse):
    args = parse(["car-availability", "--odin", "x.csv"])
    prm = cli.resolve(args)
    with pytest.raises(SystemExit, match="--out or --data-root"):
        cli.fill(args, cli.layout(args, prm), odin=cli.DataLayout.odin,
                 out=cli.DataLayout.car_availability)
