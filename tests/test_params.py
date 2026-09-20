"""Parameters: one defaults file, overridable by file, --set and flags."""

import argparse

import pytest

from ikob2 import params
from ikob2.skims.car import CarCostModel
from ikob2.skims.gtfs_pt import LegSpec, PtRouter  # noqa: F401
from ikob2.skims.pt_fare import PtFareModel
from ikob2.utils.paths import DataLayout


def test_defaults_feed_library_defaults():
    d = params.DEFAULTS
    assert CarCostModel().variable_eur_per_km == d.car.models.fossil.variable_eur_per_km
    assert LegSpec().kmh == d.bike_leg.kmh
    assert PtFareModel().regional_boarding_eur == d.pt_fare.regional_boarding_eur


def test_file_overrides_defaults_and_keeps_the_rest(tmp_path):
    f = tmp_path / "p.toml"
    f.write_text("[pt]\nwalk_kmh = 5.0\n")
    p = params.load(f)
    assert p.pt.walk_kmh == 5.0
    assert p.pt.walk_detour == params.DEFAULTS.pt.walk_detour


def test_set_overrides_file(tmp_path):
    f = tmp_path / "p.toml"
    f.write_text("[pt]\nwalk_kmh = 5.0\n")
    p = params.load(f, ["pt.walk_kmh=4.5", 'accessibility.modes=["pt"]'])
    assert p.pt.walk_kmh == 4.5
    assert p.accessibility.modes == ["pt"]


def test_unknown_keys_and_wrong_types_are_errors(tmp_path):
    with pytest.raises(KeyError):
        params.load(assignments=["pt.walkk=1"])
    f = tmp_path / "p.toml"
    f.write_text("[nonsense]\nx = 1\n")
    with pytest.raises(KeyError):
        params.load(f)
    with pytest.raises(TypeError):
        params.load(assignments=["pt.walk_kmh=fast"])
    with pytest.raises(TypeError):
        params.load(assignments=["car.parking_search=1"])


def test_open_tables_accept_new_keys():
    p = params.load(assignments=["car.models.hybrid={variable_eur_per_km=0.1}"])
    assert p.car.models.hybrid.variable_eur_per_km == 0.1
    assert CarCostModel.from_params(p.car.models.hybrid).variable_eur_per_km == 0.1


def test_flags_override_only_when_given():
    ns = argparse.Namespace(params=None, set_values=["pt.walk_kmh=4.5"],
                            walk_kmh=None, walk_detour=1.1)
    p = params.from_args(ns, {"walk_kmh": "pt.walk_kmh",
                              "walk_detour": "pt.walk_detour"})
    assert p.pt.walk_kmh == 4.5 and p.pt.walk_detour == 1.1


def test_params_are_read_only():
    with pytest.raises(AttributeError):
        params.DEFAULTS.pt = 1
    assert params.DEFAULTS.with_values({"pt.walk_kmh": 9.0}).pt.walk_kmh == 9.0
    assert params.DEFAULTS.pt.walk_kmh != 9.0


def test_no_built_in_data_root(monkeypatch, tmp_path):
    monkeypatch.delenv(params.ENV_ROOT, raising=False)
    with pytest.raises(SystemExit):
        params.data_root(None, params.DEFAULTS)
    monkeypatch.setenv(params.ENV_ROOT, str(tmp_path))
    assert params.data_root(None, params.DEFAULTS) == tmp_path
    assert params.data_root("/x", params.DEFAULTS).as_posix() == "/x"
    with pytest.raises(TypeError):
        DataLayout()          # the root is required


def test_pt_fare_from_params_uses_a_table_file(tmp_path):
    t = tmp_path / "t.csv"
    t.write_text("km,eur\n0,2\n10,4\n")
    p = params.load(assignments=[f'pt_fare.rail_table="{t}"'])
    m = PtFareModel.from_params(p.pt_fare)
    assert m.rail_table == ((0.0, 2.0), (10.0, 4.0))
