"""Peak-load speeds by road class."""

import pytest

from ikob2.skims import peak


def test_parse_speed():
    assert peak.parse_speed_kmh("50") == 50
    assert peak.parse_speed_kmh(" 100 ") == 100
    assert peak.parse_speed_kmh("30 mph") == pytest.approx(48.28, abs=0.01)
    for bad in ("none", "walk", "signals", "NL:urban", "50;30", "", None):
        assert peak.parse_speed_kmh(bad) is None


def test_scale_speed_rounds_and_floors():
    assert peak.scale_speed("100", 1.4) == "71"
    assert peak.scale_speed("130", 1.4) == "93"
    assert peak.scale_speed("80", 1.2) == "67"
    assert peak.scale_speed("50", 1.05) == "48"
    assert peak.scale_speed("30", 1.05) == "29"
    assert peak.scale_speed("6", 2.0) == "5"          # never below 5 km/h
    assert peak.scale_speed("walk", 1.4) is None
    assert peak.scale_speed("50", 0) is None


def test_factors_follow_the_stated_classes():
    f = peak.PEAK_FACTORS
    assert f["motorway"] == f["trunk"] == f["motorway_link"] == 1.4
    assert f["primary"] == f["secondary"] == f["secondary_link"] == 1.2
    assert f["tertiary"] == f["residential"] == 1.05
    assert "cycleway" not in f and "footway" not in f


def test_peak_tags():
    assert peak.peak_tags({"highway": "motorway", "maxspeed": "100"}) == {
        "highway": "motorway", "maxspeed": "71"}
    # directional tags are scaled too, other tags kept
    t = peak.peak_tags({"highway": "primary", "maxspeed:forward": "80",
                        "maxspeed:backward": "60", "name": "N1"})
    assert t == {"highway": "primary", "maxspeed:forward": "67",
                 "maxspeed:backward": "50", "name": "N1"}
    # left alone: other classes, no maxspeed, symbolic maxspeed
    assert peak.peak_tags({"highway": "cycleway", "maxspeed": "30"}) is None
    assert peak.peak_tags({"highway": "motorway"}) is None
    assert peak.peak_tags({"highway": "residential", "maxspeed": "NL:urban"}) is None


def test_extract_roundtrip_on_a_tiny_file(tmp_path):
    osmium = pytest.importorskip("osmium")
    src, dst = tmp_path / "a.osm.pbf", tmp_path / "b.osm.pbf"
    with osmium.SimpleWriter(str(src)) as w:
        w.add_node(osmium.osm.mutable.Node(id=1, location=(5.0, 52.0)))
        w.add_node(osmium.osm.mutable.Node(id=2, location=(5.1, 52.0)))
        w.add_way(osmium.osm.mutable.Way(
            id=10, nodes=[1, 2], tags={"highway": "motorway", "maxspeed": "100"}))
        w.add_way(osmium.osm.mutable.Way(
            id=11, nodes=[1, 2], tags={"highway": "footway", "maxspeed": "5"}))
        w.add_way(osmium.osm.mutable.Way(
            id=12, nodes=[1, 2], tags={"highway": "residential"}))
    stats = peak.make_peak_extract(src, dst)
    assert stats["motorway:ways"] == 1 and stats["motorway:rescaled"] == 1
    assert stats["residential:no_maxspeed"] == 1
    out = {}
    for o in osmium.FileProcessor(str(dst)):
        if o.is_way():
            out[o.id] = dict(o.tags)
    assert out[10]["maxspeed"] == "71" and out[11]["maxspeed"] == "5"
    assert "maxspeed" not in out[12]
    assert peak.coverage(stats)["motorway"]["share_rescaled"] == 1.0
