import numpy as np
import pytest

from ikob2.domain.zones import ZoneSet


def make_zones(n=3):
    return ZoneSet(
        codes=np.array([f"BU{i:08d}" for i in range(n)]),
        names=np.array([f"zone {i}" for i in range(n)]),
        centroid_x=np.arange(n, dtype=np.float64) * 100.0,
        centroid_y=np.arange(n, dtype=np.float64) * 50.0,
        crs="EPSG:28992",
        attributes={"pop": np.arange(n, dtype=np.float64) * 10.0},
    )


def test_basic_construction():
    zones = make_zones(3)
    assert zones.n_zones == 3
    assert zones.code_to_index["BU00000001"] == 1


def test_duplicate_codes_rejected():
    with pytest.raises(ValueError):
        ZoneSet(
            codes=np.array(["A", "A"]),
            names=np.array(["x", "y"]),
            centroid_x=np.array([0.0, 1.0]),
            centroid_y=np.array([0.0, 1.0]),
            crs="EPSG:28992",
        )


def test_mismatched_length_rejected():
    with pytest.raises(ValueError):
        ZoneSet(
            codes=np.array(["A", "B"]),
            names=np.array(["x"]),
            centroid_x=np.array([0.0, 1.0]),
            centroid_y=np.array([0.0, 1.0]),
            crs="EPSG:28992",
        )


def test_attribute_lookup():
    zones = make_zones(3)
    assert np.array_equal(zones.attribute("pop"), [0.0, 10.0, 20.0])
    with pytest.raises(KeyError):
        zones.attribute("missing")


def test_subset():
    zones = make_zones(4)
    subset = zones.subset(np.array([False, True, False, True]))
    assert subset.n_zones == 2
    assert list(subset.codes) == ["BU00000001", "BU00000003"]
    assert np.array_equal(subset.attributes["pop"], [10.0, 30.0])


def test_reorder_matches_external_ordering():
    zones = make_zones(3)
    reordered = zones.reorder(["BU00000002", "BU00000000", "BU00000001"])
    assert list(reordered.codes) == [
        "BU00000002", "BU00000000", "BU00000001"
    ]
    assert np.array_equal(reordered.attributes["pop"], [20.0, 0.0, 10.0])


def test_reorder_missing_code_raises():
    zones = make_zones(3)
    with pytest.raises(KeyError):
        zones.reorder(["BU00000000", "NOT_A_ZONE"])
