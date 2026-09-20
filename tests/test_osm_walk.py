from ikob2.skims.osm_walk import keep_way


def test_pedestrian_ways_kept():
    assert keep_way({"highway": "residential"})
    assert keep_way({"highway": "footway"})
    assert keep_way({"public_transport": "platform"})


def test_car_only_and_restricted_ways_dropped():
    assert not keep_way({"highway": "motorway"})
    assert not keep_way({"highway": "trunk"})
    assert not keep_way({"highway": "service"})
    assert not keep_way({"building": "yes"})
    assert not keep_way({"highway": "residential", "foot": "no"})
    assert not keep_way({"highway": "track", "access": "private"})
