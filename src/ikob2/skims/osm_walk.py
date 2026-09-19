"""
A walking-network OSM extract, for a national OpenTripPlanner graph.

The full national extract does not fit in 15 GB once OpenTripPlanner builds
its street graph (killed at 10.5 GB). Public transport itineraries only need
the pedestrian network to reach stops, so this writes a lean copy: ways whose
highway class a pedestrian can use (no motorways, trunk roads, private
service roads), only the nodes those ways refer to, and no relations.
"""

from __future__ import annotations

import logging
from array import array
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# pedestrians can use these; motorway, trunk, service, raceway and the
# like are left out
WALKABLE = frozenset({
    "primary", "primary_link", "secondary", "secondary_link", "tertiary",
    "tertiary_link", "unclassified", "residential", "living_street",
    "pedestrian", "footway", "path", "steps", "track", "cycleway",
    "corridor", "bridleway", "platform",
})


def keep_way(tags) -> bool:
    if tags.get("highway") in WALKABLE:
        return tags.get("foot") not in ("no", "private") \
            and tags.get("access") not in ("no", "private")
    return tags.get("public_transport") == "platform" \
        or tags.get("railway") == "platform"


def make_walk_extract(src: str | Path, dst: str | Path) -> dict:
    """Write `dst` from `src`; returns counts of ways and nodes kept."""
    import osmium

    refs = array("q")
    n_ways = n_kept = 0
    for w in osmium.FileProcessor(str(src), osmium.osm.WAY):
        n_ways += 1
        if keep_way(w.tags):
            n_kept += 1
            refs.extend(n.ref for n in w.nodes)
    ids = np.frombuffer(refs, dtype=np.int64)
    bitmap = np.zeros(int(ids.max() // 8) + 1, dtype=np.uint8)
    np.bitwise_or.at(bitmap, ids >> 3, (1 << (ids & 7)).astype(np.uint8))
    bits = bytes(bitmap)
    del refs, ids, bitmap
    logger.info("%d of %d ways kept", n_kept, n_ways)

    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    if Path(dst).exists():
        Path(dst).unlink()
    n_nodes = 0
    size = len(bits)
    with osmium.SimpleWriter(str(dst)) as writer:
        for obj in osmium.FileProcessor(str(src), osmium.osm.NODE
                                        | osmium.osm.WAY):
            if obj.is_node():
                i = obj.id
                if (i >> 3) < size and (bits[i >> 3] >> (i & 7)) & 1:
                    writer.add_node(obj)
                    n_nodes += 1
            elif keep_way(obj.tags):
                writer.add_way(obj)
    return {"ways_total": n_ways, "ways_kept": n_kept, "nodes_kept": n_nodes}
