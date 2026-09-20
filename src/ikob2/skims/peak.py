"""
A peak-load OSM extract: congestion as lower speeds by road class.

R5 and Valhalla route with the speeds in the OSM data, i.e. free flow.
To mimic a peak load, the `maxspeed` of every way of a road class is
divided by a congestion factor (travel time on that class multiplied by
it), so that the routers also let the congestion shape route choice:

    motorway, trunk                          1.40   TomTom Traffic Index
    primary, secondary                       1.20   Monitor Nationale
                                                    Omgevingsvisie,
                                                    Indicatoren Bereikbaarheid
    tertiary, residential (and unclassified,
    living_street)                           1.05   minor interactions in
                                                    the streets

(the `_link` classes follow their parent class). Only ways that carry a
numeric `maxspeed` are rescaled; ways without one keep the router's
default speed and therefore no congestion, which is a small share for the
classes that matter (reported by the function). Speeds are rounded to
whole km/h, so the realised factor differs slightly from the nominal one
(largest for slow streets: 30 km/h / 1.05 = 28.6 becomes 29).
Directional tags `maxspeed:forward` and `maxspeed:backward` are scaled
too.
"""

from __future__ import annotations

import logging
import re
from collections import Counter
from pathlib import Path

from ikob2.params import DEFAULTS

logger = logging.getLogger(__name__)

PEAK_FACTORS: dict[str, float] = {
    k: float(v) for k, v in DEFAULTS.peak.factors.to_dict().items()}

_NUMBER = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*(mph|km/h|kmh)?\s*$")
_SPEED_TAGS = ("maxspeed", "maxspeed:forward", "maxspeed:backward")


def parse_speed_kmh(value: str) -> float | None:
    """Numeric km/h of a maxspeed value; None for 'none', 'walk',
    'signals', lists, symbolic values such as 'NL:urban'."""
    m = _NUMBER.match(value or "")
    if not m:
        return None
    speed = float(m.group(1))
    return speed * 1.609344 if m.group(2) == "mph" else speed


def scale_speed(value: str, factor: float,
                minimum: float = DEFAULTS.peak.minimum_kmh) -> str | None:
    """maxspeed value divided by `factor`, rounded to whole km/h and
    never below `minimum`; None if the value is not numeric."""
    v = parse_speed_kmh(value)
    if v is None or factor <= 0:
        return None
    return str(int(max(minimum, round(v / factor))))


def peak_tags(tags: dict, factors: dict[str, float] = PEAK_FACTORS) -> dict | None:
    """New tags for a way, or None if the way is left alone."""
    factor = factors.get(tags.get("highway", ""))
    if factor is None:
        return None
    new = dict(tags)
    changed = False
    for key in _SPEED_TAGS:
        if key in new:
            scaled = scale_speed(new[key], factor)
            if scaled is not None and scaled != new[key]:
                new[key] = scaled
                changed = True
    return new if changed else None


def make_peak_extract(src: str | Path, dst: str | Path,
                      factors: dict[str, float] = PEAK_FACTORS) -> dict:
    """Write `dst`: `src` with the peak speeds applied. Returns counts of
    ways per highway class: total, rescaled, without a usable maxspeed."""
    import osmium

    stats: Counter = Counter()
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    if Path(dst).exists():
        Path(dst).unlink()
    with osmium.SimpleWriter(str(dst)) as writer:
        for obj in osmium.FileProcessor(str(src)):
            if obj.is_way():
                cls = obj.tags.get("highway", "")
                if cls in factors:
                    stats[f"{cls}:ways"] += 1
                    new = peak_tags(dict(obj.tags), factors)
                    if new is not None:
                        stats[f"{cls}:rescaled"] += 1
                        writer.add_way(obj.replace(tags=new))
                        continue
                    if not any(k in obj.tags for k in _SPEED_TAGS):
                        stats[f"{cls}:no_maxspeed"] += 1
            writer.add(obj)
    return dict(stats)


def coverage(stats: dict) -> dict[str, dict]:
    """Per class: ways, share rescaled."""
    out = {}
    for cls in PEAK_FACTORS:
        n = stats.get(f"{cls}:ways", 0)
        if n:
            out[cls] = {"ways": n,
                        "rescaled": stats.get(f"{cls}:rescaled", 0),
                        "share_rescaled": stats.get(f"{cls}:rescaled", 0) / n}
    return out
