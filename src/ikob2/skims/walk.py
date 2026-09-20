"""
Walking times from zone geometry, without routing.

A rough starting point: crow-fly distance between centroids times a
detour factor, at a walking speed; within a zone, the mean distance
between two random points of the zone, from its area. It ignores
barriers (water, rail), so it is optimistic across them.
"""

from __future__ import annotations

import numpy as np

from ikob2.params import DEFAULTS

# Mean distance between two uniform random points: 0.5214 * sqrt(A) for
# a square, 0.5109 * sqrt(A) for a disc.
MEAN_INTRAZONAL_FACTOR = DEFAULTS.skims.intrazonal_factor


def intrazonal_distance_m(area_m2, factor: float | None = None) -> np.ndarray:
    f = MEAN_INTRAZONAL_FACTOR if factor is None else factor
    return f * np.sqrt(np.asarray(area_m2, dtype=float))


def walk_time_matrix(
    origin_xy: np.ndarray,
    dest_xy: np.ndarray,
    *,
    origin_codes=None,
    dest_codes=None,
    origin_area_m2=None,
    speed_kmh: float = DEFAULTS.skims.walk_kmh,
    detour: float = DEFAULTS.skims.walk_detour,
    max_minutes: float | None = DEFAULTS.skims.walk_max_minutes,
    intrazonal_factor: float | None = None,
) -> np.ndarray:
    """Walking minutes (n_origins x n_dest, float32); NaN beyond
    `max_minutes`. Coordinates are projected metres (RD New).

    Where an origin and a destination are the same zone (equal codes) the
    time is the intrazonal walk from `origin_area_m2` (required then).
    """
    o = np.asarray(origin_xy, dtype=float)
    d = np.asarray(dest_xy, dtype=float)
    if speed_kmh <= 0 or detour <= 0:
        raise ValueError("speed and detour must be positive.")
    dist = np.hypot(o[:, None, 0] - d[None, :, 0],
                    o[:, None, 1] - d[None, :, 1]) * detour
    if origin_codes is not None and dest_codes is not None:
        same = (np.asarray(origin_codes, dtype=str)[:, None]
                == np.asarray(dest_codes, dtype=str)[None, :])
        if same.any():
            if origin_area_m2 is None:
                raise ValueError("origin_area_m2 is needed for intrazonal "
                                 "walking times.")
            intra = intrazonal_distance_m(origin_area_m2, intrazonal_factor)
            dist = np.where(same, intra[:, None] * detour, dist)
    minutes = dist / (speed_kmh * 1000.0 / 60.0)
    if max_minutes is not None:
        minutes = np.where(minutes <= max_minutes, minutes, np.nan)
    return minutes.astype(np.float32)
