"""
Car route distances for a skim store.

Within `radius_km` of an origin (crow-fly) the distance comes from the
local Valhalla server; beyond it, and for the far (municipality) layer,
from the crow-fly distance times the calibrated detour factor: long
matrices are memory-hungry on the server and the detour is stable (1.2 to
1.3) there. Where Valhalla finds no route the detour estimate is used.
"""

from __future__ import annotations

import logging
from typing import Callable

import numpy as np
import pandas as pd

from ikob2.params import DEFAULTS
from ikob2.skims.car import DetourModel, crowfly_km
from ikob2.skims.store import SkimStore

logger = logging.getLogger(__name__)

MatrixFn = Callable[[np.ndarray, np.ndarray], tuple]


def crowfly_distance_block(origins: pd.DataFrame, dests: pd.DataFrame,
                           detour: DetourModel) -> np.ndarray:
    """Detour-model route distance (km) for origins x destinations."""
    return detour.route_km(crowfly_km(origins[["x", "y"]].to_numpy(),
                                      dests[["x", "y"]].to_numpy()))


def build_car_distance(
    store: SkimStore,
    origins: pd.DataFrame,
    layers: dict[str, pd.DataFrame],
    detour: DetourModel,
    matrix_fn: MatrixFn,
    *,
    radius_km: float = DEFAULTS.distance.radius_km,
    origin_batch: int = DEFAULTS.distance.origin_batch,
    dest_batch: int = DEFAULTS.distance.dest_batch,
    variable: str = "distance",
    mode: str = "car",
) -> dict:
    """Fill `mode/variable` in every layer. origins / layers[name] are
    frames with id, lon, lat, x, y in the store's order. matrix_fn(o, d)
    with (n, 2) lon/lat arrays returns (km, minutes) matrices. Resumable
    per origin block like build_time_skims. Returns simple statistics."""
    if list(origins["id"]) != store.origins:
        raise ValueError("origins do not match the store's origin codes.")
    stats = {"routed_pairs": 0, "fallback_pairs": 0}
    for name, dests in layers.items():
        if list(dests["id"]) != list(store.layer(name).destinations):
            raise ValueError(f"layer '{name}' does not match the store.")
        store.allocate(name, mode, variable)
        for start, stop in store.pending_blocks(name, mode, variable,
                                                origin_batch):
            o = origins.iloc[start:stop]
            block = crowfly_distance_block(o, dests, detour)
            if name == "near" and radius_km > 0:
                crow = crowfly_km(o[["x", "y"]].to_numpy(),
                                  dests[["x", "y"]].to_numpy())
                inside = (crow <= radius_km)
                cols = np.flatnonzero(inside.any(axis=0))
                for s in range(0, len(cols), dest_batch):
                    sel = cols[s:s + dest_batch]
                    km, _ = matrix_fn(o[["lon", "lat"]].to_numpy(),
                                      dests.iloc[sel][["lon", "lat"]]
                                      .to_numpy())
                    ok = np.isfinite(km) & inside[:, sel]
                    sub = block[:, sel]
                    sub[ok] = km[ok]
                    block[:, sel] = sub
                    stats["routed_pairs"] += int(ok.sum())
                    stats["fallback_pairs"] += int(
                        (inside[:, sel] & ~ok).sum())
            store.write_rows(name, mode, variable, start,
                             block.astype(np.float32))
            logger.info("%s/%s/%s rows %d-%d done", name, mode, variable,
                        start, stop)
    return stats
