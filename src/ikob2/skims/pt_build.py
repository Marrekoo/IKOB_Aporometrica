"""Fill a skim store's public transport layer from a PtRouter."""

from __future__ import annotations

import logging

import numpy as np

from ikob2.params import DEFAULTS
from ikob2.skims.gtfs_pt import LegSpec, PtRouter
from ikob2.skims.store import SkimStore

logger = logging.getLogger(__name__)


FARE_VARIABLES = ("rail_km", "other_km", "other_boardings")
ACCESS_VARIABLE = "access_min"
EGRESS_VARIABLE = "egress_min"
EGRESS_KIND_VARIABLE = "egress_kind"    # tariff class of the hub used (hub files)


def build_pt_layer(store: SkimStore, router: PtRouter, origin_xy: np.ndarray,
                   dest_codes, dest_xy: np.ndarray, *, layer: str = "all",
                   mode: str = "pt", variable: str = "time",
                   max_minutes: float = DEFAULTS.pt.max_minutes,
                   block_size: int = DEFAULTS.pt.block_size,
                   fare_inputs: bool = True, access: LegSpec | None = None,
                   egress: LegSpec | None = None) -> None:
    """Door-to-door PT minutes for every store origin to every destination
    zone, in an extra layer `layer` (created if missing), resumable per
    origin block. origin_xy / dest_xy are RD New metres in the order of
    store.origins / dest_codes. With `fare_inputs` the variables rail_km,
    other_km and other_boardings (of the fastest journey) are stored too,
    so that fares can be computed later under any fare model. `access` /
    `egress` (LegSpec) replace walking by a bicycle leg for that end; with a
    bicycle access the ride minutes are stored as `access_min` (metered
    tariffs). Use a different `mode` name for each combination."""
    dest_codes = [str(c) for c in dest_codes]
    if len(origin_xy) != len(store.origins):
        raise ValueError("origin_xy does not match the store's origins.")
    if layer not in store.layer_names:
        store.add_layer(layer, dest_codes)
    elif list(store.layer(layer).destinations) != dest_codes:
        raise ValueError(f"Layer '{layer}' has different destinations.")
    if len(dest_xy) != len(dest_codes):
        raise ValueError("dest_xy does not match dest_codes.")
    names = [variable, *(FARE_VARIABLES if fare_inputs else ())]
    if fare_inputs and access is not None:
        names.append(ACCESS_VARIABLE)
    if fare_inputs and egress is not None:
        names.append(EGRESS_VARIABLE)
        if router.hubs_xy is not None and egress.hubs_only:
            names.append(EGRESS_KIND_VARIABLE)
    for name in names:
        store.allocate(layer, mode, name)
    pending = sorted({b for name in names
                      for b in store.pending_blocks(layer, mode, name,
                                                    block_size)})
    for start, stop in pending:
        res = router.journeys(origin_xy[start:stop], dest_xy,
                              max_minutes=max_minutes, track=fare_inputs,
                              access=access, egress=egress)
        store.write_rows(layer, mode, variable, start, res["time"])
        if fare_inputs:
            for name in names[1:]:
                store.write_rows(layer, mode, name, start, res[name])
        logger.info("pt rows %d-%d done", start, stop)
