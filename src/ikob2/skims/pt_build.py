"""Fill a skim store's public transport layer from a PtRouter."""

from __future__ import annotations

import logging

import numpy as np

from ikob2.skims.gtfs_pt import PtRouter
from ikob2.skims.store import SkimStore

logger = logging.getLogger(__name__)


def build_pt_layer(store: SkimStore, router: PtRouter, origin_xy: np.ndarray,
                   dest_codes, dest_xy: np.ndarray, *, layer: str = "all",
                   mode: str = "pt", variable: str = "time",
                   max_minutes: float = 180.0, block_size: int = 10) -> None:
    """Door-to-door PT minutes for every store origin to every destination
    zone, in an extra layer `layer` (created if missing), resumable per
    origin block. origin_xy / dest_xy are RD New metres in the order of
    store.origins / dest_codes."""
    dest_codes = [str(c) for c in dest_codes]
    if len(origin_xy) != len(store.origins):
        raise ValueError("origin_xy does not match the store's origins.")
    if layer not in store.layer_names:
        store.add_layer(layer, dest_codes)
    elif list(store.layer(layer).destinations) != dest_codes:
        raise ValueError(f"Layer '{layer}' has different destinations.")
    if len(dest_xy) != len(dest_codes):
        raise ValueError("dest_xy does not match dest_codes.")
    store.allocate(layer, mode, variable)
    for start, stop in store.pending_blocks(layer, mode, variable, block_size):
        t = router.time_matrix(origin_xy[start:stop], dest_xy,
                               max_minutes=max_minutes)
        store.write_rows(layer, mode, variable, start, t)
        logger.info("pt rows %d-%d done", start, stop)
