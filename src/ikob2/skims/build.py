"""Fill a skim store from a router, block by block (resumable)."""

from __future__ import annotations

import logging
from typing import Mapping

import numpy as np
import pandas as pd

from ikob2.skims.router import Router, TimeRequest
from ikob2.skims.store import SkimStore

logger = logging.getLogger(__name__)


def build_time_skims(
    router: Router,
    store: SkimStore,
    origins: pd.DataFrame,
    layers: Mapping[str, pd.DataFrame],
    requests: Mapping[str, TimeRequest],
    *,
    block_size: int = 25,
    variable: str = "time",
) -> None:
    """Compute travel-time matrices for every (layer, mode) of `requests`.

    origins / layers[name] are frames with id, lon, lat, in the order of
    the store's origin and layer-destination codes. Origin blocks
    already recorded in the store are skipped, so a rerun continues an
    interrupted build.
    """
    if list(origins["id"]) != store.origins:
        raise ValueError("origins do not match the store's origin codes.")
    for name, dests in layers.items():
        if list(dests["id"]) != list(store.layer(name).destinations):
            raise ValueError(f"layer '{name}' destinations do not match "
                             f"the store.")
        for mode, request in requests.items():
            if request.mode != mode:
                raise ValueError(f"request for '{mode}' has mode "
                                 f"'{request.mode}'.")
            store.allocate(name, mode, variable)
            blocks = store.pending_blocks(name, mode, variable, block_size)
            logger.info("%s/%s: %d block(s) to compute", name, mode,
                        len(blocks))
            for start, stop in blocks:
                sub = origins.iloc[start:stop]
                values = router.time_matrix(sub, dests, request)
                store.write_rows(name, mode, variable, start,
                                 np.asarray(values, dtype=np.float32))
