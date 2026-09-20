"""
Shared-bicycle chains as alternative journeys.

Skim modes in the store (built by `cli.skims build-pt --mode-name ...`):

    pt     walk access, walk egress            (plain public transport)
    pt_wb  walk access, bicycle egress at rail hubs (OV-fiets)
    pt_bw  bicycle access, walk egress
    pt_bb  bicycle access, bicycle egress

Prices added to the public transport fare of the same journey:

    OV-fiets egress   a flat charge per rental (default EUR 4.80 per 24 h)
    dockless access   unlock fee + rate per riding minute (EUR 1.00 + 0.20)
    own bicycle       free

Variants (`shared_bike_modes`):

    v0  own bicycle only: residents with a private bicycle may ride to the
        stop; walk egress. No shared bicycle. The baseline for v1 and v2.
    v1  egress only: everyone may take an OV-fiets at the destination rail
        station instead of walking. Options: plain, walk + OV-fiets.
    v2  access and egress by ownership. Residents with a private bicycle:
        own bicycle for access (free), OV-fiets for egress. Residents
        without: dockless for access, OV-fiets for egress. Options: plain,
        bicycle access, bicycle access + OV-fiets, walk + OV-fiets. The
        share with a bicycle is the buurt's private bicycle share.

    v3  the options of v2 judged leg by leg (paper eq. 3.2): the bicycle legs
        have their own time margin (bicycle) and the public transport leg
        the PT margin on its own time (total minus the bicycle legs); the
        cost margin is on the journey total. Independent thresholds only.
    v4  (`dockless_mode`) shared bicycle as a stand-alone mode: residents
        with a private bicycle ride it free, the others take a dockless
        bicycle door to door, with unlock fee and a rate per minute.

Every variant is a person's set of alternatives: a pair is acceptable if any
option passes both gates (`run.accessibility.OptionSet`).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ikob2.run.accessibility import (LegOption, LegOptionSet, MixedMode,
                                     ModeMatrices, OptionSet)

VARIANTS = ("v0", "v1", "v2", "v3")


@dataclass(frozen=True)
class SharedBikeTariffs:
    ovfiets_eur: float = 4.80            # per rental period (egress)
    dockless_unlock_eur: float = 1.00
    dockless_per_min_eur: float = 0.20

    def __post_init__(self):
        if min(self.ovfiets_eur, self.dockless_unlock_eur,
               self.dockless_per_min_eur) < 0:
            raise ValueError("Tariffs must be >= 0.")


def _option(store_mode: dict, fare: np.ndarray, *, extra=0.0,
            cost_id="pt_chain") -> ModeMatrices:
    return ModeMatrices(store_mode["time"],
                        np.asarray(fare, dtype=np.float32)
                        + np.asarray(extra, dtype=np.float32), cost_id)


def shared_bike_modes(chains: dict, fares: dict, bike_share: np.ndarray,
                      tariffs: SharedBikeTariffs = SharedBikeTariffs(),
                      variants=("v0", "v1", "v2"),
                      bike_fixed_min: float = 1.0) -> dict:
    """{'pt_v0': ..., 'pt_v1': ..., 'pt_v2': ...} for the requested variants.

    chains : mode -> {'time': matrix, 'access_min': matrix (bicycle
        access modes)} for 'pt', 'pt_wb', 'pt_bw', 'pt_bb'.
    fares  : mode -> public transport fare matrix of that journey.
    bike_share : share of residents with a private bicycle per origin.
    """
    unknown = set(variants) - set(VARIANTS)
    if unknown:
        raise ValueError(f"Unknown variant(s) {sorted(unknown)}; "
                         f"use {VARIANTS}.")
    p = np.asarray(bike_share, dtype=float)
    if p.ndim != 1 or (p < 0).any() or (p > 1).any():
        raise ValueError("bike_share: one share in [0, 1] per origin.")
    ov = tariffs.ovfiets_eur
    dock = (tariffs.dockless_unlock_eur
            + tariffs.dockless_per_min_eur * chains["pt_bw"]["access_min"]) \
        if "pt_bw" in chains else None
    dock_bb = (tariffs.dockless_unlock_eur
               + tariffs.dockless_per_min_eur * chains["pt_bb"]["access_min"]) \
        if "pt_bb" in chains else None

    def opt(mode, extra=0.0):
        return _option(chains[mode], fares[mode], extra=extra)

    out = {}
    plain = opt("pt")
    if "v0" in variants:
        owners = OptionSet((plain, opt("pt_bw")), "pt")
        out["pt_v0"] = MixedMode(((p, owners), (1 - p, OptionSet((plain,), "pt"))))
    if "v1" in variants:
        out["pt_v1"] = OptionSet((plain, opt("pt_wb", ov)), "pt")
    if "v2" in variants:
        owners = OptionSet((plain, opt("pt_bw"), opt("pt_bb", ov),
                            opt("pt_wb", ov)), "pt")
        others = OptionSet((plain, opt("pt_bw", dock), opt("pt_bb", dock_bb + ov),
                            opt("pt_wb", ov)), "pt")
        out["pt_v2"] = MixedMode(((p, owners), (1 - p, others)))
    if "v3" in variants:
        for need in ("access_min", "egress_min"):
            if need not in chains.get("pt_bb", {}):
                raise ValueError(f"v3 needs '{need}' in the pt_bb chain.")
        out["pt_v3"] = _legwise(chains, fares, p, tariffs, dock, dock_bb,
                                bike_fixed_min)
    return out


def _legwise(chains, fares, p, tariffs, dock, dock_bb, fixed):
    """v2's options with the times split into (bicycle access, bicycle
    egress, public transport) legs."""
    zero = np.zeros_like(chains["pt"]["time"])

    def leg(mode, extra=0.0, access=False, egress=False):
        t = chains[mode]["time"]
        a = (chains[mode]["access_min"] + fixed) if access else zero
        b = (chains[mode]["egress_min"] + fixed) if egress else zero
        pt = np.maximum(t - np.nan_to_num(a) - np.nan_to_num(b), 0.0)
        pt = np.where(np.isfinite(t), pt, np.nan)
        cost = np.asarray(fares[mode], dtype=np.float32) \
            + np.asarray(extra, dtype=np.float32)
        return LegOption((a, b, pt), cost, "pt_chain")

    owners = LegOptionSet((leg("pt"), leg("pt_bw", access=True),
                           leg("pt_bb", tariffs.ovfiets_eur, True, True),
                           leg("pt_wb", tariffs.ovfiets_eur, False, True)))
    others = LegOptionSet((leg("pt"), leg("pt_bw", dock, access=True),
                           leg("pt_bb", dock_bb + tariffs.ovfiets_eur, True,
                               True),
                           leg("pt_wb", tariffs.ovfiets_eur, False, True)))
    return MixedMode(((p, owners), (1 - p, others)))


def dockless_mode(bike_time: np.ndarray, bike_share: np.ndarray,
                  tariffs: SharedBikeTariffs = SharedBikeTariffs(),
                  fixed_min: float = 1.0) -> MixedMode:
    """v4: bicycle accessibility of the whole population when residents
    without a private bicycle can rent a dockless one door to door (unlock
    fee plus rate per minute of riding, `fixed_min` to unlock)."""
    p = np.asarray(bike_share, dtype=float)
    if p.ndim != 1 or (p < 0).any() or (p > 1).any():
        raise ValueError("bike_share: one share in [0, 1] per origin.")
    t = np.asarray(bike_time, dtype=np.float32)
    own = OptionSet((ModeMatrices(t),), "bike")
    rent = OptionSet((ModeMatrices(
        t + fixed_min,
        tariffs.dockless_unlock_eur + tariffs.dockless_per_min_eur * t,
        "dockless"),), "bike")
    return MixedMode(((p, own), (1 - p, rent)))
