"""
Shared-bicycle chains as alternative journeys.

Skim modes in the store (built by `cli.skims build-pt --mode-name ...`):

    pt            walk access, walk egress (plain public transport)
    pt_bw         bicycle access, walk egress
    pt_wb_<kind>  walk access, bicycle egress from a hub of tariff <kind>
    pt_bb_<kind>  bicycle access, bicycle egress from a hub of <kind>

(`pt_wb`, `pt_bb` without a kind are egress at rail stops, priced as
OV-fiets.)

Prices added to the public transport fare of the same journey:

    OV-fiets egress   a flat charge per rental (default EUR 4.80 per 24 h)
    dockless access   Lime tiers by rental duration (EUR 3 / 4 / 5 up to
                      20 / 30 / 40 minutes), or unlock fee + rate per riding
                      minute (EUR 1.00 + 0.20): `dockless_model`
    own bicycle       free

Variants (`shared_bike_modes`):

    v0  own bicycle only: residents with a private bicycle may ride to the
        stop; walk egress. No shared bicycle. The baseline for v1 and v2.
    v1  egress only: everyone may take a shared bicycle from a hub near the
        alighting stop instead of walking. Options: plain, walk + shared
        bicycle (one per hub kind).
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
        bicycle door to door at the dockless price (`dockless_model`).

Every variant is a person's set of alternatives: a pair is acceptable if any
option passes both gates (`run.accessibility.OptionSet`).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ikob2.params import DEFAULTS
from ikob2.run.accessibility import (LegOption, LegOptionSet, MixedMode,
                                     ModeMatrices, OptionSet)

VARIANTS = ("v0", "v1", "v2", "v3")   # PT chains; v4 is `dockless_mode`
DOCKLESS_MODELS = ("lime_tiers", "unlock_per_minute", "flat")


@dataclass(frozen=True)
class SharedBikeTariffs:
    ovfiets_eur: float = DEFAULTS.shared_bike.ovfiets_eur   # per rental (egress)
    dockless_unlock_eur: float = DEFAULTS.shared_bike.dockless_unlock_eur
    dockless_per_min_eur: float = DEFAULTS.shared_bike.dockless_per_min_eur
    lime_tiers: tuple = tuple(tuple(t) for t in DEFAULTS.shared_bike.lime_tiers)
    hub_tariffs: tuple = tuple(DEFAULTS.shared_bike.hub_tariffs)
    dockless_model: str = DEFAULTS.shared_bike.dockless_model
    lime_scale: float = DEFAULTS.shared_bike.lime_scale
    flat_eur: float = DEFAULTS.shared_bike.flat_eur

    def __post_init__(self):
        if min(self.ovfiets_eur, self.dockless_unlock_eur,
               self.dockless_per_min_eur, self.flat_eur) < 0:
            raise ValueError("Tariffs must be >= 0.")
        if not self.lime_scale >= 0:
            raise ValueError("lime_scale must be >= 0.")
        if self.dockless_model not in DOCKLESS_MODELS:
            raise ValueError(f"dockless_model must be one of "
                             f"{DOCKLESS_MODELS}, got {self.dockless_model!r}.")
        bounds = [b for b, _ in self.lime_tiers]
        if not self.lime_tiers or bounds != sorted(bounds):
            raise ValueError("lime_tiers: [longest minutes, EUR] rows, "
                             "increasing in minutes.")

    def lime_eur(self, minutes) -> np.ndarray:
        """Lime price of a rental of `minutes`: the tier of its duration
        (the last tier's price beyond the last bound), or the flat price per
        rental in the `flat` model (S4); times `lime_scale`. NaN stays NaN."""
        m = np.asarray(minutes, dtype=float)
        if self.dockless_model == "flat":
            price = np.full(m.shape, self.flat_eur * self.lime_scale)
        else:
            bounds = np.array([b for b, _ in self.lime_tiers])
            tiers = np.array([p for _, p in self.lime_tiers]) * self.lime_scale
            price = tiers[np.minimum(np.searchsorted(bounds, m, side="left"),
                                     len(tiers) - 1)]
        return np.where(np.isfinite(m), price, np.nan)

    def dockless_eur(self, ride_min, fixed_min: float) -> np.ndarray:
        """Price of a dockless rental of `ride_min` riding minutes:
        Lime tiers by rental duration (ride plus fixed minutes), a flat price
        per rental, or an unlock fee plus a rate per riding minute."""
        ride = np.asarray(ride_min, dtype=float)
        if self.dockless_model in ("lime_tiers", "flat"):
            return self.lime_eur(ride + fixed_min)
        return self.dockless_unlock_eur + self.dockless_per_min_eur * ride

    def egress_eur(self, ride_min, kind: str, fixed_min: float) -> np.ndarray:
        """Price of a bicycle egress of `ride_min` minutes from a hub of
        tariff `kind`: OV-fiets a flat charge, Lime by rental duration
        (ride plus fixed minutes)."""
        if kind not in self.hub_tariffs:
            raise ValueError(f"Unknown hub tariff {kind!r}; use "
                             f"{self.hub_tariffs}.")
        ride = np.asarray(ride_min, dtype=float)
        price = (self.lime_eur(ride + fixed_min) if kind == "lime"
                 else np.full(ride.shape, self.ovfiets_eur))
        return np.where(np.isfinite(ride), price, np.nan).astype(np.float32)


DEFAULT_TARIFFS = SharedBikeTariffs()   # frozen: safe to share as a default


# ── price scales by segment (concessions: U-pas, low-income schemes) ─

def load_price_scales(path, household_types=None, income_classes=None
                      ) -> pd.DataFrame:
    """Table of multipliers on the Lime price by segment. Columns
    `household_type`, `income_class` (each a name or `*` for all) and `scale`
    (>= 0; 0.5 halves the price, 0 is free). Rows are applied in order and a
    later row overrides an earlier one, so a general row can be followed by
    exceptions."""
    from ikob2.segments.config import HOUSEHOLD_TYPES, INCOME_CLASSES

    ht = tuple(household_types or HOUSEHOLD_TYPES)
    ic = tuple(income_classes or INCOME_CLASSES)
    df = pd.read_csv(path, dtype={"household_type": str, "income_class": str})
    need = ["household_type", "income_class", "scale"]
    if [c for c in need if c not in df.columns]:
        raise ValueError(f"{path}: needs columns {need}, has "
                         f"{list(df.columns)}.")
    df["scale"] = pd.to_numeric(df["scale"], errors="coerce")
    if df["scale"].isna().any() or (df["scale"] < 0).any():
        raise ValueError(f"{path}: scale must be a number >= 0.")
    for col, known in (("household_type", ht), ("income_class", ic)):
        bad = sorted(set(df[col].str.strip()) - {"*", *known})
        if bad:
            raise ValueError(f"{path}: unknown {col} {bad}; use * or "
                             f"{list(known)}.")
    return df[need].assign(**{c: df[c].str.strip() for c in need[:2]})


def segment_price_scales(names, table: pd.DataFrame | None) -> dict:
    """segment name (household_class, e.g. 'single_D3') -> price multiplier
    (default 1) from a price-scale table."""
    out = {n: 1.0 for n in names}
    if table is None:
        return out
    for row in table.itertuples(index=False):
        for n in names:
            ht, ic = n.rsplit("_", 1)
            if row.household_type in ("*", ht) and row.income_class in ("*", ic):
                out[n] = float(row.scale)
    return out


def _option(store_mode: dict, fare: np.ndarray, *, ov=0.0, lime=None,
            rentals: float = 0.0, cost_id="pt_chain") -> ModeMatrices:
    """One journey: the PT fare, an OV-fiets charge `ov` and the Lime part
    `lime` (a matrix, `rentals` Lime rentals) whose price can be scaled per
    segment."""
    lime_part = None if lime is None else np.asarray(lime, dtype=np.float32)
    cost = (np.asarray(fare, dtype=np.float32) + np.float32(ov)
            + (0.0 if lime_part is None else lime_part))
    return ModeMatrices(store_mode["time"], cost.astype(np.float32), cost_id,
                        rail_share=store_mode.get("rail_share"),
                        lime=lime_part, lime_rentals=rentals,
                        fare=np.asarray(fare, dtype=np.float32))


def shared_bike_modes(chains: dict, fares: dict, bike_share: np.ndarray,
                      tariffs: SharedBikeTariffs = DEFAULT_TARIFFS,
                      variants=("v0", "v1", "v2"),
                      bike_fixed_min: float = DEFAULTS.bike_leg.fixed_minutes
                      ) -> dict:
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

    # Egress chains, one per hub kind (skim modes pt_wb_<kind>, pt_bb_<kind>):
    # the tariffs differ, so each kind is its own alternative journey and a
    # person accepts the pair if ANY of them clears both gates (a slower hub
    # may be the one that is affordable). Without kinds: pt_wb / pt_bb at the
    # OV-fiets flat charge.
    kinds = [k for k in tariffs.hub_tariffs
             if f"pt_wb_{k}" in chains or f"pt_bb_{k}" in chains]
    egress = [(f"pt_wb_{k}", f"pt_bb_{k}", k) for k in kinds] \
        or [("pt_wb", "pt_bb", None)]
    # the bicycle access leg is the same in every egress kind
    bb0 = next((bb for _, bb, _k in egress if bb in chains), None)
    dock_bb = None if bb0 is None else tariffs.dockless_eur(
        chains[bb0]["access_min"], bike_fixed_min)

    def egress_part(mode, kind):
        """(OV-fiets charge, Lime matrix, Lime rentals) of a bicycle egress."""
        if kind is None or kind == "ovfiets":
            return tariffs.ovfiets_eur, None, 0
        return 0.0, tariffs.egress_eur(chains[mode]["egress_min"], kind,
                                       bike_fixed_min), 1

    dock = tariffs.dockless_eur(chains["pt_bw"]["access_min"],
                                bike_fixed_min) if "pt_bw" in chains else None

    def opt(mode, ov=0.0, lime=None, rentals=0):
        return _option(chains[mode], fares[mode], ov=ov, lime=lime,
                       rentals=rentals)

    def egress_opt(mode, kind, dockless=None):
        """Bicycle egress at a hub of `kind`, with dockless access (Lime
        matrix `dockless`) or without."""
        ov, lime, n = egress_part(mode, kind)
        if dockless is not None:
            lime = dockless if lime is None else dockless + lime
            n += 1
        return opt(mode, ov, lime, n)

    out = {}
    plain = opt("pt")
    if "v0" in variants:
        owners = OptionSet((plain, opt("pt_bw")), "pt")
        out["pt_v0"] = MixedMode(((p, owners), (1 - p, OptionSet((plain,), "pt"))))
    if "v1" in variants:
        out["pt_v1"] = OptionSet(
            (plain, *(egress_opt(wb, k) for wb, _, k in egress
                      if wb in chains)), "pt")
    if "v2" in variants:
        owners = OptionSet((plain, opt("pt_bw"),
                            *(egress_opt(bb, k) for _, bb, k in egress),
                            *(egress_opt(wb, k) for wb, _, k in egress)), "pt")
        others = OptionSet((plain, opt("pt_bw", lime=dock, rentals=1),
                            *(egress_opt(bb, k, dock_bb) for _, bb, k in egress),
                            *(egress_opt(wb, k) for wb, _, k in egress)), "pt")
        out["pt_v2"] = MixedMode(((p, owners), (1 - p, others)))
    if "v3" in variants:
        for _, bb, _k in egress:
            for need in ("access_min", "egress_min"):
                if need not in chains.get(bb, {}):
                    raise ValueError(f"v3 needs '{need}' in the {bb} chain.")
        out["pt_v3"] = _legwise(chains, fares, p, dock, dock_bb,
                                bike_fixed_min, egress, egress_part)
    return out


def _legwise(chains, fares, p, dock, dock_bb, fixed, egress, egress_part):
    """v2's options with the times split into (bicycle access, bicycle
    egress, public transport) legs."""
    zero = np.zeros_like(chains["pt"]["time"])

    def leg(mode, ov=0.0, lime=None, rentals=0, access=False, egress=False):
        t = chains[mode]["time"]
        a = (chains[mode]["access_min"] + fixed) if access else zero
        b = (chains[mode]["egress_min"] + fixed) if egress else zero
        pt = np.maximum(t - np.nan_to_num(a) - np.nan_to_num(b), 0.0)
        pt = np.where(np.isfinite(t), pt, np.nan)
        lime_part = None if lime is None else np.asarray(lime, np.float32)
        cost = (np.asarray(fares[mode], dtype=np.float32) + np.float32(ov)
                + (0.0 if lime_part is None else lime_part))
        return LegOption((a, b, pt), cost.astype(np.float32), "pt_chain",
                         lime=lime_part, lime_rentals=rentals,
                         rail_share=chains[mode].get("rail_share"),
                         fare=np.asarray(fares[mode], dtype=np.float32))

    def egress_leg(mode, kind, access=False, dockless=None, egress=True):
        ov, lime, n = egress_part(mode, kind)
        if dockless is not None:
            lime = dockless if lime is None else dockless + lime
            n += 1
        return leg(mode, ov, lime, n, access, egress)

    owners = LegOptionSet((
        leg("pt"), leg("pt_bw", access=True),
        *(egress_leg(bb, k, access=True) for _, bb, k in egress),
        *(egress_leg(wb, k) for wb, _, k in egress)))
    others = LegOptionSet((
        leg("pt"), leg("pt_bw", lime=dock, rentals=1, access=True),
        *(egress_leg(bb, k, access=True, dockless=dock_bb)
          for _, bb, k in egress),
        *(egress_leg(wb, k) for wb, _, k in egress)))
    return MixedMode(((p, owners), (1 - p, others)))


def dockless_mode(bike_time: np.ndarray, bike_share: np.ndarray,
                  tariffs: SharedBikeTariffs = DEFAULT_TARIFFS,
                  fixed_min: float = DEFAULTS.bike_leg.fixed_minutes
                  ) -> MixedMode:
    """v4: bicycle accessibility of the whole population when residents
    without a private bicycle can rent a dockless one door to door (priced by
    `tariffs.dockless_eur`, `fixed_min` to unlock)."""
    p = np.asarray(bike_share, dtype=float)
    if p.ndim != 1 or (p < 0).any() or (p > 1).any():
        raise ValueError("bike_share: one share in [0, 1] per origin.")
    t = np.asarray(bike_time, dtype=np.float32)
    own = OptionSet((ModeMatrices(t),), "bike")
    price = np.asarray(tariffs.dockless_eur(t, fixed_min), dtype=np.float32)
    rent = OptionSet((ModeMatrices(t + fixed_min, price, "dockless",
                                   lime=price, lime_rentals=1),), "bike")
    return MixedMode(((p, own), (1 - p, rent)))
