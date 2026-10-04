"""Person-based against location-based price cuts: who is eligible, who pays,
who gains.

A Lime price cut can be granted by income (a concession for some income
classes, citywide) or by home address (the residents of some buurten, S4),
or by both. For each scenario run, against a baseline run, this module
reports

  * eligibility: the share of the target group (by default deciles D2-D4)
    with a lower price at their address (coverage; one minus the exclusion
    error) and the share of the eligible residents outside the target group
    (the inclusion error), both in persons (Cornia and Stewart, 1993);
  * cost: the public cost per year: the revenue foregone at baseline volume
    of a price cut (`scenario.cost` of run.json) and the yearly cost of extra
    hubs (`hub_cost_eur_year`, set by the caller), and the shares of the
    price cost spent on the target group and in the zone (not defined when
    hubs are part of the cost);
  * gain: acceptable job-persons, per euro, and the shares going to the
    target group and to the zone's residents;
  * cells: population and gain by income class and place (zone or rest),
    for a figure.

The zone is a list of buurten (the S4 price zone); the scales and zones of a
run come from its run.json (`lime_price_scales`, `lime_price_zones`).
"""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np
import pandas as pd

KEY = ["buurtcode", "segment"]


def _cells(base: pd.DataFrame, mode: str, zone: set) -> pd.DataFrame:
    d = base[base["mode"] == mode][KEY + ["income_class", "population",
                                          "accessibility"]].copy()
    d["place"] = np.where(d["buurtcode"].astype(str).isin(zone), "zone", "rest")
    return d.set_index(KEY)


def targeting(base: pd.DataFrame, runs: Mapping[str, tuple[pd.DataFrame, dict]],
              zone: Sequence[str], mode: str = "pt_v2",
              target_classes: Sequence[str] = ("D2", "D3", "D4")
              ) -> dict[str, pd.DataFrame]:
    """summary and cells for the scenario runs `runs` (label -> (the run's
    accessibility table, its run.json)) against `base`."""
    zone = {str(z) for z in zone}
    target = set(target_classes)
    d = _cells(base, mode, zone)
    in_target = d["income_class"].isin(target)
    pop = d["population"]
    summary, cells = [], []
    for label, (run, meta) in runs.items():
        r = run[run["mode"] == mode].set_index(KEY)["accessibility"].reindex(d.index)
        gain = (r - d["accessibility"]) * pop
        scales = meta.get("lime_price_scales") or {}
        zones = meta.get("lime_price_zones")
        at_zone = (d.index.get_level_values("buurtcode").astype(str).isin(set(zones))
                   if zones is not None else np.ones(len(d), dtype=bool))
        cut = np.array([float(scales.get(s, 1.0)) < 1.0
                        for s in d.index.get_level_values("segment")])
        eligible = cut & at_zone
        cost = (meta.get("scenario") or {}).get("cost") or {}
        price = cost.get("compensation_eur_year", 0.0)
        hubs = meta.get("hub_cost_eur_year", 0.0)
        total = price + hubs if (cost or hubs) else np.nan
        by_seg = cost.get("by_segment_eur_year") or {}
        by_org = cost.get("by_origin_eur_year") or {}
        c_target = sum(v for k, v in by_seg.items() if k.rsplit("_", 1)[1] in target)
        c_zone = sum(v for k, v in by_org.items() if str(k) in zone)
        g = gain.sum()
        # the cost of hubs cannot be attributed to segments or places
        attributable = not hubs
        summary.append({
            "scenario": label, "hubs": meta.get("hubs", 0),
            "price_cost_eur_year": price, "hub_cost_eur_year": hubs,
            "target_eligible_share": pop[eligible & in_target].sum() / pop[in_target].sum(),
            "eligible_outside_target_share": (pop[eligible & ~in_target].sum()
                                              / pop[eligible].sum()
                                              if pop[eligible].sum() > 0 else np.nan),
            "eligible_persons": pop[eligible].sum(),
            "cost_eur_year": total,
            "cost_share_target": c_target / total if total and attributable else np.nan,
            "cost_share_zone": c_zone / total if total and attributable else np.nan,
            "gain_job_persons": g,
            "gain_per_eur": g / total if total else np.nan,
            "gain_share_target": gain[in_target].sum() / g if g else np.nan,
            "gain_share_zone": gain[d["place"] == "zone"].sum() / g if g else np.nan})
        c = pd.DataFrame({"population": pop, "gain": gain,
                          "income_class": d["income_class"],
                          "place": d["place"]}).groupby(
            ["income_class", "place"])[["population", "gain"]].sum()
        c["gain_per_person"] = c["gain"] / c["population"].where(c["population"] > 0)
        cells.append(c.reset_index().assign(scenario=label))
    return {"summary": pd.DataFrame(summary),
            "cells": pd.concat(cells, ignore_index=True)[
                ["scenario", "income_class", "place", "population", "gain",
                 "gain_per_person"]]}
