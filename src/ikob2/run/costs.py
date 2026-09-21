"""Public cost of the shared-bicycle measures, for the comparison of S1 and S2
at equal annual expenditure.

Division of cost and risk (assumptions, `costs` parameters):

  * S2, extra hubs: the set-up (signage, marking, bicycle parking) spread over
    the permit term, plus the yearly rebalancing fee and enforcement surcharge.
    The total PUBLIC cost counts the set-up in full (municipality and
    province); the municipal net cost is reported next to it. The social tariff
    compensation of the cluster model is a price concession and belongs to S1.
  * S1, lower prices: the municipality compensates the operator for the
    revenue foregone at BASELINE volume (the operator is neutral, the public
    carries the price risk). Volume responses to the lower price accrue to the
    operator.
  * Volume: the model counts acceptable pairs, not trips. Its rentals are
    scaled to the observed annual rides of the concession area, and the model's
    mean price per rental is assumed to hold across that area.
"""

from __future__ import annotations

from typing import Mapping

import pandas as pd  # noqa: F401  (type of `spend`)


def annuity_factor(years: float, rate: float) -> float:
    """Years of cost an up-front amount is spread over: `years` without
    discounting, else the present value of one euro a year for `years`."""
    if years <= 0:
        raise ValueError("horizon_years must be positive.")
    if rate == 0:
        return years
    return (1.0 - (1.0 + rate) ** -years) / rate


def hub_annual_cost(costs, n_hubs: float = 1.0) -> dict:
    """Yearly cost of `n_hubs` extra hubs (EUR): capex spread over the horizon
    (public = gross, municipal = net of the provincial share), rebalancing and
    enforcement."""
    per = 1.0 / costs.hub_cluster_size
    factor = annuity_factor(costs.horizon_years, costs.discount_rate)
    capex = costs.hub_setup_eur_cluster * per * n_hubs
    running = (costs.hub_rebalancing_eur_year_cluster
               + costs.hub_enforcement_eur_year_cluster) * per * n_hubs
    return {"capex_per_year_public": capex / factor,
            "capex_per_year_municipal": capex * (1 - costs.province_capex_share)
            / factor,
            "running_per_year": running,
            "public": capex / factor + running,
            "municipal": capex * (1 - costs.province_capex_share) / factor
            + running}


def annual_rentals(costs) -> float:
    """Rentals per year of the concession area, from the observed months."""
    return costs.rides_observed * 12.0 / costs.rides_months


def rides_per_bike_per_day(costs) -> float:
    return annual_rentals(costs) / costs.fleet / 365.0


def s1_compensation(costs, revenue_base: float, revenue_new_at_base_volume: float,
                    rentals_base: float) -> dict:
    """Yearly compensation of the operator for lower prices, at baseline volume.

    revenue_* and rentals_base are in model units (EUR x acceptable pairs and
    pairs); the ratio scales them to the observed rides."""
    if rentals_base <= 0:
        raise ValueError("No baseline rentals.")
    per_rental = (revenue_base - revenue_new_at_base_volume) / rentals_base
    mean_price = revenue_base / rentals_base
    rides = annual_rentals(costs)
    return {"annual_rentals": rides, "mean_price_eur": mean_price,
            "compensation_per_rental_eur": per_rental,
            "annual_revenue_eur": rides * mean_price,
            "compensation_eur_year": rides * per_rental,
            "share_of_revenue": per_rental / mean_price}


def hubs_for_budget(costs, budget_eur_year: float, view: str = "public") -> float:
    """How many extra hubs a yearly budget pays for."""
    return budget_eur_year / hub_annual_cost(costs, 1.0)[view]


def pt_fare_cost(spend: "pd.DataFrame", persons: Mapping[str, float],
                 fare_scale: Mapping[str, float]) -> dict:
    """Public cost per year of a public transport fare concession, at baseline
    volume: the fare revenue foregone,

        sum over segments of persons_s x spend[decile_s] x (1 - scale_s),

    with `spend` the ODiN table of `segments.pt_spend` (fare spending per person
    and year by income decile) and `persons` the persons per segment
    ('<household type>_<decile>'). The shrunk local estimate is the central
    figure; the national-only estimate (trips and fares of all of the
    Netherlands) is the low estimate."""
    tab = spend.set_index("income_class")
    central = low = 0.0
    by_decile: dict[str, float] = {}
    for seg, n in persons.items():
        dec = seg.rsplit("_", 1)[1]
        if dec not in tab.index:
            continue
        cut = 1.0 - float(fare_scale.get(seg, 1.0))
        if cut == 0.0:
            continue
        c = n * float(tab.loc[dec, "spend_eur_year"]) * cut
        nat = n * float(tab.loc[dec, "trips_national"] * tab.loc[dec, "fare_national"]) * cut
        central += c
        low += nat
        by_decile[dec] = by_decile.get(dec, 0.0) + c
    return {"eur_year": central, "eur_year_national": low, "by_decile": by_decile}
