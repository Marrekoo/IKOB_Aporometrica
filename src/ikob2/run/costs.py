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
