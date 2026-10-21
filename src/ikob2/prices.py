"""Price level of the money in a run.

Every amount a money gate compares must be at one price level: the
reference budgets and the values of time are in euros of `prices.level`
(2022, the price level of the KWB population year). Each tariff source
states the date of its prices (`pt_fare.price_date`, `shared_bike.price_date`,
`car.price_date`, `costs.price_date`), and its amounts are converted with the
consumer price index, all items (`prices.cpi`, CBS 86141NED):

    amount at the run's level = amount x CPI(level) / CPI(price date)

so a fare keeps its real value at its own date, expressed in the euros of
the budgets.
"""

from __future__ import annotations


def factor(prm, price_date: str) -> float:
    """CPI(prices.level) / CPI(price_date): the multiplier that converts an
    amount at `price_date` to the run's price level."""
    cpi = prm.prices.cpi.to_dict()
    level = str(prm.prices.level)
    missing = [d for d in (level, str(price_date)) if d not in cpi]
    if missing:
        raise KeyError(f"prices.cpi has no index for {missing}; known: "
                       f"{sorted(cpi)}.")
    values = (float(cpi[level]), float(cpi[str(price_date)]))
    if min(values) <= 0:
        raise ValueError("Price indices must be positive.")
    return values[0] / values[1]
