# Reference-budget envelope: provenance

`data/envelope/reference_budgets.csv` (the paper's Table 6) comes from the R
script `X_M calc.R` (not in this repository): per household type and income
decile, `low` / `high` are the minimum / maximum over its scenario grid of
X_M, the monthly mobility residual of the household divided by its number of
priced home-based tours per month (EUR per home-based tour), and `km_low` /
`km_high` the distances they buy. A run of that script (ODiN 2023, without
the Budgetonderzoek tables) reproduces the file to its rounding (0.005).

| Script | What it shows |
|---|---|
| `breakdown.py` | for every cell, the scenario that sets `low` and `high`, and the share of the width each assumption accounts for on its own |
| `journeys_per_tour.py` | one-way journeys per home-based tour in ODiN: the divisor that turns the per-tour budgets into the per-journey budgets the model uses (`accessibility.legs_per_tour` = 2.2) |

    Rscript "X_M calc.R"                          # in a folder with data/ODIN_23.csv; writes out/
    python envelope/breakdown.py <that folder>/out --csv envelope/results/breakdown.csv
    python envelope/journeys_per_tour.py <ODiN csv> [...]

`results/breakdown.csv` is the breakdown of the current table. Its main
findings:

* `high` is set in every cell by the fixed lower bound of 8 tours per month
  per household (`n_min_fixed`), `low` by the highest tour rate of the
  household type over the deciles; the tour count accounts for 60-99% of the
  width. The share of the example basket given up (gamma) accounts for 30-56%
  for singles up to D7 and at most 18% elsewhere; the rent bracket and
  commuting for little, the PT tariff for nothing.
* 20 of the 36 cells lie above the highest Nibud anchor (singles from D8,
  couples from D7, families from D5). There no example basket exists and the
  budget is the residual after the MINIMUM basket: an upper bound.
* Journeys per home-based tour: 2.19 (2.17 weighted) on ODiN 2022-23 and on
  ODiN 2023 alike; 12% of tours have one journey, 66% two, 15% three.

`journeys_per_tour.py` prints aggregates only: ODiN microdata stay local.
