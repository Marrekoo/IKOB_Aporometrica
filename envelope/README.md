# Reference-budget envelope: provenance

`data/envelope/reference_budgets.csv` (the paper's Table 6) gives, per
household type and income decile, `low` / `high`: the minimum / maximum over a
scenario grid of X_M, a household's monthly mobility residual divided by its
number of priced home-based tours per month (EUR per home-based tour), and
`km_low` / `km_high`, the distances they buy. It was made by the R script
`X_M calc.R` (not in this repository). `ikob2.envelope` re-derives it from
source tables and ODiN aggregates and reproduces the file exactly.

## Re-derivation (`ikob2.envelope`, `cli.envelope`)

    python -m ikob2.cli.layout --root <root> create      # seeds inputs/envelope/sources/
    python -m ikob2.cli.envelope --data-root <root> aggregates   # ODiN -> intermediate/envelope/odin/
    python -m ikob2.cli.envelope --data-root <root> build        # -> intermediate/envelope/

| Stage | Module | Input | Output |
|---|---|---|---|
| source tables | `sources` | `data/envelope/sources/*.csv`, each with a `source` column | |
| ODiN aggregates | `odin` | ODiN 2023 (`envelope.odin`) | tour rates per household type x decile x age band, household composition, passenger share, commute tours, commuter rates, PT km; aggregates only |
| baskets and anchors | `nibud` | Nibud basket (B.4.1), Warnaar anchors, social-assistance incomes, rents, CBS equivalence factors, price index | `anchors.csv`: income, rent, minimum and example basket at every anchor |
| income axis | `income` | CBS income percentiles | the standardised income of each decile |
| residuals | `nibud` | anchors, income axis | `envelope.csv`: residual after the example and the minimum basket per decile and rent scenario |
| tours and X_M | `xm` | residuals, ODiN aggregates, car costs | `tour_bounds.csv`, `grid.csv` (X_M for every scenario), `reference_budgets.csv` |

The source tables:

| Table | Content |
|---|---|
| `nibud_basket` | the minimum basket per post and Nibud household (EUR/month, Jan 2023); the AOW single's rent corrected to 543 (noted) |
| `nibud_households`, `nibud_posts`, `nibud_type_keys` | household types and published totals; the role of each post; which Nibud household represents each model household type |
| `bijstand_published` | income at social-assistance level and the published saldo |
| `warnaar_anchors` | Warnaar's net income, rent and residual `b_norm` at minimum wage, modal and 1.5 x modal |
| `rents`, `equivalence_cbs`, `price_index` | rent lineages, CBS equivalence factors, the uprating kappa = 1107 / 1076 |
| `cbs_income_percentiles` | CBS percentiles p10..p90, 2021-2024 (table ID not recorded in the R script; unverified there) |
| `car_bundles`, `car_class` | fixed and per-km car costs per class; the class of each household type |
| `odin_household_types` | ODiN HHSam -> household type |

Parameters (gamma grid, the lower bound of 8 tours per month, thresholds,
PT rates) are in `defaults.toml` [envelope]. The test
`tests/test_envelope.py` checks the stages and, with the ODiN aggregates in
the data folder, that the result equals `reference_budgets.csv`.

Two properties of the R script are reproduced as they are and marked in the
code: the car bundles only enter the scenarios of the "chipkaart" PT tariff
(a side effect of `tidyr::crossing()` sorting; it changes the km columns,
not the budgets), and Warnaar's couple at minimum wage implies an example
basket below the minimum basket, so that anchor has no gamma range.

## Analyses of the current table

| Script | What it shows |
|---|---|
| `breakdown.py` | for every cell, the scenario that sets `low` and `high`, and the share of the width each assumption accounts for on its own |
| `export_sources_from_r.R` | writes the literal tables of `X_M calc.R`, the numbers of `data/envelope/sources/` |
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
