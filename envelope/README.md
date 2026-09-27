# Reference-budget envelope: provenance

`data/envelope/reference_budgets.csv` (the paper's Table 6) gives, per
household type and income decile, `low` / `high`: the minimum / maximum over a
scenario grid of X_M, a household's monthly mobility residual divided by its
number of priced home-based tours per month (EUR per home-based tour), and
`km_low` / `km_high`, the distances they buy; `central` (the central
scenario), `unit` (tour or journey), `upper_bound` and `gate_slack` follow.
It was made by the R script `X_M calc.R` (not in this repository).
`ikob2.envelope` re-derives it from source tables and published ODiN
aggregates and reproduces the file exactly; switches in `defaults.toml`
[envelope] apply the corrections below one at a time.

## Re-derivation (`ikob2.envelope`, `cli.envelope`)

    python -m ikob2.cli.layout --root <root> create      # seeds inputs/envelope/sources/ and odin/
    python -m ikob2.cli.envelope --data-root <root> build        # -> intermediate/envelope/
    python -m ikob2.cli.envelope --data-root <root> --set envelope.unit=journey build
    python -m ikob2.cli.envelope --data-root <root> aggregates   # only with ODiN microdata

The ODiN aggregates of 2023 and 2022-23 are published in `data/envelope/odin/`
(what they contain, how they were made, cell sizes and licence:
`data/envelope/odin/README.md`), so `build` needs no microdata.

| Stage | Module | Input | Output |
|---|---|---|---|
| source tables | `sources` | `data/envelope/sources/*.csv`, each with a `source` column | |
| ODiN aggregates | `odin` | ODiN 2023 or 2022-23 | tour and journey rates per household type x decile x age band, household composition, passenger share, commuting, PT km, distance distribution; published in `data/envelope/odin/` |
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
PT rates, the switches below) are in `defaults.toml` [envelope].
`tests/test_envelope.py` checks the stages, that the published aggregates
reproduce `reference_budgets.csv` (also in CI), that every switch builds a
valid envelope, and, where the ODiN microdata are available, that a fresh
aggregation equals the published aggregates.

## Switches: corrections of the R method

Each switch is a parameter in `defaults.toml` [envelope]; the defaults
reproduce the published table.

| Switch | Default (X_M calc.R) | Alternative | What it corrects |
|---|---|---|---|
| `unit` | `tour`: EUR per priced home-based tour | `journey`: EUR per priced one-way journey | the model prices journeys; counting them directly replaces the divisor `legs_per_tour` (a journey table has `unit` = journey and is not divided) |
| `n_lower` | `fixed`: 8 tours per household and month | `lowest_decile`: the lowest tour rate of the household type | the fixed floor alone sets `high` in every cell |
| `income_bridge` | `none` | `per_adult`: + `basic_premium_eur` (137.75) per adult | CBS disposable income is net of the basic health premium, which the Nibud basket also contains |
| `aggregates` | `2023` | `2022_2023` | ODiN years pooled, as in the rest of the model |
| `car_all_tariffs` | `false`: car options only in the "chipkaart" scenarios | `true` | a side effect of `tidyr::crossing()` sorting; changes the km columns only |
| `quantile_kappa` | `true`: decile baskets uprated by kappa | `false`: baskets at their Jan-2023 prices | the price level of the baskets against the 2023 income axis (a choice, not settled) |
| `gamma_anchor` | `0.5`: Warnaar's b_norm is the midpoint | `0`: b_norm is the example-basket residual | what b_norm means in the source (to be confirmed) |
| `spread` | `all`: low/high over every assumption | `gamma`: low/high over gamma only, the rest central | separates the spread across households (gamma) from uncertainty about assumptions, which then belongs in sensitivity runs |

Always exported: `central` (gamma 0.5, N_emp, rent interp, commuting
average, PT nibud_flat), `upper_bound` (the cell lies above the highest
Nibud anchor: residual after the minimum basket) and `gate_slack` (the high
budget reaches more than 95% of ODiN's non-commute tours or journeys).

Effect, as the model reads the budgets (EUR per journey; tour tables divided
by `legs_per_tour` = 2.2), median ratio to the baseline over the cells
(`envelope/switches.py`; per cell: `results/switch_effects.csv`):

| Variant | low D2-D4 | central D2-D4 | high D2-D4 | low D2-D10 | central D2-D10 | high D2-D10 |
|---|---|---|---|---|---|---|
| baseline (reproduces the published table) | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| unit: per journey | 1.05 | 1.06 | 1.06 | 1.05 | 1.05 | 1.06 |
| n_lower: lowest decile | 1.00 | 1.00 | 0.37 | 1.00 | 1.00 | 0.37 |
| income_bridge: per adult | 1.75 | 1.45 | 1.35 | 1.14 | 1.13 | 1.10 |
| aggregates: ODiN 2022-23 | 0.97 | 1.00 | 1.00 | 1.03 | 1.02 | 1.00 |
| car_all_tariffs | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| quantile_kappa: off | 1.19 | 1.11 | 1.08 | 1.03 | 1.03 | 1.02 |
| gamma_anchor: 0 | 1.44 | 1.14 | 1.00 | 1.00 | 1.00 | 1.00 |
| spread: gamma only | 1.59 | 1.00 | 0.26 | 1.10 | 1.00 | 0.27 |
| corrections combined | 1.87 | 1.55 | 0.56 | 1.25 | 1.22 | 0.49 |
| corrections + spread gamma | 2.80 | 1.55 | 0.42 | 1.30 | 1.22 | 0.37 |

"Corrections combined" is `unit` journey, `n_lower` lowest_decile,
`income_bridge` per_adult, `aggregates` 2022_2023 and `car_all_tariffs`;
`quantile_kappa`, `gamma_anchor` and `spread` are choices that need a
decision rather than corrections. Per journey the budgets rise by about 5%
against the tour table divided by 2.2, because a priced tour has 2.08 priced
journeys (2.2 counts all journeys, walking and cycling included).

Two properties of the R script are reproduced by default and marked in the
code: the car bundles only enter the scenarios of the "chipkaart" PT tariff
(`car_all_tariffs`), and Warnaar's couple at minimum wage implies an example
basket below the minimum basket, so that anchor has no gamma range.

## Analyses of the current table

| Script | What it shows |
|---|---|
| `breakdown.py` | for every cell, the scenario that sets `low` and `high`, and the share of the width each assumption accounts for on its own |
| `export_sources_from_r.R` | writes the literal tables of `X_M calc.R`, the numbers of `data/envelope/sources/` |
| `switches.py` | the effect of every switch on the budgets (`results/switch_effects.csv`, `results/switch_summary.csv`) |
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
* 19 of the 36 cells lie above the highest Nibud anchor (singles from D8,
  couples from D7, families from D5). There no example basket exists and the
  budget is the residual after the MINIMUM basket: an upper bound.
* Journeys per home-based tour: 2.19 (2.17 weighted) on ODiN 2022-23 and on
  ODiN 2023 alike; 12% of tours have one journey, 66% two, 15% three.

`journeys_per_tour.py` prints aggregates only: ODiN microdata stay local.
