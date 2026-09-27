# Reference-budget envelope

`data/envelope/reference_budgets.csv` gives, per household type and income
decile, the money a household can spend on a priced one-way journey once
everything a capabilities approach counts as basic (Nussbaum) is paid for:
`low` / `high` bound a uniform distribution across households, `central` is
its midpoint, `km_low` / `km_high` are the distances they buy, and `unit`,
`upper_bound` and `gate_slack` describe the cell. EUR per journey, 2022
euros (the price level of the KWB population year).

It is built by `ikob2.envelope` from published source tables and ODiN
aggregates (below). The method starts from the R script `X_M calc.R` (not
in this repository), whose table the same code reproduces exactly with that
script's settings (`envelope/x_m_calc.toml` ->
`data/envelope/reference_budgets_x_m_calc.csv`, EUR per home-based tour);
the adopted method changes it in the ways listed under *Method*.

## Re-derivation (`ikob2.envelope`, `cli.envelope`)

    python -m ikob2.cli.layout --root <root> create      # seeds inputs/envelope/sources/ and odin/
    python -m ikob2.cli.envelope --data-root <root> build        # the adopted method -> intermediate/envelope/
    python -m ikob2.cli.envelope --params envelope/x_m_calc.toml --data-root <root> build   # X_M calc.R
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

## Method

**The distribution.** Families facing the same hardship react differently:
some give up more of the Nibud example basket than others to be able to
travel. The share given up, gamma, is taken uniform over [0, 1] across the
households of a cell, so the budget per journey is uniform between the
residual after the example basket (`low`, gamma = 0) and after the minimum
basket (`high`, gamma = 1), each spread over the household's journeys. All
other assumptions sit at their central value (tours per month from ODiN,
interpolated rent, average unreimbursed commuting, Nibud's PT rate). Warnaar's
published residual b_norm is the midpoint between his two outer anchors and
adds no information beyond them (`gamma_anchor` = 0.5).

Where no example basket exists, gamma has no range and the budget is one
amount (`low` = `high`, a step margin): above the highest Nibud anchor (19
cells, `upper_bound`: the residual after the minimum basket, the most the
household could spend) and at couples and couples with children in D2
(Warnaar's couple at minimum wage implies an example basket below the
minimum basket). Couples in D2 therefore get the most generous point, which
lies above the low end of D3.

**The choices**, as parameters in `defaults.toml` [envelope] (the settings of
X_M calc.R in `envelope/x_m_calc.toml`):

| Parameter | Adopted | X_M calc.R | Why |
|---|---|---|---|
| `unit` | `journey`: EUR per priced one-way journey | `tour`: EUR per priced home-based tour | the model prices journeys; a journey table is not divided by `legs_per_tour` |
| `spread` | `gamma`: low/high over gamma only | `all`: over every assumption as well | the spread describes differences between households; uncertainty about assumptions belongs in sensitivity runs |
| `price_base` | `2022`: every amount in 2022 euros | `published`: each input at its own price date | the price level of the KWB population year (below) |
| `income_bridge` | `per_adult`: + the basic health premium per adult | `none` | CBS disposable income is net of the premium, which the Nibud basket also contains |
| `n_lower` | `lowest_decile` | `fixed`: 8 tours per household and month | only matters with `spread` all; data-based instead of a constant |
| `aggregates` | `2022_2023` | `2023` | ODiN years pooled, as in the rest of the model |
| `car_all_tariffs` | `true` | `false` (car options only in the "chipkaart" scenarios, a side effect of `tidyr::crossing()` sorting) | changes the km columns only |
| `gamma_anchor` | `0.5` | `0.5` | b_norm is the midpoint of the outer anchors |

**Price level.** Every input is converted from its own price date to the
2022 year average with the CPI (`sources/price_levels.csv`): the Nibud basket,
the social-assistance incomes and Nibud's PT rates from January 2023 (x
121.43 / 123.23), Warnaar's anchors and the car costs from Warnaar's level,
kappa above the basket, the tussenrapport rents from July 2022, the CBS
incomes and the basic premium (EUR 1,653 a year) from the 2023 average (x
121.43 / 126.04). Incomes and minima stay those of 2023, the year of the
Nibud minimum budgets: only the price level changes. (Deflating the minima
but taking 2022 incomes would compare 2022 incomes with the 2023 social
minimum, which rose about 10% in real terms in January 2023, and would push
single parents in D2 below the social-assistance anchor.)

**Effect** of each choice on its own, from the X_M calc.R settings, and of
the adopted method as a whole, as the model reads the budgets (EUR per
journey; tour tables divided by `legs_per_tour` = 2.2); median ratio over
the cells (`envelope/switches.py`; per cell `results/switch_effects.csv`):

| Variant | low D2-D4 | central D2-D4 | high D2-D4 | low D2-D10 | central D2-D10 | high D2-D10 |
|---|---|---|---|---|---|---|
| X_M calc.R settings | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| unit: per journey | 1.05 | 1.06 | 1.06 | 1.05 | 1.05 | 1.06 |
| n_lower: lowest decile | 1.00 | 1.00 | 0.37 | 1.00 | 1.00 | 0.37 |
| income_bridge: per adult | 1.75 | 1.45 | 1.35 | 1.14 | 1.13 | 1.10 |
| aggregates: ODiN 2022-23 | 0.97 | 1.00 | 1.00 | 1.03 | 1.02 | 1.00 |
| car_all_tariffs | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| price_base: 2022 euros | 0.76 | 0.92 | 0.98 | 0.97 | 0.97 | 0.97 |
| spread: gamma only | 1.59 | 1.00 | 0.26 | 1.10 | 1.00 | 0.27 |
| gamma_anchor: 0 (not adopted) | 1.44 | 1.14 | 1.00 | 1.00 | 1.00 | 1.00 |
| adopted method (defaults) | 2.47 | 1.36 | 0.39 | 1.26 | 1.16 | 0.35 |

The residuals of D2-D4 are small differences between large amounts (income
minus rent minus basket), so a few per cent on an input moves them by much
more: the 2022 price level alone (inputs 1.5-3.7% lower) lowers their low
end by 24%.

**Income and rent between anchors.** The deciles are ordered by CBS
disposable income, which includes the allowances actually received, so the
loss of allowances as gross income rises (the poverty trap) compresses the
deciles but cannot make a higher decile poorer. Rent, however, is
interpolated between Warnaar's anchors: couples in D3 already lie above his
modal anchor, whose rent is much higher (no rent allowance, no social
housing), so from D2 to D3 a couple's rent rises by EUR 237 of a EUR 393
income gain and its residual after the minimum basket by only EUR 156. For
the other household types the rent step is small (single parents +46,
couples with children +49, singles -8). Income-tested benefits outside cash
income (remission of local taxes, special assistance, discount passes) are
not modelled: the Nibud basket charges full local taxes at every income.

## Analyses of the X_M calc.R table

| Script | What it shows |
|---|---|
| `breakdown.py` | for every cell, the scenario that sets `low` and `high`, and the share of the width each assumption accounts for on its own |
| `export_sources_from_r.R` | writes the literal tables of `X_M calc.R`, the numbers of `data/envelope/sources/` |
| `switches.py` | the effect of every switch on the budgets (`results/switch_effects.csv`, `results/switch_summary.csv`) |
| `journeys_per_tour.py` | one-way journeys per home-based tour in ODiN: the divisor for a table per tour (`accessibility.legs_per_tour` = 2.2) |

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
