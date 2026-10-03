# Reference-budget envelope

`data/envelope/reference_budgets.csv` gives, per household type and income
decile, the money a household can spend on a priced one-way journey once
everything a capabilities approach counts as basic (Nussbaum) is paid for:
`low` / `high` bound a uniform distribution across households, `central` is
its midpoint, `km_low` / `km_high` are the distances they buy, and `unit`,
`upper_bound` and `gate_slack` describe the cell. EUR per journey, 2022
euros (the price level of the KWB population year).

It is built by `ikob2.envelope` from published source tables and ODiN
aggregates (below).

## Re-derivation (`ikob2.envelope`, `cli.envelope`)

    python -m ikob2.cli.layout --root <root> create      # seeds inputs/envelope/sources/ and odin/
    python -m ikob2.cli.envelope --data-root <root> build        # -> intermediate/envelope/
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

The source tables (the `source` column names the publication behind each
row):

| Table | Content |
|---|---|
| `nibud_basket` | the minimum basket per post and Nibud household (EUR/month, Jan 2023); the AOW single's rent corrected to 543 (noted) |
| `nibud_households`, `nibud_posts`, `nibud_type_keys` | household types and published totals; the role of each post; which Nibud household represents each model household type |
| `bijstand_published` | income at social-assistance level and the published saldo |
| `warnaar_anchors` | Warnaar's net income, rent and residual `b_norm` at minimum wage, modal and 1.5 x modal |
| `rents`, `equivalence_cbs`, `price_index` | rent lineages, CBS equivalence factors, the uprating kappa = 1107 / 1076 |
| `cbs_income_percentiles` | CBS percentiles p10..p90, 2021-2024 (the StatLine table is not recorded) |
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

**The choices**, as parameters in `defaults.toml` [envelope], each with the
alternative the code also offers:

| Parameter | Default | Alternative | Why |
|---|---|---|---|
| `unit` | `journey`: EUR per priced one-way journey | `tour`: EUR per priced home-based tour | the model prices journeys; a journey table is not divided by `legs_per_tour` |
| `spread` | `gamma`: low/high over gamma only | `all`: over every assumption as well | the spread describes differences between households; uncertainty about assumptions belongs in sensitivity runs |
| `price_base` | `2022`: every amount in 2022 euros | `published`: each input at its own price date | the price level of the KWB population year (below) |
| `income_bridge` | `per_adult`: + the basic health premium per adult | `none` | CBS disposable income is net of the premium, which the Nibud basket also contains |
| `n_lower` | `lowest_decile`: the lowest journey rate over the deciles | `fixed`: 8 per household and month | only matters with `spread` all; data-based instead of a constant |
| `aggregates` | `2022_2023` | `2023` | ODiN years pooled, as in the rest of the model |
| `car_all_tariffs` | `true`: car options in every PT-tariff scenario | `false`: only in the "chipkaart" scenarios | changes the km columns only |
| `gamma_anchor` | `0.5` | `0` | b_norm is the midpoint of the outer anchors |

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

**Sensitivity** to each choice: the alternative of one switch at a time
against the defaults, as the model reads the budgets (EUR per journey; a
table per tour divided by `legs_per_tour` = 2.2); median ratio over the
cells (`envelope/switches.py`; per cell `results/switch_effects.csv`):

| Variant | low D2-D4 | central D2-D4 | high D2-D4 | low D2-D10 | central D2-D10 | high D2-D10 |
|---|---|---|---|---|---|---|
| defaults | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| unit: per tour | 0.94 | 0.94 | 0.94 | 0.94 | 0.94 | 0.94 |
| n_lower: fixed 8 tours | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| income_bridge: none | 0.58 | 0.68 | 0.74 | 0.88 | 0.88 | 0.90 |
| aggregates: ODiN 2023 | 0.98 | 0.99 | 0.99 | 0.98 | 0.98 | 0.98 |
| car_all_tariffs: false | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| price_base: published | 1.17 | 1.08 | 1.03 | 1.04 | 1.04 | 1.03 |
| spread: all assumptions | 0.72 | 1.00 | 1.22 | 0.93 | 1.00 | 1.30 |
| gamma_anchor: 0 | 1.36 | 1.13 | 1.00 | 1.00 | 1.00 | 1.00 |

The residuals of D2-D4 are small differences between large amounts (income
minus rent minus basket), so a few per cent on an input moves them by much
more: the 2022 price level (inputs 1.5-3.7% lower than at their own
dates) lowers their low end by 15%.

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

## Scripts

| Script | What it shows |
|---|---|
| `switches.py` | the sensitivity to every switch (`results/switch_effects.csv`, `results/switch_summary.csv`) |
| `journeys_per_tour.py` | one-way journeys per home-based tour in ODiN: the divisor for a table per tour (`accessibility.legs_per_tour` = 2.2) |

    python envelope/switches.py
    python envelope/journeys_per_tour.py <ODiN csv> [...]

Journeys per home-based tour: 2.19 (2.17 weighted) on ODiN 2022-23 and on
ODiN 2023 alike; 12% of tours have one journey, 66% two, 15% three.
`journeys_per_tour.py` prints aggregates only: ODiN microdata stay local.
