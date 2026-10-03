# ODiN aggregates for the reference-budget envelope

Aggregate tables derived from ODiN (Onderweg in Nederland, CBS and
Rijkswaterstaat), published so that the reference-budget envelope
(`ikob2.envelope`, `envelope/README.md`) can be rebuilt and checked without
the microdata.

| Folder | Microdata | Use |
|---|---|---|
| `2023/` | ODiN 2023 (`ODIN_23.csv`) | the published `data/envelope/reference_budgets.csv` (`envelope.aggregates = "2023"`) |
| `2022_2023/` | ODiN 2022 and 2023 pooled (`ODIN_22_23_clean.csv`) | the switch `envelope.aggregates = "2022_2023"` |

Made with `python -m ikob2.cli.envelope aggregates` (`ikob2.envelope.odin`,
which documents every definition). The 2023 part of the pooled file gives
exactly the tables of `2023/`; `tests/test_envelope.py` checks, where the
microdata are available, that a fresh aggregation equals the published one.

## Licence and disclosure

ODiN microdata are available through DANS under its access conditions and
are not in this repository. These files contain aggregates only: weighted
means (person weight FactorP, split over income deciles) and the number of
respondents behind each. Their publication was approved by the author.
Some cells rest on few respondents because the tables are complete over
household type x decile x age band, including combinations that hardly
occur:

| Table | Cells | n < 5 | n < 30 | smallest n |
|---|---|---|---|---|
| 2023 `band_rates` (cell level) | 105 | 25 | 28 | 1 |
| 2023 `commute_tours` | 40 | 0 | 8 | 11 |
| 2023 `commuter_rates` (cell level) | 49 | 8 | 10 | 2 |
| 2022-23 `band_rates` (cell level) | 108 | 21 | 29 | 1 |
| 2022-23 `commute_tours` | 40 | 0 | 1 | 23 |
| 2022-23 `commuter_rates` (cell level) | 50 | 3 | 10 | 2 |

The envelope uses a cell's own discretionary tour rate only from 150
respondents (`envelope.min_band_n`, otherwise the pooled rate) and a
commuter rate only from 50 (`envelope.min_commuter_n`); the other rates
enter at any cell size.

## Tables

All rates are per person per survey day (the envelope multiplies by
`envelope.days_per_month`); columns starting with `j` count priced one-way
journeys instead of priced home-based tours.

| File | Columns |
|---|---|
| `crosswalk.csv` | income (ODiN HHGestInkG class), quantile, share of the class in the decile |
| `band_rates.csv` | level (cell, type, quantile), hh_type, quantile, age_band, rate_disc / jrate_disc (non-commute), rate_comm / jrate_comm (unreimbursed commute; cell level), n |
| `composition.csv` | hh_type, quantile, weighted mean household size and members per age band (HHLft1-4) |
| `passenger_share.csv` | hh_type, share of tours (pass_share) and journeys (pass_share_j) as car passenger |
| `commute_tours.csv` | hh_type, quantile, km by car and by other modes per unreimbursed commute tour (and per journey: `_j`), share by other modes, n tours |
| `commuter_rates.csv` | level (cell, quantile), hh_type, quantile, commute tours (rate) and journeys (jrate) per day of unreimbursed commuters, n |
| `pt_km.csv` | km of train and of bus/tram/metro journeys (unweighted totals) |
| `units.csv` | priced journeys per priced home-based tour |
| `km_cdf.csv` | unit (tour, journey), km, weighted share of non-commute priced tours / journeys up to km (a grid from 0.5 to 1,000 km) |
