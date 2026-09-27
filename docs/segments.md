# Household-type x income segments (`ikob2.segments`)

Per CBS buurt, persons in 44 segments: 4 household types (`single`,
`couple`, `single_parent`, `couple_children`) x 11 income classes (`D1` ..
`D10`, the standardised disposable household income deciles, and
`onbekend`). The method is GSPREE: a structure model supplies the
association between household type and income, and IPF fits it to each
buurt's own margins.

## Method

1. **Seed** (CBS 86161NED, municipality level): households per type x
   income decile. Decile columns are percentages of the household total,
   which is published x 1000; `onbekend` is the non-negative remainder of the
   rounded percentages (`marginals.build_gemeente_seed`).
2. **Structure model** (`structure`): per (type, class) cell a Poisson-family
   GLM (`statsmodels`),
   `mu = exp(alpha_cell + b_sted_cell * z_sted + b_woz_cell * z_woz)`,
   fitted on municipalities with standardised urbanisation and mean house
   value. With fewer than `segments.structure_min_rows` rows or
   `structure_min_gemeenten` municipalities it falls back to the saturated
   cell model (cell means). Non-finite predictions are floored at 1e-8.
3. **Buurt margins** (`marginals`, `kwb`): household types from KWB
   percentages, with the single-parent share from 71487ned (fallback
   `segments.single_parent_fallback_share`, and the national mix
   `segments.fallback_hh_composition` where KWB has no data); income classes
   from the municipal income shape with D1-D4 calibrated to the buurt's own
   KWB low-income share (`segments.local_income_calibration`).
4. **IPF** (`ipf.ipf_batch`) rakes every buurt's predicted table to its two
   margins in one batched computation; each table stops at its own
   convergence (`segments.ipf_tol`, `ipf_max_iter`).
5. **Persons**: households are converted to persons with household sizes
   1 / 2 / 2.4 / 3.6 (`segments.hh_size`), and each row is rescaled to the KWB
   `aantal_inwoners` (`population_scaled`; `household_based` keeps the
   unscaled layer, `--population-basis`).

Every negative KWB value is a CBS suppression code and is read as missing.
Buurten with zero or suppressed household counts get zeros.

## Usage

    python -m ikob2.cli.segments --data-root <root> run

reads the KWB file of `accessibility.kwb_year` and the StatLine snapshots,
and writes `intermediate/segments/nl_segments.gpkg` (layer `buurt_segments`)
and a per-buurt diagnostics CSV (IPF status, iterations, largest margin
error); `--kwb`, `--statline` and `--out` override the paths.
`--gemeenten GMxxxx ...` (before the subcommand) restricts the output;
`segments.income_period` and `segments.children_period` (or
`--income-period`, `--children-period`) choose the StatLine periods.
`cli.accessibility` runs the same pipeline in memory.

## Reference budgets (`segments.bridge`)

`inputs/envelope/reference_budgets.csv` (seeded from the repository's
`data/envelope/`) gives per household type and decile
the bounds `[low, high]` of the per-journey cost threshold (EUR), plus the
car-kilometre equivalents. `load_reference_budgets(path, censored=...)`
loads it:

* Decile 1 has no bounds. `censored="atom"` makes it a segment with atom 1
  (no priced journey acceptable); `"drop"` leaves it out.
* `onbekend` has no row; `envelope_segment_names(env)` lists the 40 segments
  with a budget, and only those are run.
* **Basis: per home-based tour, converted to per journey.** The envelope
  script divides a household's monthly mobility residual by its number of
  priced *home-based tours*: a person's regular trips in order, a new tour
  starting after every trip that ends at home. A tour is therefore a chain of
  one-way journeys (ODiN verplaatsingen) from home back to home. The model
  prices one one-way journey door to door (the whole fare plus shared-bicycle
  rentals), so the bounds are divided by the number of journeys per tour,
  `accessibility.legs_per_tour` = 2.2. On ODiN 2022-23 (and on ODiN 2023, the
  envelope script's input) a priced home-based tour has 2.19 journeys on
  average (2.17 weighted); 12% of tours have one journey, 66% two, 15% three;
  the median tour is 19 km against 8 km per priced journey
  (`envelope/journeys_per_tour.py`). The parameter name is historical: it
  counts journeys, not the legs ("ritten") within a journey.
* **What sets the bounds.** `low` and `high` are the minimum and maximum of
  the per-tour budget over the envelope script's scenario grid (share of the
  example basket given up, tours per month, rent bracket, commuting, PT
  tariff); `envelope/breakdown.py` shows which scenario sets each bound.
  `high` is set in every cell by a fixed 8 tours per month, `low` by the
  highest tour rate of the household type; the tour count accounts for 60-99%
  of the width. Above the highest Nibud anchor (20 of the 36 cells: singles
  from D8, couples from D7, families from D5) the bounds rest on the minimum
  basket and are upper bounds.
* **Non-monotone in income.** Single households: D6 low 25.26 against D5
  34.58. The high-rent scenario of D6 jumps to the anchor rent of 880 EUR a
  month (a free-market rent at 1.5 x modal income) where D5 still has 540; it
  is a property of the rent bracket, not of travel behaviour.
* **Other bases.** `legs_per_tour` (a number, or a mapping per household
  type) sets the divisor: `load_reference_budgets(path, legs_per_tour=1)`
  reads the table as per journey, `rescale_budgets(env, ...)` converts a
  loaded envelope. The atom does not change.

A general **envelope table** (`load_envelope`) has columns
`household_type, income_class, low, high[, atom]`; every requested segment
needs a row.

## Segments in the engine

`build_segments` turns an envelope into engine `Segment`s: a time margin
shared by all segments of a mode and, for priced modes, each segment's cost
margin, composed with a copula and pooled by income class:

    from ikob2.domain.filter_config import CurveSpec, CopulaSpec
    from ikob2.engine.runner import SegmentedRunner
    from ikob2.segments.bridge import build_segments, load_reference_budgets
    from ikob2.utils.paths import DataLayout

    env = load_reference_budgets(DataLayout(root).budgets(),
                                 censored="atom")
    segs = build_segments(CurveSpec("weibull", (k, eta)), envelope=env,
                          money_cost_id="pt_fare",
                          copula=CopulaSpec("gumbel", 1.5),
                          pool_by="income_class")
    runner = SegmentedRunner(decay_epsilon=1e-9)
    a = runner.run_hansen(None, segs,
                          cost_matrices={"time": t, "pt_fare": c},
                          opportunities=pools)      # income class -> jobs vector

* **Rectangular runs.** With `state=None`, `run_hansen` takes origins x
  destinations matrices and pool vectors over the destinations: origins need
  only skim rows, destinations only jobs (111 x 14,318 float32 is 6 MB per
  matrix). Zone weights need a `ModelState`.
* **Hansen and Shen.** `run_hansen` is `a_i = sum_j D_j f(t_ij, c_ij)` per
  segment, without competition. `run` is the Shen measure; it needs every
  origin's population and is square.
* **Pools.** `pool_by="income_class"` needs one opportunity vector per income
  class: the job pools of `jobs.sector_pools` (`data_lineage.md`).
* **Free modes** pass no envelope and get a time-only filter, so all their
  segments share one composed matrix.
* `populations_for_zones` aligns segment persons to the engine's zone order
  (missing zones are empty); `aggregate_by` gives population-weighted means
  by income class or household type.

`run.accessibility.run_accessibility` wraps all of this for the study-area
runs, including job types, specifications, option sets and availability.

## Baseline

`tests/test_segments.py` pins the pipeline's own output: a synthetic case
that always runs and the full 2022 KWB run (set `IKOB_KWB_2022_GPKG` to the
CBS file to enable it).

## Known limitations

* The structure model is fitted on covariates standardised across
  municipalities but predicted with covariates standardised across buurten
  (a different mean and sd), so the coefficients are applied on another scale
  than they were estimated on.
* D1-D4 calibration only rescales the municipal shape; it cannot create mass
  where that shape has none. In 18 buurten of 5 small municipalities (2022)
  the D1-D4 share misses its target.
* `onbekend` is a rounding remainder, not an income group.
* 606 buurten with zero households and 95 with suppressed counts get zeros.
