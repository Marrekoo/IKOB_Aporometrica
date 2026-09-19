# Household-type x income segments (`ikob2.segments`)

Python port of `gspree_segments.R`. Produces, per CBS buurt, persons in
44 segments: 4 household types (`single`, `couple`, `single_parent`,
`couple_children`) x 11 income classes (`D1`..`D10`, `onbekend`).

## Method

1. **Seed** (CBS 86161NED, municipality level): household counts per
   type x standardised-income decile. Decile columns are percentages of
   the household total, which is published x 1000; `onbekend` is the
   non-negative remainder.
2. **Structure model**: a Poisson-family GLM
   `mu = exp(alpha_cell + b_sted_cell * z_sted + b_woz_cell * z_woz)`
   per (type, class) cell, fitted on municipalities (`statsmodels`).
   Without enough covariate data it degrades to the saturated cell model
   (cell means). This captures the type x income *association*.
3. **Marginals per buurt** from KWB: household types (KWB percentages;
   single-parent split from 71487ned; fallback national mix if missing)
   and income classes (municipal income shape, with D1-D4 calibrated to
   the buurt's own KWB low-income share).
4. **IPF** rakes each buurt's predicted table to its two marginals.
5. Households are converted to persons with fixed household sizes
   (1 / 2 / 2.4 / 3.6); a second layer rescales each row to KWB
   `aantal_inwoners`.

## Usage

    python -m ikob2.cli.segments fetch --out data/statline   # once, network
    python -m ikob2.cli.segments run --kwb "<wijkenbuurten_2022_v3.gpkg>" \
        --statline data/statline --out output/nl_segments.gpkg

`fetch` stores CSV snapshots of the two StatLine tables, so runs are
offline and reproducible. CBS suppression codes (`-99999999`) in the
KWB inputs are treated as missing.

## Using the segments in the engine (`ikob2.segments.bridge`)

The 44 segments enter the accessibility engine as ordinary
`SegmentedRunner` segments:

    from ikob2.segments.bridge import (build_segments,
        populations_for_zones, load_envelope)

    env = load_envelope("data/envelope.csv")     # cost margin per segment
    segs = build_segments(CurveSpec("weibull", (k, eta)),
                          envelope=env, money_cost_id=fare.matrix_id,
                          copula=CopulaSpec("gumbel", 1.5),
                          pool_by="income_class")
    pops = populations_for_zones(result.population_scaled, zone_codes, segs)

    runner = SegmentedRunner(decay_epsilon=1e-9)
    a = runner.run_hansen(state, segs, cost_matrices={...},
                          opportunities={...})            # paper's measure
    shen = runner.run(state, segs, pops, cost_matrices={...})  # competition

* **Reference budgets** (`data/envelope/reference_budgets.csv`, the
  paper's Table 6: EUR per tour, low and high, plus the distance
  equivalents) load with `load_reference_budgets(path, censored=...)`.
  Deciles Q1..Q10 are the standardised-income deciles D1..D10. The first
  decile is censored (no bounds): `censored="atom"` makes it a segment
  for whom no priced trip is acceptable, `"drop"` leaves it out. There
  is no row for `onbekend`; pass `only=envelope_segment_names(env)` to
  run the 40 ranked segments.
  **Basis.** The budgets come from ODiN tours (trip chains of one or more
  legs, not necessarily round trips), the fare matrix prices one one-way
  trip. By default the table is read as a per-one-way-trip budget
  (`legs_per_tour=1.0`, no conversion). Pass the average number of priced
  legs per tour, as a number or per household type, to divide the bounds
  down: `load_reference_budgets(path, legs_per_tour={"single": 1.6, ...})`
  (or `rescale_budgets` on a loaded envelope). The atom does not change.
* **Envelope table** (CSV): `household_type, income_class, low, high[, atom]`
  - the per-trip cost threshold is uniform on [low, high] EUR; `atom` is
  the share for whom no priced trip is acceptable (a censored cell is
  `atom = 1`). Every requested segment needs a row (`only=` restricts).
  Free modes pass no envelope and get a time-only filter, so all their
  segments share one composed matrix.
* **Rectangular runs.** For a study area, pass `state=None` to
  `run_hansen` with `cost_matrices` of shape (origins x destinations)
  and pool vectors over the destinations: `a = runner.run_hansen(None,
  segs, cost_matrices={"time": t, fare_id: c}, opportunities=pools)`.
  Origins only need skim rows; destinations only need jobs. Utrecht
  city (111 buurten) against all 14,412 buurten is 6 MB per matrix
  instead of 831 MB square. Variants and zone weights need a state and
  are not available there. The competition-adjusted `run` needs every
  origin's population and stays square.
* **Hansen vs Shen.** `run_hansen` is `a_i = sum_j D_j f(t_ij, c_ij)` per
  segment with no competition and no populations: the expected number of
  acceptable opportunities. `run` is the competition-adjusted measure.
* **Pools** carry income-matched opportunities: `pool_by="income_class"`
  needs one opportunity vector per income class.
* **Job pools from LISA:** `jobs_impute` spreads municipal LISA sector
  jobs over buurten, `jobs.sector_income_weights` + `sector_pools` turn
  them into one pool per income class (`docs/data_lineage.md`).
* `populations_for_zones` aligns segment persons to the engine's zone
  order; zones without a row are empty. `aggregate_by` reports
  population-weighted means by income class or household type.

## Provenance and baseline

The method is a port of the earlier R script (`gspree_segments.R`). Before
its sentinel handling was removed, the port reproduced that script's
output for the 2022 KWB file (14,412 buurten) to 8e-7 persons in any
segment cell, so the algorithm itself is unchanged. The R script left the
CBS suppression code `-99999999` in `stedelijkheid` and `gem_woz`
(181 and 1,890 buurten in the 2022 file), which entered the municipal
means and z-scores and pushed them as far as -15 sd, so the fitted slopes
were extrapolated far outside the data. This is removed: those values are
missing. Relative to the R output the within-buurt segment distribution
shifted by a mean total variation distance of 0.031 (95th percentile
0.155), national segment totals by up to 16%.

**The output of this pipeline is the baseline.** The R output is not kept
and there is no parity test against it. Regression tests pin the
pipeline's own numbers: a synthetic case that always runs, and the full
2022 KWB run (set `IKOB_KWB_2022_GPKG` to the CBS file to enable it).

## Implementation notes

* **Vectorised.** All buurten are raked in one batched IPF (each table
  stops at its own convergence iteration) instead of a per-buurt loop.
* **StatLine keys** are fetched from the OData `TypedDataSet` endpoint
  (keys, not titles) and stored as snapshots.
* Non-finite structure-model predictions are floored like NA (1e-8).

## Known limitations (inherited from the R design, not changed)

* The model is fitted on municipality-level covariates standardised
  across municipalities, but predicted with covariates standardised
  across buurten (different mean/sd). Coefficients are therefore
  applied on a different scale than they were estimated on.
* D1-D4 calibration only rescales the municipal shape; it cannot create
  mass where the municipal shape has none. In 18 buurten of 5 small
  municipalities (2022 file) the D1-D4 share misses its target.
* `onbekend` is the rounding remainder of published integer
  percentages, not a real income group.
* 606 buurten with zero households (and 95 with suppressed counts) are
  skipped and get zeros.
