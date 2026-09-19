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
offline and reproducible. `--covariate-sentinels keep` reproduces the R
behaviour exactly (see below); the default treats CBS suppression codes
as missing.

## Differences from the R script

* **Sentinel in covariates (deliberate).** R keeps the CBS suppression
  code `-99999999` in `stedelijkheid` and `gem_woz`, so it enters the
  municipal means and z-scores (2022 file: 181 and 1,890 buurten).
  Default here: missing. `keep` restores R for parity.
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
