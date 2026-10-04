# Figures and tables of the paper (R)

The figures and the rounded tables of "Either you can reach it or you cannot"
(Utrecht shared bicycles). The scripts plot and round what the model
computes; they do not recompute statistics. Plain R, no other tools.

## Run

    export IKOB_DATA_ROOT=/path/to/data       # or a line in ~/.Renviron
    Rscript paper/figures/run_all.R           # all nine steps (about 15 seconds)
    Rscript paper/figures/run_all.R 3 5       # only some steps (step 1 must have run once)

or in RStudio: open `paper/figures/` as the working directory and
`source("run_all.R")`. Output goes to `<data root>/outputs/paper_figures/`
(`IKOB_FIGURES_OUT` for another folder).

The data folder needs:

* `outputs/comparisons/specs/` and `outputs/comparisons/targeting/`
  (`python -m ikob2.cli.paper_tables`) and, for the precision of the
  numbers, `outputs/comparisons/precision/precision.csv`
  (`python -m ikob2.cli.precision`);
* `outputs/runs/scen_s0`, `scen_s4`, `scen_s2t` (`paper/runs.toml`), for the map;
* `inputs/tariffs/lime_price_zones_overvecht_kanaleneiland.csv`,
  `inputs/hubs/utrecht_hubs.csv` and `intermediate/hubs/utrecht_hubs_s2t.csv`
  (seeded by `cli.layout create`).

The data deposit of the paper (`docs/reproduce.md`) holds all of these.

Settings (specifications, runs of the map, rounding) are in the CONFIG block
of `scripts/00_setup.R`. Tested with R 4.6.1 and dplyr 1.2.1, tidyr 1.3.2,
readr 2.2.0, purrr 1.2.2, stringr 1.6.0, tibble 3.3.1, ggplot2 4.0.3,
scales 1.4.0, sf 1.1.1, patchwork 1.3.2, jsonlite 2.0.0.

## Steps

| Step | Script | Output | Paper |
|---|---|---|---|
| 1 | `01_load.R` | reads the paper tables, the precision table and the map inputs; `cache/data.rds` | |
| 2 | `02_baseline.R` | baseline accessibility by decile: M0, M1, M2 and the M3 range | 4.2 |
| 3 | `03_incidence.R` | gain by decile: S1 against S2c, S4 against S2t; appendix: S1a, S4a, S3c, S3t | 4.3 |
| 4 | `04_agreement.R` | correlation with M2 of levels and of the gains of each scenario | 4.2-4.3 |
| 5 | `05_interchange.R` | R: dispersion across segments (CV), share of segments where the price cut adds nothing, by decile | 4.1 |
| 6 | `06_targeting.R` | price cut by income, by address, and hubs: coverage, cost, gain per euro; gain by group and place | 4.4 |
| 7 | `07_gap.R` | reachability gap by decile, before and after the price cuts | 4.5 |
| 8 | `08_map.R` | map: S4 against S2t in the 15 buurten | 4.4 |
| 9 | `09_index.R` | `README_output.md`: every figure and table with its paper section | |

Figures are written as PNG (300 dpi) and PDF to `figures/`, tables at full
precision to `tables/` and rounded for the paper to `tables_paper/`.

## Rounding

The tables in `tables_paper/` and the numbers in the figures are rounded as
follows:

* at most two significant digits (`display_digits`), three for accessibility
  levels (`level_digits`); correlations and CV to two decimals; shares as
  whole percents (two significant digits below 10%);
* a gain below half a job per person (`gain_floor`) and a share below 0.1%
  (`pct_floor`) are 0: below the resolution of an expected count;
* for M2, never more digits than the data support: the last digit kept is at
  the leading digit of the standard deviation over the runs with perturbed
  inputs (`precision.csv`; JCGM 100:2008, 7.2.6), and the figure captions
  state the largest relative spread. Without that file only the display
  rules apply.
