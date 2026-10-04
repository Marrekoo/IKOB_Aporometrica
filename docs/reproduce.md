# Reproducing the paper runs

How to recompute the results of "Either you can reach it or you cannot"
(Utrecht shared bicycles) from this repository and its data deposit. The
set-up of the runs is in `paper_runs.md`, every step's details in
`pipeline.md`.

## What comes from where

| Part | Where |
|---|---|
| Software, parameters, run plans, figure scripts | this repository (`src/`, `src/ikob2/defaults.toml`, `paper/runs.toml`, `paper/perturbation.toml`, `paper/figures/`) |
| Reference files (budgets, margins, tariffs, occupation tables, jobs by education, hubs, OV-fiets locations, bicycle ownership, StatLine snapshots, scenario hubs) | this repository, `data/`; `cli.layout create` copies them into a data folder |
| GTFS feed, OpenStreetMap extracts, CBS KWB GeoPackage, LISA municipal table | the data deposit on Zenodo (`inputs.tar`): these versions cannot be downloaded again |
| Skim stores, imputed jobs per buurt and sector | the data deposit (`intermediate.tar`); recomputable from the inputs (`pipeline.md`, steps 3, 5-7) |
| IKOB job table (`inputs/ikob/Alle_Zones_2030_2040.xlsx`) | Stichting CROW, https://github.com/Stichting-CROW/ikob-scripts, `segs/Databronnen SEGS compleet/`, commit f90ac72; only needed to recompute the jobs |
| Car availability and PT fare spending per segment (ODiN aggregates) | the data deposit (`intermediate.tar`, `intermediate/ownership/`), with the respondents behind each cell |
| ODiN microdata | DANS, with permission; only needed to recompute those aggregates (`pipeline.md`, step 4) |
| The runs and paper tables of the paper | the data deposit (`outputs.tar.gz`), to compare with |

Every run writes `run.json` with its resolved parameters, the SHA-256 of its
input files and the code version; `MANIFEST.sha256` of the deposit lists the
same hashes.

## Steps

1. Install the locked environment (`README.md`, *Install and test*) and run
   the tests.
2. Unpack the deposit into a data folder `<root>`, check it and add the
   reference files:

       for a in inputs.tar intermediate.tar outputs.tar.gz; do tar -xf "$a" -C <root>; done
       (cd <root> && sha256sum -c /path/to/MANIFEST.sha256)
       python -m ikob2.cli.layout --root <root> create

   To keep the deposit's runs for comparison, move `<root>/outputs` aside
   before step 4.
3. Car availability per household type and decile (runs with
   `--ownership`) comes with the deposit; to recompute it, `python -m
   ikob2.cli.segments --data-root <root> car-availability` with the ODiN
   microdata in `inputs/odin/`.
4. The runs and tables:

       python -m ikob2.cli.batch --plan paper/runs.toml --data-root <root> --jobs 2
       python -m ikob2.cli.paper_tables --data-root <root>
       python -m ikob2.cli.batch --plan paper/perturbation.toml --data-root <root> --jobs 2
       python -m ikob2.cli.precision --data-root <root>

   On the author's desktop with 2 jobs the 140 runs of `perturbation.toml`
   take about 1.5 hours.
5. Compare with the deposit: `accessibility.csv` of every run is identical
   for the same code version (`code` in `run.json`), and so are the tables
   in `outputs/comparisons/specs`, `targeting` and `precision`.

6. The figures and the rounded tables of the paper:

       IKOB_DATA_ROOT=<root> Rscript paper/figures/run_all.R

   (plain R; `paper/figures/README.md`), written to
   `<root>/outputs/paper_figures/`.

## Recomputing the inputs

From the published sources instead of the deposit: `pipeline.md`, steps
2 (StatLine), 3 (jobs; needs the IKOB job table), 5-7 (skims; need the GTFS
feed and the OpenStreetMap extracts of the deposit for the same travel
times) and `cli.hubs propose` for the scenario hubs (`scenarios.md`; the
deposit's and `data/hubs_scenarios/` are its output).
