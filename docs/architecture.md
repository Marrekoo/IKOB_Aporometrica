# Architecture

`ikob2` is a Python package (`src/ikob2`, Python >= 3.11) that turns open
data into accessibility indicators by origin, segment and mode. Nothing in
it needs network access except the explicit `fetch` and download steps.

## Layers

    cli/        command line entry points (python -m ikob2.cli.<name>)
    run/        glue: matrices + tables -> accessibility per origin/segment/mode
    segments/   population segments, jobs, margins, budgets (data -> tables)
    skims/      travel time, distance and cost matrices; routing servers
    engine/     SegmentedRunner, matrix registry, scheduler (generic engine)
    core/       pure numerics: decay families, copulas, Hansen/Shen
    domain/     value objects: zones, segments, filters, fares, model state
    data/       readers and validation (GeoPackage, legacy files)
    outputs/    export helpers (tidy tables, GeoPackage)
    utils/      DataLayout (folder structure)

Dependencies point downwards: `core` imports nothing from the rest,
`engine` uses `core` and `domain`, `run` and `segments` use `engine` and
`skims`, `cli` uses everything.

## Modules

| Package | Module | Purpose |
|---|---|---|
| core | `families` | survival families, hazard, elasticity, implied VOT, TTT |
| core | `decay_curves` | numerically safe curve evaluation, epsilon sparsification |
| core | `compose` | survival copulas (independence, Frank, Gumbel-Hougaard, Frechet) |
| core | `accessibility`, `competition` | Hansen and Shen measures |
| engine | `runner` | `SegmentedRunner.run_hansen` (rectangular or square) and Shen |
| engine | `cache`, `scheduler`, `experiment_expander` | lazy matrix registry, parallel runs |
| domain | `segments`, `filter_config` | segment identity `(time_cost_id, money_cost_id, ClassFilter)`, JSON filter schema |
| domain | `fare`, `zones`, `state` | fare model, `ZoneSet`, immutable model state |
| segments | `pipeline`, `structure`, `ipf`, `marginals`, `kwb` | GSPREE population segments |
| segments | `lisa`, `jobs_impute`, `establishments`, `jobs`, `wfh` | jobs by sector, income class and job type |
| segments | `time_margins`, `bridge` | Weibull margins, reference budgets, segments for the engine |
| segments | `statline` | CBS StatLine snapshots |
| skims | `store` | on-disk memory-mapped skim store |
| skims | `router`, `build`, `walk`, `zones` | r5py/R5 routing, zone points, walking times |
| skims | `car`, `distance`, `osrm`, `valhalla_server` | car time and cost, distances, detour calibration |
| skims | `peak` | peak-load OSM extract |
| skims | `gtfs_pt`, `pt_fare`, `pt_build` | GTFS frequency-model router, NS fares, PT layer |
| skims | `otp_server`, `gtfs_subset`, `osm_walk` | OpenTripPlanner server, feed subsetting, walking-network extract |
| run | `accessibility`, `compare` | per-mode accessibility, run comparison |
| utils | `paths` | `DataLayout` |

## Design decisions that matter for the results

* **Filter identity.** A segment's weight matrix is determined by
  `(time_cost_id, money_cost_id, ClassFilter)`; segments with the same
  identity share one composed matrix, and pools multiply matrix-vector
  products, not matrices. Engine deduplication and the additivity
  invariant use the same definition.
* **Single owner of epsilon.** The sparsification threshold is set once per
  run and applied to the composed matrix, never to marginals, because
  `C(u,v) <= min(u,v)` makes this equivalent and marginals must remain
  probabilities.
* **Fail loudly.** Filter configs reject unknown keys; loaders report the
  actual columns when a CBS vintage changes; CBS suppression sentinels
  (-99999999) become NaN with a count.
* **Reproducible from inputs.** Only `inputs/` is primary data; everything in
  `intermediate/` and `outputs/` is recomputed by a documented command
  (`pipeline.md`). Runs write `run.json` with their parameters.
* **Large data stay on disk.** Skims are memory-mapped `.npy`, blocks are
  written resumably; routing servers run as local processes with memory
  guards, because the development machine has 15 GB and no Docker.

## Parameters

Model and run parameters are not written in the code. `src/ikob2/defaults.toml`
holds every value (value of time, car and PT fare models, PT router, bicycle
leg, shared-bicycle tariffs, skim limits, segment and jobs settings, server
ports, budget and margin file names). A run resolves them, in increasing
priority, from that file, `--params FILE` (a partial toml/json with the same
layout), `--set section.key=value` (repeatable) and the dedicated flags
(`--walk-kmh`, `--theta`, ...). Unknown keys and wrong types are errors, and
`run.json` records the resolved values under `parameters`. There is no
built-in data folder: use `--data-root`, `$IKOB_DATA_ROOT` or `paths.data_root`.
`ikob2.params.DEFAULTS` supplies the default arguments of library functions,
so library and command line cannot disagree.

Not parameters: CBS table and column identifiers, the LISA sector mappings,
numerical safeguards (clips, tolerances) and the legacy `cli.run` /
`cli.batch_run` workflow.

## Speed

Four costs dominated a run with the shared-bicycle variants (175 s to 66 s):
statistics for a log line computed even when logging was off, CSR copies of
filters that are nearly full (`numerics.sparse_max_density`), the Frechet
check on copulas that satisfy the bounds by construction, and the cost
margins computed once per job type instead of once.

## Tests

`pytest` (about 460 tests, 10 s): unit tests of every numerical building
block (survival properties, copula bounds, IPF margins, fares against the NS
table), of the pipeline with small synthetic inputs, and of the server
clients with mocked HTTP. A regression baseline for the population
segments is pinned in `tests/test_segments.py` (synthetic and KWB 2022). Routers are validated against independent
engines (`servers.md`), not by unit tests alone.

## Running

    pip install -e ".[test,routing,legacy]"     # Java 21 needed for R5 and OTP
    pytest
    # end-to-end sequence: see pipeline.md

The `cli.run`, `cli.batch_run`, `cli.create_project` and `cli.validate`
modules belong to the earlier single-population IKOB workflow (Hansen and
Shen on project files); the paper's results use `cli.segments`,
`cli.skims`, `cli.accessibility`, `cli.compare`.
