# Architecture

`ikob2` is a Python package (`src/ikob2`, Python >= 3.11) that turns open
data into accessibility indicators by origin, segment and mode. Only the
explicit `fetch`, download and server steps use the network.

## Layers

    cli/        command-line entry points (python -m ikob2.cli.<name>)
    run/        accessibility per origin/segment/mode, scenarios, R, gap, costs
    segments/   population segments, jobs, margins, budgets, specifications
    envelope/   the reference-budget envelope from its sources
    skims/      travel time, distance and cost matrices; routing servers
    engine/     SegmentedRunner, lazy matrix registry
    core/       numerics: survival families, copulas, Hansen/Shen
    domain/     value objects: filters, segments, zones, model state
    data/       CBS GeoPackage reader, validation
    outputs/    run products for analysis (tables, GeoPackage, diagnostics)
    utils/      DataLayout (data folder structure)
    params.py   parameter resolution; defaults.toml holds the values

Dependencies point downwards: `core` uses `params` (and `domain.state` for
the Shen measure); `engine` uses `core`, `domain` and `data.validation`;
`segments` and `skims` use `core`, `domain` and `params`; `run` uses all of
these; `cli` uses everything.

## Modules

| Package | Module | Purpose |
|---|---|---|
| core | `families` | survival families, hazard, elasticity, implied VOT, TTT, moment matching |
| core | `decay_curves` | numerically safe curve evaluation, atom, epsilon sparsification |
| core | `compose` | survival copulas (independence, Frank, Gumbel-Hougaard, Frechet bounds), `compose_filters` |
| core | `accessibility`, `competition` | Hansen and Shen measures |
| core | `numerics` | dtype, sparse/dense helpers, safe division |
| engine | `runner` | `SegmentedRunner.run_hansen` (rectangular or square), `.run` (Shen, square) |
| engine | `cache` | lazy matrix registry (recipes, pinning) |
| domain | `filter_config`, `segments` | `CurveSpec`, `CopulaSpec`, `ClassFilter` (with parsing of curve and copula blocks); `Segment` and its filter identity |
| domain | `zones`, `state` | `ZoneSet`, immutable `ModelState` (square runs) |
| segments | `pipeline`, `structure`, `ipf`, `marginals`, `kwb`, `config` | household x income segments per buurt |
| segments | `lisa`, `jobs_impute`, `establishments`, `jobs`, `wfh` | jobs by sector, income class and job type |
| segments | `time_margins`, `bridge`, `specs` | Weibull margins, reference budgets, specifications M0-M3 |
| segments | `car_availability`, `ownership`, `pt_spend` | mode availability; PT fare spending by decile (ODiN) |
| segments | `statline` | CBS StatLine snapshots |
| envelope | `sources`, `odin`, `nibud`, `income`, `xm` | the reference-budget envelope from its source tables and ODiN aggregates (`envelope/README.md`) |
| skims | `store` | on-disk memory-mapped skim store |
| skims | `router`, `build`, `walk`, `zones` | r5py/R5 routing, zone points, walking times |
| skims | `car`, `distance`, `osrm`, `valhalla_server` | car time and cost, route distances, detour calibration |
| skims | `peak` | peak-load OSM extract |
| skims | `gtfs_pt`, `pt_build`, `pt_fare` | GTFS frequency-model router, PT layer, fares |
| skims | `hubs`, `hub_siting` | shared-bicycle hub files, siting of extra hubs |
| skims | `otp_server`, `gtfs_subset`, `osm_walk` | OpenTripPlanner server, feed subsetting, walking-network extract |
| run | `accessibility` | `ModeMatrices`, `OptionSet`, `LegOptionSet`, `MixedMode`, `run_accessibility` |
| run | `shared_bike`, `scenarios`, `costs` | chain variants and tariffs, operator usage and S4 calibration, public cost |
| run | `compare`, `interchange`, `gap` | run comparison, ratio R, reachability gap |
| outputs | `export`, `diagnostics` | analysis products of a run, aggregated money-gate curves |
| data | `geopackage`, `validation` | CBS buurt GeoPackage reader, collected validation |
| utils | `paths` | `DataLayout` |

### Command-line entry points

| Module | Subcommands / purpose |
|---|---|
| `cli.layout` | `create`, `link`: data folder |
| `cli.segments` | `fetch`, `run`, `jobs`, `car-availability`, `pt-spend` |
| `cli.envelope` | `aggregates`, `build`: the reference-budget envelope |
| `cli.skims` | `build`, `calibrate-detour`, `build-distance`, `make-peak`, `build-pt`, `inspect` |
| `cli.servers` | `valhalla build/start/stop/status`, `otp prepare/build/start/stop/status` |
| `cli.accessibility` | one accessibility run |
| `cli.hubs` | `propose`: extra hubs for S2 |
| `cli.compare` | two runs: ranks, levels, effectiveness |
| `cli.interchange` | ratio R between two scenarios |
| `cli.paper_tables` | tables over a specification grid |

## Design rules

* **Filter identity.** A segment's weight matrix is determined by
  `(time_cost_id, money_cost_id, ClassFilter)`. Segments with the same
  identity share one composed matrix, and pools multiply matrix-vector
  products, not matrices.
* **One owner of epsilon.** The sparsification threshold is set once per run
  and applied to the composed matrix, never to marginals: since
  `C(u, v) <= min(u, v)` this is equivalent, and marginals stay
  probabilities.
* **Fail loudly.** Curve and copula blocks and parameter files reject unknown keys;
  loaders report the actual columns when a CBS vintage changes; every
  negative value in a CBS numeric attribute is a suppression code and becomes
  NaN, with the codes found reported.
* **Reproducible from inputs.** Only `inputs/` is primary data; everything in
  `intermediate/` and `outputs/` is recomputed by a documented command
  (`pipeline.md`). Runs write `run.json` with their parameters and the path
  and SHA-256 of every input file.
* **Large data stay on disk.** Skims are memory-mapped `.npy`, written in
  resumable blocks; routing servers run as local processes with memory
  limits (a 15 GB machine, no Docker).
* **Costs at run time.** Only distances and leg attributes are stored; fares
  and tariffs are applied when a run is set up, so price assumptions change
  without rerouting.

## Parameters

`src/ikob2/defaults.toml` holds every model and run value (paths, values of
time, specification, car and PT fare models, PT router, bicycle legs,
shared-bicycle tariffs, costs, hub siting, skim limits, segment and job
settings, server ports). A run resolves them in increasing priority from:

    defaults.toml  <  --params FILE (toml/json, partial)  <  --set section.key=value  <  dedicated flags

Unknown keys and wrong types are errors, and `run.json` records the resolved
values under `parameters`. There is no built-in data folder: use
`--data-root`, `$IKOB_DATA_ROOT` or `paths.data_root`. `ikob2.params.DEFAULTS`
supplies the default arguments of library functions, so library and command
line use the same values. Reference files under `paths` (budgets, margins,
price and fare scales) are file names looked up in their folder of the data
layout, unless the value is a path that exists as given; the repository's
`data/` folder only seeds a new data folder (`cli.layout create`).

Every command line (`cli.*`) takes `--params` and `--set`. Not parameters:
CBS table and column identifiers, the LISA sector mappings and numerical
safeguards (clips, tolerances).

## Performance

* Composed filters are stored as CSR only when their density is below
  `numerics.sparse_max_density`; nearly full filters stay dense.
* Leg-wise option sets are evaluated in `numerics.threads` worker threads,
  with BLAS limited to one thread per worker.
* Cost margins are computed once per segment and reused across job types.
* Log statistics are only computed when their level is enabled.
* The Frechet check can be switched off (`numerics.check_frechet`).

## Tests and checks

`pytest` runs about 550 tests in about 25 seconds: unit tests of the
numerical building blocks (survival properties, copula bounds, union of
options, IPF margins, fares against the NS table), hand computations of the
measure (bicycle, M0, PT fare spending), the pipeline on small synthetic
inputs, the scenarios and cost model, and the server clients with mocked HTTP.

* **End to end:** `tests/test_example.py` builds the synthetic data folder of
  `examples/tiny`, runs `cli.accessibility` and compares all 480 rows with
  `examples/tiny/expected/accessibility.csv`.
* **Segment baseline:** `tests/test_segments.py` pins the pipeline's output
  (synthetic, and the 2022 KWB file when `IKOB_KWB_2022_GPKG` points to it).
* **Skips:** a few tests need local data (set `IKOB_DATA_ROOT`) or routing
  inputs, and skip without them.
* **Coverage:** CI runs the tests with `--cov` and fails below 80% of the
  statements of `ikob2` (`[tool.coverage.report]`).
* **Lint:** `ruff check` with pyflakes, the pycodestyle errors and bugbear,
  and docstrings on every public class and function of the API modules
  (`run/`, `cli/`, `segments/specs`, `bridge`, `jobs`, `skims/pt_fare`, `car`;
  `[tool.ruff]` in `pyproject.toml`).
* **Environment:** `requirements-lock.txt` pins every package (Python 3.13;
  Java 21 for R5 and OTP). CI (`.github/workflows/tests.yml`) runs lint and
  tests on it on every push; `.pre-commit-config.yaml` runs ruff before each
  commit and pytest before each push.

The routers are validated against independent engines (`servers.md`); the
scripts of the validation analyses are in `validation/`.
