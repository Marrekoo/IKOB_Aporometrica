# IKOB Aporometrica (`ikob2`)

A threshold-gate model of job accessibility. A trip is acceptable to a
person only if it passes two gates: its travel time is below the person's
maximum acceptable time, and its out-of-pocket cost is below the person's
maximum acceptable cost. Both thresholds vary across the population, so each
gate is a survival function, and the accessibility of a population segment
is the expected number of jobs that pass both:

    a[i, s, m] = sum_w sum_j  D[j, c(s), w] * f_s( t_ijm, c_ijm ; m, w )

    f_s(t, c) = C( S_T(t; m, w), S_M(c; s) )          (C = product by default)

* `i` origin buurt, `j` destination buurt, `m` mode, `s` segment
  (household type x income decile), `w` job type (admits working from home
  or not);
* `D` jobs matched to the segment's income class, `t` door-to-door minutes,
  `c` euro per one-way journey;
* `S_T` a Weibull time margin per mode and job type, `S_M` a uniform cost
  margin per segment with an atom at zero, `C` a survival copula.

The case study is the city of Utrecht (111 origin buurten) against all
Dutch buurten, by car, bicycle and public transport, with shared-bicycle
chains as alternative public-transport journeys and a set of pricing and
hub scenarios.

## Documentation

| Document | Contents |
|---|---|
| [docs/model_theory.md](docs/model_theory.md) | the model symbol by symbol, the specifications M0-M3, alternative journeys, assumptions |
| [docs/architecture.md](docs/architecture.md) | package layers and modules, parameters, design rules, tests |
| [docs/pipeline.md](docs/pipeline.md) | the commands from raw data to accessibility tables, in order |
| [docs/scenarios.md](docs/scenarios.md) | shared-bicycle variants, scenarios S0-S4, costs, R, the reachability gap, paper tables |
| [docs/paper_runs.md](docs/paper_runs.md) | the set-up of the paper runs (`paper/runs.toml`, `cli.batch`) |
| [docs/data_specification.md](docs/data_specification.md) | every input, intermediate and output file |
| [docs/segments.md](docs/segments.md) | household x income segments, reference budgets, the engine bridge |
| [docs/data_lineage.md](docs/data_lineage.md) | jobs: sources, imputation onto buurten, income and home-working split |
| [docs/families.md](docs/families.md) | survival families, hazard diagnostics, time margins |
| [docs/skims.md](docs/skims.md) | skim store, car/bike/walk/PT times, car cost, peak load, PT fares, bicycle legs |
| [docs/servers.md](docs/servers.md) | local Valhalla and OpenTripPlanner servers, validation of the PT router |
| [CLAUDE.md](CLAUDE.md) | working rules for contributors and AI assistants: tests, data licences, fixed modelling decisions |

## Install and test

    python -m venv .venv && . .venv/bin/activate
    pip install -r requirements-lock.txt      # the exact tested environment (Python 3.13)
    pip install -e . --no-deps
    pytest                                    # ~550 tests, ~25 s
    ruff check src tests examples validation

`requirements-lock.txt` pins every package; R5 and OpenTripPlanner also need
Java 21. `pip install -e ".[test,routing,excel,dev]"` installs from the
version ranges of `pyproject.toml` instead. CI runs the lint and the tests on
the locked environment on every push (`.github/workflows/tests.yml`);
`pre-commit install --hook-type pre-commit --hook-type pre-push` runs them
locally before commits and pushes.

## Example

    python examples/tiny/make_data.py examples/tiny/data
    python -m ikob2.cli.accessibility --data-root examples/tiny/data \
        --study tiny --run example --modes car bike pt

A complete run on a small synthetic data folder in a few seconds; the stored
output is in `examples/tiny/expected/` and is checked by the tests
(`examples/tiny/README.md`).

## Run (outline; full commands in docs/pipeline.md)

    python -m ikob2.cli.layout --root "<root>" create   # folders + reference files
    python -m ikob2.cli.segments --data-root "<root>" fetch
    python -m ikob2.cli.segments --data-root "<root>" jobs   # jobs per sector per buurt
    python -m ikob2.cli.segments --data-root "<root>" car-availability
    python -m ikob2.cli.skims build ...              # car, bike, walk
    python -m ikob2.cli.skims build-distance ...     # car route distances
    python -m ikob2.cli.skims build-pt ...           # public transport (+ bicycle legs)
    python -m ikob2.cli.accessibility --data-root "<root>" \
        --study utrecht_nl --run s0 --modes car bike pt --ownership
    python -m ikob2.cli.compare --data-root "<root>" s0 s1

Inputs are open data (CBS, LISA, OpenStreetMap, GTFS, the NS price list,
ODiN); `docs/data_specification.md` lists them. Everything under
`intermediate/` and `outputs/` is recomputed from `inputs/`; every run writes
`run.json` with its resolved parameters.

## Parameters

Every model and run parameter is in `src/ikob2/defaults.toml`. Override with
`--params my.toml`, `--set section.key=value` or a dedicated flag; set the
data folder with `--data-root` or `$IKOB_DATA_ROOT`. See
[docs/architecture.md](docs/architecture.md#parameters).

## Scope

Modelled: 44 household x income segments per buurt; LISA jobs imputed onto
buurten and matched to income deciles; Weibull time margins by mode and job
type; uniform cost margins with an atom; copula dependence; car, bicycle,
walking and GTFS public-transport skims with peak load, car cost and NS
fares; car and bicycle availability; shared-bicycle chains (variants
v0-v4); specifications M0, M0u, M0s, M1, M1c, M1', M2, M3; scenarios S0-S4 with their
public cost; the interchangeability ratio R; the reachability gap; paper
tables.

Not modelled: a walking time margin (walking is skimmed but not gated),
measured congestion (peak load is a road-class factor), timetable-based PT
waiting and transfer penalties, competition for jobs in the paper runs
(Shen is available for square runs only), supply limits of shared bicycles.

Cite as in [CITATION.cff](CITATION.cff). Licence: MIT ([LICENSE](LICENSE)). The data have their own licences (see [CLAUDE.md](CLAUDE.md), *Data*).
