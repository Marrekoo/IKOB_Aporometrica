# IKOB Aporometrica (`ikob2`)

Reproducible code for a threshold-gate model of accessibility: how many jobs
can a member of each household-type x income-decile segment reach when a trip
is only acceptable if it passes both a **time gate** and a **money gate**,
each a survival function of the traveller's maximum acceptable time and cost.
The case study is Utrecht (111 origin buurten) against all 14,318 Dutch
buurten, by car, bicycle and public transport.

    a[i, s, m] = sum_w sum_j  D[j, class(s), w] * S_T(t_ijm; m, w) * S_M(c_ijm; s)

`S_T` is a Weibull time margin, `S_M` a uniform cost margin from reference
budgets with an atom at zero, `D` imputed LISA jobs by income class and
home-working type. Full statement: [docs/model_theory.md](docs/model_theory.md).

## Documentation (read in this order)

| Document | For |
|---|---|
| [docs/model_theory.md](docs/model_theory.md) | the model, symbol by symbol, assumptions and limits |
| [docs/architecture.md](docs/architecture.md) | package layers, modules, design decisions, tests |
| [docs/data_specification.md](docs/data_specification.md) | every input, intermediate and output file, format and source |
| [docs/pipeline.md](docs/pipeline.md) | the commands, run by run, and the preliminary results |
| [docs/segments.md](docs/segments.md) | household x income segments (GSPREE), budgets, engine bridge |
| [docs/data_lineage.md](docs/data_lineage.md) | jobs: legacy files, LISA imputation, sector-to-income mapping |
| [docs/families.md](docs/families.md) | survival families, diagnostics, time margins |
| [docs/skims.md](docs/skims.md) | travel times, car cost, peak load, PT frequency model and fares |
| [docs/servers.md](docs/servers.md) | Valhalla and OpenTripPlanner, national coverage, PT validation |

## Install and test

    pip install -e ".[test,routing,legacy]"     # Java 21 for R5 and OpenTripPlanner
    pytest                                      # about 460 tests, ~10 s

## Reproduce a run (summary; details in docs/pipeline.md)

    python -m ikob2.cli.layout create --root "<data root>"
    python -m ikob2.cli.segments fetch                 # StatLine snapshots (or use data/statline)
    python -m ikob2.cli.segments jobs ...              # jobs per sector per buurt
    python -m ikob2.cli.skims build ...                # car, bike, walk
    python -m ikob2.cli.skims build-pt ...             # public transport
    python -m ikob2.cli.accessibility --data-root "<data root>" \
        --study utrecht_nl --run my_run --modes car bike pt
    python -m ikob2.cli.compare --data-root "<data root>" run_a run_b

Inputs are open data (CBS, LISA, OpenStreetMap, GTFS, NS 2026 price list);
`docs/data_specification.md` lists sources. Intermediate results and outputs
are recomputed from `inputs/`; each run writes `run.json` with its parameters.

## Status

Implemented: segments, jobs imputation, Weibull/uniform margins, atom,
copulas, car/bike/PT skims with peak load and fares, the shape-comparison
experiment, validation of the PT router against OpenTripPlanner.
Not implemented: shared-bicycle chains and scenarios S1-S4, specifications
M1/M1'/M3 as named in the paper, the interchangeability ratio and the
reachability gap, NDW floating-car congestion. Results in `docs/pipeline.md`
are preliminary.

Licence: see [LICENSE](LICENSE).
