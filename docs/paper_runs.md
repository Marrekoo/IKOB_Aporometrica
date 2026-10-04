# Paper runs: set-up

The runs behind the paper "Either you can reach it or you cannot: threshold
microfoundations for accessibility" (Utrecht shared bicycles). The plan is
`paper/runs.toml`; every run records its parameters, its input files
(path and SHA-256) and the code version (`code` in `run.json`).

## Study area and inputs

| Element | Setting | Source |
|---|---|---|
| Origins | the 111 buurten of the municipality of Utrecht (skim store `utrecht_nl`, `paths.study`) | CBS Kerncijfers wijken en buurten 2022 |
| Destinations | every Dutch buurt with jobs (14,318) | idem |
| Jobs | LISA municipal jobs 2022 imputed onto buurten (`cli.segments jobs`); matched to income deciles and job types through sector x occupation cells (`data/occupations`) | `data_lineage.md` |
| Segments | 4 household types x 10 income deciles per buurt | `segments.md` |
| Time margins | Weibull per mode and job type (`data/margins/S_T_work.csv`) | `families.md` |
| Cost margins | reference budgets per priced one-way journey, 2022 euros (`data/envelope/reference_budgets.csv`; `[envelope]`) | `envelope/README.md` |
| PT | GTFS NL, Tuesday 15 September 2026, 07:00-09:00; NS 2026 fares | `skims.md` |
| Shared bicycles | OV-fiets (EUR 4.80), the 27 municipal Lime hubs, dockless Lime access everywhere; Lime tiers EUR 3 / 4 / 5 up to 20 / 30 / 40 rental minutes; bicycle legs at most 20 minutes plus 1 fixed minute | `scenarios.md` |
| Bicycle ownership | per buurt, Utrecht buurtteam survey 2025 | `inputs/veh_owners` |
| Car availability | ODiN 2022-23 per household type and decile | `cli.segments car-availability` |

All other parameters are those of `src/ikob2/defaults.toml` at the commit
in `run.json`.

## Specifications (`[specs]` of the plan)

| Tag | Specification |
|---|---|
| m0 | cut-off on generalised time at the median acceptable time (one value of time per mode) |
| m0u | dual cut-off: median acceptable time, one cost cut-off for everyone (median of the population's cost thresholds) |
| m0s | dual cut-off: median acceptable time, the cost cut-off of each segment (median of its cost margin) |
| m1 | exponential generalised cost, national values of time |
| m1p | exponential margins: time with the mean acceptable time, cost with the segment's mean budget |
| m2 | Weibull time margin and uniform cost margin with an atom, independent |
| m3t1.25 ... m3tinf | M2 margins joined by a Gumbel-Hougaard copula, theta 1.25, 1.5, 2, 4, infinity |

## Scenarios (`[scenarios]` of the plan)

| Name | Intervention | Set-up |
|---|---|---|
| s0 | baseline | Lime tiers, 27 municipal hubs |
| s1 | Lime prices halved for everyone | `lime_price_scales_all_50.csv` |
| s1a | Lime prices halved for deciles D2-D4 (by income, citywide) | `lime_price_scales_d2_d4_50.csv` |
| s2c | 27 extra Lime hubs, citywide | skim modes `pt_wb_lime_s2c`, `pt_bb_lime_s2c` |
| s2t | 12 extra Lime hubs in the 15 target buurten | skim modes `pt_wb_lime_s2t`, `pt_bb_lime_s2t` |
| s3c, s3t | s1 with s2c, s1 with s2t | both settings |
| s4 | Lime prices halved for the residents of the 15 target buurten (by address) | `lime_price_scales_all_50.csv` with `lime_price_zones_overvecht_kanaleneiland.csv` |
| s4a | Lime prices halved for D2-D4 residents of the 15 target buurten | `lime_price_scales_d2_d4_50.csv` with the zone |

The 15 target buurten are the 10 buurten of Overvecht and
Kanaleneiland-Noord, Kanaleneiland-Zuid, Bedrijvengebied Kanaleneiland,
Transwijk-Noord and Transwijk-Zuid.

## Run sets (`[[sets]]` of the plan)

| Runs | Specifications x scenarios | Arguments | Number |
|---|---|---|---|
| `sp_<spec>_<scenario>` | all x all | PT chains with shared bicycles (`--modes pt --shared-bike v2 --ownership`), jobs matched to income | 99 |
| `spc_<spec>_<scenario>` | all x s0, s1, s1a, s2c, s2t, s4, s4a | as `sp`, `--common-jobs` (the controlled comparison) | 77 |
| `spt_<spec>_<scenario>` | m0, m1, m1p, m2 x s0, s2c, s2t | as `sp`, `--no-cost-gate` (time only, for the reachability gap) | 12 |
| `scen_<scenario>` | m2 x all | car, bicycle and PT, `--report-usage` (Lime usage and public cost), the analysis products | 9 |
| `scen_<scenario>_nobike` | m2 x s0, s1, s1a, s2c, s2t, s4, s4a | PT chains with bicycle ownership 0 everywhere (residents without a private bicycle) | 7 |

204 runs in all.

## Preparation

1. Data folder: `python -m ikob2.cli.layout --root <root> create` seeds the
   reference files (budgets, margins, tariff and zone tables, occupation
   tables, StatLine snapshots).
2. Skims (`skims.md`): car, bicycle and walking, and the PT modes `pt`,
   `pt_wb`, `pt_bw`, `pt_bb`, `pt_wb_<kind>`, `pt_bb_<kind>` for the hub kinds
   `lime` and `ovfiets`.
3. Extra hubs, sited on the population-weighted PT accessibility of the M2
   baseline (`sp_m2_s0`; `scenarios.md`):

        python -m ikob2.cli.hubs propose --data-root <root> --label s2c \
            --base-accessibility <root>/outputs/runs/sp_m2_s0/accessibility.csv
        python -m ikob2.cli.hubs propose --data-root <root> --label s2t \
            --within lime_price_zones_overvecht_kanaleneiland.csv \
            --base-accessibility <root>/outputs/runs/sp_m2_s0/accessibility.csv

   and their skim modes (`build-pt` with `--hub-file
   hubs/utrecht_hubs.csv:lime --hub-file hubs/utrecht_hubs_<label>.csv:lime
   --hub-kind lime --mode-name pt_wb_lime_<label>`, and `--access bike` for
   `pt_bb_lime_<label>`).

## Running and tables

    python -m ikob2.cli.batch --plan paper/runs.toml --data-root <root> --jobs 2
    python -m ikob2.cli.paper_tables --data-root <root>

`cli.batch` writes the commands to `outputs/logs/runs_commands.sh` and a log
per run to `outputs/logs/runs/`; `--skip-existing` resumes. `cli.paper_tables`
takes its specifications, scenarios, pairs and targeting runs from `[paper]`
of the parameters and writes `outputs/comparisons/specs/` and
`outputs/comparisons/targeting/`. The figures and the remaining tables are
made from these outputs by the R analysis pipeline.

## Precision

    python -m ikob2.cli.batch --plan paper/perturbation.toml --data-root <root> --jobs 2
    python -m ikob2.cli.precision --data-root <root>

140 runs (20 draws x 7 scenarios, M2) with the jobs and population drawn within
the rounding of their sources; the result is the number of significant digits
per statistic (`docs/scenarios.md`, precision).
