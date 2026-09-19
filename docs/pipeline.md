# From data to accessibility: the pipeline

    python -m ikob2.cli.segments fetch                # StatLine snapshots, once
    python -m ikob2.cli.segments jobs --kwb ... --lisa ... --legacy ... \
        --education ... --out output/sector_jobs_2022.csv
    python -m ikob2.cli.skims build --kwb ... --study GM0344 --osm ... \
        --modes car bike walk --out data/skims/utrecht_nl
    python -m ikob2.cli.accessibility --kwb ... --skims data/skims/utrecht_nl \
        --sector-jobs output/sector_jobs_2022.csv --out output/run_s0 \
        --modes car bike

| Piece | Module | Source |
|---|---|---|
| Segment populations (44 per buurt) | `segments.pipeline` | KWB 2022, CBS 86161NED, 71487ned |
| Jobs by LISA sector per buurt | `segments.jobs_impute` | LISA 2025 (year 2022), KWB establishments, legacy totals |
| Job pools by income decile | `segments.jobs` | sector wages (CBS 81431NED) |
| Job type: admits home working | `segments.wfh` | CBS 85718NED, 82072NED (rough) |
| Time margin (Weibull) | `segments.time_margins` | `data/margins/S_T_work.csv` |
| Cost margin (uniform + atom) | `segments.bridge` | `data/envelope/reference_budgets.csv` (Table 6) |
| Skims | `skims` | OSM via r5py; walking from zone size |
| Car time and cost | `skims.car` | legacy IKOB rates, crow-fly distance x detour |
| Glue | `run.accessibility`, `cli.accessibility` | |

## Data folder

`/home/marco/IKOB data/` is laid out by `ikob2.utils.paths.DataLayout`
(`python -m ikob2.cli.layout create`): `inputs/` (kwb, lisa, osm, gtfs,
legacy_ikob, survey, odin; read only), `cache/statline/` (StatLine
snapshots), `intermediate/` (segments, jobs, `skims/<study>`,
calibration) and `outputs/runs/<run>/`. Files that were already in the
root stay there; `inputs/` links to them. With `--data-root` the run
command fills its paths from the layout:

    python -m ikob2.cli.accessibility --data-root "/home/marco/IKOB data" \
        --study utrecht_nl --run s0_prelim --modes car bike

## What is computed

For every origin buurt, segment (household type x income decile) and mode:

    a = sum over job types w in {no home working, home working possible},
        sum over destinations j of
          D[j, income decile, w] * S_T(t_ij; mode, w) * S_M(c_ij; segment)

(specification M2: independent gates). `S_T` is the mode's Weibull, `S_M`
the segment's uniform cost margin with its atom at zero; free modes (bike)
have no cost margin. The two job types are computed separately and added
(the measure is linear in the opportunities). The result table also holds
the atom and the accessibility divided by `1 - atom` (undefined where the
atom is 1). Unreachable pairs get a very large time, hence weight 0.

## Car

Legacy IKOB (`ikob/utils.compute_car_gtt`, defaults in
`configuration_definition.py`): drive time + parking search time
(arrival 12/8/4/0/0 minutes by urbanisation class 1..5, departure a
quarter) + value of time x (variable cost + road charge) x distance,
variable cost 16 ct/km (fossil) or 5 ct/km (electric). Households
without a car use a shared car (0.33 EUR/km + 0.05 EUR/min) or a taxi
(2.40 EUR/km + 0.40 EUR/min). The gate needs time and money separately,
so `skims.car.car_time_and_cost` returns both. Two differences: the
parking search is added as origin departure plus destination arrival (the
legacy formula pairs origin arrival with destination departure), and the
distance comes from the crow-fly distance between centroids times a
detour factor, since r5py's matrices carry no distance
(`skims.car.DetourModel`, calibrated on routed pairs; default 1.3).
Parking costs per zone and road charges are supported but no data is
wired in.

### Car distance calibration (OSRM)

`python -m ikob2.cli.skims calibrate-detour` routes a sample of origin x
destination pairs with the OSRM table service (public demo server, a
handful of requests; self-host OSRM for more) and fits the median
route/crow-fly ratio per distance band. First calibration (Utrecht
origins, 10,840 pairs): detour 2.05 below 1.5 km, 1.6 at 2 to 6 km, 1.5
at 11 km, 1.33 at 37 km and 1.23 at 90 km and beyond, so short urban
trips are far more circuitous than the placeholder 1.3.
Result: `data/calibration/car_detour.json` (also in the data folder).

## Home working

The two Weibull fits differ by whether the JOB admits working from home.
`segments.wfh` derives a rough share per LISA sector: the CBS incidence of
(at least sometimes) working from home by education level (2024) weighted
with the education mix of employee jobs per SBI section (2010, the only
year published), averaged over each sector's sections. It compresses the
range (33% for agriculture and hospitality to 68% for education; 45%
overall against 52% of workers who at least sometimes work from home),
ignores occupation, and reads "at least sometimes" as "admits".

## First national run (preliminary)

`s0_prelim`: 111 Utrecht origins, all 14,318 buurten as destinations,
car and bike, independent gates, one-way budgets, first decile censored
(atom 1). Population-weighted mean acceptable jobs by car about 200,000
for couples and about 146,000 for single households (single households
sit in lower budget deciles); by bike 21,000 to 28,000. The censored
first decile reaches almost nothing by car (only free intrazonal
trips). It runs in 12 seconds once the skims exist.

Read as a smoke test of the chain, not as results: the skim is free-flow,
car distances come from a crow-fly detour model, the sector jobs are
imputed, walking and public transport are missing, and the time margin
is a survey fit not yet checked against these skims.

## Impedance-shape comparison (El-Geneidy-style test)

Two time-only runs (no cost margin, no home-working split), car and bike,
same skims and jobs, one common time margin:

    ... --run elgeneidy_step45 --time-shape step --cutoff 45 --no-cost-gate
    ... --run elgeneidy_exp45  --time-shape exponential --cutoff 45 --no-cost-gate
    python -m ikob2.cli.compare elgeneidy_step45 elgeneidy_exp45

`step` is the hard 45-minute cut-off; `exponential` is calibrated to it
with the same mean acceptable time (rate 1/45, the moment matching of the
paper; `--exp-calibration half` instead gives 50% acceptance at 45
minutes). Preliminary result (111 Utrecht origins, 40 segments):

| | car | bike |
|---|---|---|
| Spearman, all origin x segment cells | 0.94 | 0.99 |
| Spearman, origin means | 0.90 | 0.92 |
| overlap of the top 10% of origins | 64% | 64% |
| mean level, exponential / cut-off | 1.40 | 0.74 |
| ratio by income decile (range) | 1.17 to 1.73 | 0.72 to 0.86 |

The ranking is largely the same, while levels differ by 26% to 40% and
the ratio differs across income deciles: the exponential's long tail
credits far-away jobs, and deciles whose job pool lies further away gain
more (car decile 8: 1.73, decile 10: 1.17). Agreement on rankings does not
mean agreement on the distribution across groups.

## Not covered yet

Walking (no time margin), public transport (skims, distances, fares),
the shared-bicycle chains and scenarios S1-S4, specifications M1, M1' and
M3 (the copula option covers M3 for priced modes), the interchangeability
ratio and other paper outputs, congestion.
