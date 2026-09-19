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

## Home working

The two Weibull fits differ by whether the JOB admits working from home.
`segments.wfh` derives a rough share per LISA sector: the CBS incidence of
(at least sometimes) working from home by education level (2024) weighted
with the education mix of employee jobs per SBI section (2010, the only
year published), averaged over each sector's sections. It compresses the
range (33% for agriculture and hospitality to 68% for education; 45%
overall against 52% of workers who at least sometimes work from home),
ignores occupation, and reads "at least sometimes" as "admits".

## Not covered yet

Walking (no time margin), public transport (skims, distances, fares),
the shared-bicycle chains and scenarios S1-S4, specifications M1, M1' and
M3 (the copula option covers M3 for priced modes), the interchangeability
ratio and other paper outputs, congestion.
