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

## Peak load and public transport (preliminary)

* **Peak load** (car times with congestion factors by road class, see
  `docs/skims.md`): mean acceptable jobs by car fall to 0.61 of the free-flow
  value (deciles 0.52 to 0.66) while the ranking of origins hardly changes
  (Spearman 0.99, top-10% overlap 91%). The Weibull margin is steep around
  its median (about 40 minutes by car), so a 10 to 20% longer trip removes
  a large share of acceptable jobs.
* **Public transport** from GTFS with the frequency model (waiting
  `min(headway/2, 7.5)`, no transfer penalty, walking 4 km/h): 45,886
  stops and 72,664 line-stops for the September weekday, computed for
  111 origins x 14,318 destinations in under two minutes; 76% of the
  pairs are reachable within 180 minutes. Example from Leidsche Rijn:
  Amsterdam 62 minutes, Rotterdam 79, Arnhem 73, Groningen 158.
  Time-only run of all three modes (`s0_modes_timeonly`, population-
  weighted mean acceptable jobs): car 215,000, public transport 52,000,
  bike 25,000.
* **PT fares** (rail: the official NS 2026 price list; bus/tram/metro: 1.08 +
  0.18 per km, one boarding charge) and the cost gate:
  `s0_modes_gated_ns2026` (car, bike, public transport, all cost-gated):
  mean acceptable jobs car 186,000 (0.87 of time-only), public transport
  46,000 (0.89), bike unchanged. The earlier power-law fares gave nearly the
  same result (45,900). For public transport the gate removes decile 1
  (atom) and about a fifth of decile 2 (budget 5 to 24 EUR against median
  fares of 20 EUR); deciles 4 to 10 are hardly affected.

## Impedance specifications and availability (preliminary)

`--spec m1|m1p|m2|m3` with `--ownership` (runs `all_*`; population-weighted
mean acceptable jobs by mode, conditional on having the mode):

| Spec | Car | Public transport | Bike |
|---|---|---|---|
| M1 (VoT 12.05 car; PT rail-weighted 15.10 / 10.80) | 96,700 | 41,300 | 21,500 |
| M1' | 183,500 | 86,300 | 21,500 |
| M2 | 185,600 | 46,200 | 25,000 |
| M3, theta 1.25 to infinity | 186,600 to 188,000 | 46,400 to 46,700 | 25,000 |

M1 is lower than M2 for car because of the cost gate: its single mean
acceptable cost (EUR 8) is far below the envelope means of deciles 3 to 10,
so it removes 55-65% of accessibility in every decile (M2's gate barely
binds there), and it has no atom, so decile 1 keeps 82,000 jobs by car
against 159 under M2. The time shape alone raises car (+8%) and public
transport (x2.1) because the exponential has a fat tail, and lowers bike
(-14%). With PT priced at a single VoT of 15.10 (rail) M1 gives 43,900; at
10.80 (bus/tram/metro) 33,200; weighting by the rail share of kilometres
gives 41,300. Dependence (M3) changes accessibility by at most 1.3%.
Ownership (car by household type and income from ODiN, bicycle by buurt)
lowers expected accessibility by about 25% for car and 10% for bike.

### Dependence bounds and budget basis (checks on M3)

`--copula countermonotone` (the Frechet lower bound) and `--legs-per-tour 2`
(per-trip budgets halved), mean acceptable jobs, conditional:

| | Car | Public transport |
|---|---|---|
| Countermonotone (lower bound) | 181,700 | 45,000 |
| M2 | 185,600 | 46,200 |
| M3, theta = infinity (upper bound) | 188,000 | 46,700 |
| Countermonotone, budgets / 2 | 174,100 | 42,900 |
| M2, budgets / 2 | 180,800 | 45,100 |
| M3, theta = infinity, budgets / 2 | 186,100 | 46,300 |

Only deciles 2 and 3 move (D2 car, one-way budgets: 125,200 to 205,800
around M2's 173,500, i.e. 0.72 to 1.19; with budgets halved: 76,400 to
178,600 around 122,300, 0.62 to 1.46, and D3, D4 also move). Deciles 5 and
above are unaffected in both cases. The bounds bracket the population mean
by 3.5% (car) and 3.8% (PT) at one-way budgets and 7% and 8% at halved
budgets.

## Shared-bicycle chains: variants v0, v1, v2 (preliminary)

`--shared-bike v0 v1 v2` (run `shared_v012`, M2, cost-gated, conditional
accessibility, mean acceptable jobs over all segments): plain PT 46,200;
v0 (own bicycle to the stop for owners) 71,800 (x1.55); v1 (OV-fiets at the
destination rail station, everyone) 66,800 (x1.45); v2 (bicycle at both ends,
by ownership) 105,900 (x2.29). Car is 185,600 and bicycle alone 25,000.
Accessibility is monotone across variants in every cell (v2 >= v1, v0 >=
plain). The gain is spread over deciles 2 to 10 (x2.1 in D2 to x2.9 in D8
for v2; D2 gains least under v1, x1.31 against x1.4-1.6, because the OV-fiets
charge is large against its budget) and household types. The
shared-bicycle contribution over the own-bicycle baseline (v2/v0) is
x1.34-1.65 and is largest where the private-bicycle share is low (correlation
-0.74 with the buurt share). Plain PT understates PT accessibility for
bicycle owners: scenarios S1-S4 should be measured against v0, not against
plain PT. Assumptions to check: hubs are all rail stops, unlimited supply, a
1-minute fixed time, dockless available at every origin, no stock-outs.

### Dockless tariff baseline

Dockless access (and the door-to-door variant v4) is priced by the Lime tiers
(`shared_bike.dockless_model = "lime_tiers"`: EUR 3 / 4 / 5 up to 20 / 30 / 40
minutes of rental, ride plus fixed minutes), the operator active in the
province of Utrecht from January 2026. `unlock_per_minute` (unlock fee plus a
rate per riding minute) remains available. With the Lime tiers instead of
EUR 1.00 + 0.20 per minute the results hardly change (v2 -0.002%, v3 +0.004%):
for the 10-20 minute access rides of the case the two structures cost about
the same.

### Scenarios S1 and S4, concession prices, effectiveness

* **S1**: every Lime tier price halved, `--lime-scale 0.5` (dockless access and
  Lime hubs; OV-fiets is not changed).
* **Price table by segment**: `paths.lime_price_scales` /
  `--price-scales FILE`, rows `household_type,income_class,scale` (`*` = all,
  later rows override earlier ones), for concessions such as the U-pas. It
  scales the Lime part of the journey cost per segment, in the baseline and in
  every scenario. `data/tariffs/lime_price_scales.csv` (all 1) is the default;
  `lime_price_scales_targeted_example.csv` halves D2 to D4.
* **S4**: `--dockless-model flat` replaces the tiers by one price per rental.
  Without `--flat-eur` it is calibrated for revenue neutrality
  (`--flat-method fixed_point`, the default, or `weighted_mean`). Model: a
  person uses the fastest option they find acceptable; volume is the number
  of acceptable pairs (jobs, population) with a Lime rental in the chosen
  option, and the closed-form choice probability is
  `S_T(t_k)[S_M(c_k) - S_M(max(c_k, cmin_(k-1)))]` (checked against a Monte
  Carlo simulation). Fixed point: the price for which price x rentals at that
  price equals the baseline revenue. Both calibrations ran on `pt_v2`
  (independent thresholds, cost gate). The calibrated price is in `run.json`
  under `scenario`.
* **Effectiveness**: `--report-usage` stores the Lime revenue and rentals per
  segment; `cli.compare run_a run_b --mode pt_v2` then writes who gains against
  what it costs (`effectiveness_income_class.csv`, `..._household_type.csv`):
  gain and cost shares by group and their ratio.

Results (`pt_v2`, mean over the case; preliminary): S1 blanket +13.05 million
job-persons (+0.037%) at a revenue loss of 49.6%; halving only D2 to D4 gives
the same gain (+13.02 million) at a loss of 11.4%, 4.4 times more gain per euro
foregone. Under the blanket cut D10 and D9 carry 30% and 19% of the cost and
gain nothing; D2 gets 73% of the gain. S4 (flat EUR 3.011, both calibrations)
changes nothing (+0.0003%): bicycle legs are capped at 20 minutes, so almost
every rental is in the EUR 3 tier and the flat price equals the tier price.

## Not covered yet

Walking (no time margin), a validated rail tariff (the fare shape between tariff units is linear),
the shared-bicycle chains and scenarios S1-S4, specifications M1, M1' and
M3 (the copula option covers M3 for priced modes), the interchangeability
ratio and other paper outputs, congestion.
