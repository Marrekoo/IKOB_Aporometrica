# Model theory

The model that the code computes, symbol by symbol, with the module that
implements each part.

## 1. The measure

An *origin* `i` is a CBS buurt of the study area, a *destination* `j` any
Dutch buurt with jobs, a *mode* `m` car, bicycle or public transport (or a
shared-bicycle variant, section 5), and a *segment* `s` a household type
crossed with an income decile: 4 x 10 = 40 segments carry a budget (the
segment tables also hold an `onbekend` income class, which has no budget and
is not run). The accessibility of segment `s` at `i` by mode `m` is the
expected number of jobs a member of `s` finds acceptable:

    a[i, s, m] = sum_w sum_j  D[j, c(s), w] * f_s(t_ijm, c_ijm ; m, w)

| Symbol | Meaning | Built in |
|---|---|---|
| `D[j, c, w]` | jobs in `j` matched to income class `c`, split by job type `w` (admits working from home or not) | `segments.jobs_impute`, `segments.jobs`, `segments.wfh` |
| `t_ijm` | door-to-door travel time, minutes | `skims` |
| `c_ijm` | out-of-pocket cost of the one-way journey, euro (0 for bicycle and walking) | `skims.car`, `skims.pt_fare`, `run.shared_bike` |
| `S_T(t; m, w)` | survival function of the maximum acceptable travel time (Weibull) | `segments.time_margins` |
| `S_M(c; s)` | survival function of the maximum acceptable cost (uniform with an atom) | `segments.bridge` |
| `f_s` | joint acceptance: `C(S_T, S_M)` for a survival copula `C` | `core.compose`, `segments.specs` |

The measure is Hansen-type, `sum_j D_j f(impedance_ij)`, with two
differences: `f` is the probability that a member of the segment accepts the
trip rather than a fitted distance decay, and time and money are separate
gates rather than one generalised cost. It is linear in `D`, so the two job
types are evaluated separately and added (`run.accessibility.run_accessibility`).

### Availability of modes

`a[i, s, m]` is conditional on having mode `m`. With `--ownership` the run
also reports the availability `p[i, s, m]` and the product
`accessibility_expected = a * p`, the expected number of acceptable jobs of
a random member of the segment by that mode.

* **Car**: share of adults in a household with a car, by household type and
  income decile, from ODiN 2022-23 (person weights). Study-area cells are
  shrunk to the national cell, `p = (n_l p_l + k p_n) / (n_l + k)`, `k = 30`
  (`segments.car_availability`). No spatial variation within the study area.
* **Bicycle**: share of residents with a private bicycle per buurt, from the
  Utrecht buurtteam survey 2025 assigned to buurten (`segments.ownership`);
  equal for all segments of a buurt.
* **Public transport, walking**: `p = 1`.

"No car in the household" is not "cannot travel by car" (lifts, car sharing,
company cars), and the joint availability of car and bicycle is not
modelled.

## 2. Time margin: Weibull

    S_T(t) = exp(-(t / eta)^k)

`eta` is the scale and `k` the shape. `k > 1` means the hazard of giving up
rises with travel time (a soft threshold); `k = 1` is the exponential and
`k -> inf` the hard cut-off. Parameters are per mode and job type, fitted by
interval-censored maximum likelihood to stated maximum commuting times
(`inputs/survey/S_T_work.csv`: `wfh, mode, eta, k, median, class`). All shapes
lie between 2.7 and 3.2. The median is `eta (ln 2)^(1/k)`: for car about 40
minutes for jobs without home working and 49 minutes for jobs that admit it.
Walking has no work margin and is not gated.

The job type belongs to the job, not the traveller: the margin used for a
pair depends on which part of `D` it multiplies.

For impedance-shape experiments `--time-shape exponential|step --cutoff T`
replaces the Weibull margins by one common curve (section 8).

## 3. Cost margin: uniform with an atom

The reference-budget envelope (`inputs/envelope/reference_budgets.csv`) gives
each household type x decile a lower and upper bound `[low, high]` in euro
per journey. The maximum acceptable cost is uniform on that interval:

    S_M(c) = 1                          c <= low
           = (high - c) / (high - low)  low < c < high
           = 0                          c >= high

Decile 1 has no envelope row (censored). It carries an *atom* `pi` of mass at
zero: `S_M(c) = (1 - pi) S_u(c)` for `c > 0` and `S_M(0) = 1`, with `pi = 1`
for decile 1, so no priced journey is acceptable to it. The atom is reported
per row (`atom`), and `accessibility_normalised = accessibility / (1 - atom)`.
Free modes have `S_M = 1`.

The budgets are per ODiN tour, one whole one-way journey for a single purpose
with all its legs; this is what the model's door-to-door cost prices, so the
table is used as it stands (`legs_per_tour = 1`). A larger `legs_per_tour`
divides the bounds as a sensitivity (`segments.bridge.rescale_budgets`). The
bounds are the residual envelope divided by the class's ODiN trips, so they
need not increase with income (see `segments.md`).

## 4. Dependence between the gates

If time and money tolerance are dependent, the joint survival is
`P(tau > t, mu > c) = C(S_T(t), S_M(c))` for a survival copula `C`
(`core.compose`): independence (default), Frank, Gumbel-Hougaard, and the
Frechet bounds (comonotone, countermonotone), all available on the command
line (`--copula`, with `--theta` for Frank and Gumbel-Hougaard). The
Frechet-Hoeffding bounds `max(u+v-1, 0) <= C <= min(u, v)` are checked on
composed matrices of parametric copulas (`numerics.check_frechet`). With `S_M = 1` every
copula reduces to the time margin, so time-only runs use the same code path.

**Reading the dependence parameter.** Stronger positive dependence raises
`C(S_T, S_M)` and shrinks the group that fails the gates: those who tolerate
long trips also tolerate high costs. The excluded group becomes smaller and
more concentrated in the low deciles where the money gate binds. An analysis
that assumes dependence, or a generalised cost (the comonotone extreme),
reports fewer excluded people than one that assumes independence. Results for
the lowest deciles are therefore reported over the dependence range
(countermonotone to comonotone), not at one `theta`.

### Impedance specifications (`segments.specs`, `--spec`)

| Spec | Joint acceptance `f(t, c)` | Notes |
|---|---|---|
| M0 | `1{t + c/VoT <= T*}` | cumulative opportunities in generalised time; `T*` the median acceptable time of the Weibull |
| M1 | `exp(-(t + c/VoT)/beta)` | generalised cost; `beta` the mean acceptable time of the Weibull (per mode and job type), one VoT per mode; no atom |
| M1c | as M1 with one VoT | the VoT set so that the implied mean acceptable cost equals `accessibility.m1c_cost_mean_eur`, or by default the population-weighted median of the segments' mean envelopes |
| M1' (`m1p`) | `exp(-t/beta) exp(-c/mu_s)` | a VoT per segment so that `mu_s = beta VoT_s` equals the segment's mean envelope `(1 - pi)(low + high)/2` |
| M2 | `S_T(t) S_M(c)` | Weibull time margin, uniform cost margin with an atom, independent (default) |
| M3 | `C_GH(S_T, S_M; theta)` | Gumbel-Hougaard copula of the M2 margins, `--theta`; 1 is M2, `inf` the comonotone limit |

M1' and M2 share the first moment of both margins, so the M1' -> M2 contrast
isolates the shape of the thresholds and M2 -> M3 the dependence. M1, M1c
and M1' have no atom.

**Values of time** (EUR per hour, Dutch national value-of-time study,
`[vot]`): car 12.05, rail 15.10, bus/tram/metro 10.80. Public transport is
priced per journey at a rail-share-weighted value,
`VoT_ij = s_ij 15.10 + (1 - s_ij) 10.80`, `s_ij = rail_km / (rail_km + other_km)`;
the cost is rescaled to the rail value, `c' = c 15.10 / VoT_ij`, so that one
exponential cost gate equals the generalised-cost form (`vot_weighted_cost`).
`--vot mode=value` overrides the values.

## 5. Alternative journeys

When a person can choose between journeys for the same pair (plain public
transport, public transport with a bicycle at one or both ends), the pair is
acceptable if *any* option passes both gates. Option `(t, c)` is acceptable
to thresholds `(tau, mu)` iff `tau >= t` and `mu >= c`, so the acceptable
region is a staircase. With the `K` options sorted by time and `cmin_k` the
lowest cost among the options at least as fast as option `k`:

    P(any option acceptable) = sum_k f(t_k, cmin_k) - sum_{k<K} f(t_{k+1}, cmin_k)

exactly, for any joint survival `f`. Options both slower and dearer than
another add nothing. `run.accessibility.OptionSet` and `union_terms`
implement this; `MixedMode` mixes option sets over a population split by
origin (residents with and without a private bicycle). M1, M1c and M1' use
the same formula with their exponential margins (the PT cost rescaled per
option by its rail share); M0 takes the option with the least generalised
time.

**Leg-wise gates** (`LegOptionSet`, variant v3): each leg of a chain is
judged on its own time margin (bicycle legs on the bicycle margin, the PT leg
on the PT margin for the in-PT time) and the cost margin on the journey
total. With independent thresholds the joint survival of an option is the
product of its leg survivals and the cost survival.

The variants and their tariffs are in `scenarios.md`.

## 6. Diagnostics of a margin

For any margin `core.families` computes the hazard `h = -d log S / dz`, the
elasticity `z h(z)`, the implied value of time `h_T / h_M` (constant only for
exponential margins), the total-time-on-test transform, the mean threshold
and moment matching (`families.md`).

For each run, `outputs.diagnostics` aggregates the cost margins of an
origin's segments into the population-weighted mixture `S_bar(c)` and
reports its hazard and TTT transform (`money_gate_*.csv`). A mixture of
increasing-hazard margins has a lower hazard than its components, so the
aggregate can drift toward the decreasing-hazard, power-law look of fitted
gravity decay.

## 7. Competition (Shen)

`core.competition`, `core.accessibility` and `engine.runner.SegmentedRunner.run`
implement the Shen (1998) competition-adjusted measure,
`V_j = sum_k P_k f(c_kj)` and `A_i = sum_j D_j f(c_ij) / V_j`, pooled by income
class; destinations nobody can reach contribute zero. It needs every origin's
population and is square (origins = destinations). The study-area runs are
rectangular and use the Hansen form (`run_hansen`).

## 8. Robustness experiments

* **Impedance shape**: a hard cut-off versus an exponential calibrated to it
  (`--time-shape step|exponential --cutoff 45 --exp-calibration mean|half
  --no-cost-gate`), compared with `cli.compare`: rank correlation, top-decile
  overlap and level ratios by group.
* **Specifications**: the grid M0-M3 over the scenarios, with correlations
  of levels and of gains between specifications (`cli.paper_tables`).
* **Dependence bounds**: `--copula countermonotone` and `--spec m3 --theta inf`.
* **Budget basis**: `--legs-per-tour`.
* **Congestion**: a peak-load skim store (`skims.md`).
* **Controlled comparison**: `--common-jobs` gives every segment all jobs, so
  segments differ only in their margins.
* **PT router against OpenTripPlanner**: `servers.md`.

## 9. Where the inputs come from

| Quantity | Method | Doc |
|---|---|---|
| Segment populations | Poisson structure model on CBS municipal tables, IPF to buurt margins | `segments.md` |
| Jobs by sector per buurt | LISA municipal jobs distributed over buurten (log-linear model, KWB establishments, IPF) | `data_lineage.md` |
| Jobs by income class | sectors ranked by CBS wages, laid along the income axis; pools partition the jobs | `data_lineage.md` |
| Job type (home working) | share per sector from CBS 85718NED and 82072NED | `data_lineage.md` |
| Travel times | OSM (car, bicycle, walking), GTFS frequency model (PT), road-class peak load | `skims.md` |
| Cost | car: per-km rate x routed distance + parking time; PT: NS 2026 rail table + regional boarding and per-km fare; shared bicycles: operator tariffs | `skims.md`, `scenarios.md` |

## 10. Assumptions and limits

* Independent gates unless a copula is chosen; copula parameters are imposed,
  not estimated.
* One time margin per mode and job type; no variation by income.
* Travel times between buurt centroids; distant destinations can be
  represented by a municipality point (`skims.md`).
* PT waiting follows a headway rule, not the timetable; no transfer penalty.
* Rail fares follow the NS 2026 table between tariff units, linearly;
  bus/tram/metro fares are a regional linear approximation.
* Jobs below the municipal level are imputed (`data_lineage.md`).
* The cost of a journey is gated as a whole; the budget is per journey.
