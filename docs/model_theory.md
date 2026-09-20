# Model theory

This document states the model that the code computes, symbol by symbol,
and says which module implements each part. Numbers quoted here come from
the runs described in `pipeline.md`; nothing is tuned to them.

## 1. The measure

An *origin* `i` is a neighbourhood (CBS buurt), a *destination* `j` is any
buurt with jobs, a *mode* `m` is car, bicycle or public transport, and a
*segment* `s` is a household type crossed with an income decile (4 x 10 = 40
segments in the runs; 44 in the segment tables, see `segments.md`). The
accessibility of segment `s` at `i` by mode `m` is the number of jobs a
member of `s` finds acceptable:

    a[i, s, m] = sum_w sum_j  D[j, c(s), w] * S_T(t_ijm ; m, w) * S_M(c_ijm ; s)

* `D[j, c, w]` - jobs in `j` matched to income class `c` and job type `w`
  (`w` = admits working from home or not). `c(s)` is the income decile of
  the segment. Built in `segments.jobs_impute`, `segments.jobs`,
  `segments.wfh` (section 5).
* `t_ijm` - door-to-door travel time in minutes; `c_ijm` - out-of-pocket
  cost per trip in euro (zero for the bicycle). Built in `skims` (section 6).
* `S_T` - survival function of the maximum acceptable travel time,
  a Weibull (section 2). `S_M` - survival function of the maximum acceptable
  trip cost, a uniform with an atom at zero (section 3).
* The product `S_T * S_M` is the *independent-gates* specification (called M2
  in the paper). A dependence between the two thresholds is available through
  a survival copula (section 4) and is off by default.
* The measure is linear in `D`, so the two job types are computed separately
  and added (`run.accessibility.run_accessibility`).

It is a Hansen-type measure, `sum_j D_j f(impedance_ij)`, in which `f` is not
a fitted gravity decay but the probability that a member of the segment
accepts the trip, and in which time and money are separate gates rather than
one generalised cost. The paper argues why (a fixed value of time makes the
gates interchangeable; see the "fixed-VOT trap" diagnostics in `families.md`).

### Availability of modes

`a[i, s, m]` is the accessibility of someone *who has* mode `m`. The share
who have it is an availability `p[i, s, m]`, and the run reports both
(`availability`) and their product (`accessibility_expected`), the expected
number of acceptable jobs of a random member of the segment by that mode:

* **Car**: share of adults in a household with a car, by household type and
  income decile, from ODiN 2022-23 (weighted). Utrecht cells are shrunk
  towards the national cell, `p = (n_l p_l + k p_n)/(n_l + k)`, k = 30
  (`segments.car_availability`). No spatial variation within the study area.
* **Bicycle**: share of residents with a private bicycle per buurt, from the
  Utrecht buurtteam survey (2025), assigned to buurten by hand
  (`segments.ownership`); the same for every segment of a buurt.
* **Public transport, walking**: available to everyone (p = 1).

Limits: "no car in the household" is not "cannot go by car" (lifts, car
sharing, company cars); ODiN records the household, not who may use the car;
the bicycle share is per residents, not per segment; the chance of having
both a car and a bicycle is not modelled (modes are not combined yet).

## 2. Time margin: Weibull

`S_T(t) = exp(-(t / eta)^k)`, `eta` the scale, `k` the shape (`k > 1`: the
hazard of giving up rises with travel time, i.e. a soft threshold; `k = 1`
is the exponential; `k -> infinity` is the hard cut-off). Parameters are per
mode and job type (`data/margins/S_T_work.csv`: columns `wfh, mode, eta, k,
median, class`), fitted to survey data on stated maximum commuting time.
The median is `eta (ln 2)^(1/k)`; for cars without home working it is about
40 minutes. Walking has no work margin (very few people in the Netherlands
walk to work) and is not computed. Loader: `segments.time_margins`.

Other shapes (power, Lomax, Tanner/gamma friction, lognormal, log-logistic,
step, triangular, piecewise linear/quadratic) live in `core.families` with
their hazards; `families.md` lists them. The exponential and the hard
cut-off are used in the El-Geneidy-style comparison (section 8).

## 3. Cost margin: uniform with an atom

For each household type and decile the paper's reference-budget envelope
(Table 6, `data/envelope/reference_budgets.csv`) gives a lower and an upper
per-trip bound `[low, high]` in euro (and the equivalent car kilometres).
The bounds already reflect the trips per class in the national travel survey
(ODiN): a decile that makes more trips has a lower per-trip budget, so the
bounds need not increase with income. The maximum acceptable cost is modelled
as uniform on `[low, high]`:

    S_M(c) = 1                        c <= low
           = (high - c)/(high - low)  low < c < high
           = 0                        c >= high

Decile 1 has no envelope row (censored): for that group no priced trip is
acceptable. This is an *atom* `pi` of mass at zero cost, `S_M(c) = (1-pi)
S_u(c)` for `c > 0` and `1` at `c = 0`, with `pi = 1` for decile 1 (the
`atom` column of the output). Free modes have `S_M = 1`. Budgets are per
one-way trip by default; `legs_per_tour` (default 1.0) rescales them because
ODiN tours can consist of several legs (`segments.bridge.rescale_budgets`).
A note in `segments.md` explains why budgets can be non-monotone across
deciles.

## 4. Dependence between the gates (copula)

If tolerance for time and money are dependent, the joint survival is
`P(tau > t, mu > c) = C(S_T(t), S_M(c))` for a survival copula `C`
(`core.compose`): independence (product, default), Frank, Gumbel-Hougaard,
and the Frechet bounds (co- and countermonotone). The Frechet-Hoeffding
sandwich `max(u+v-1,0) <= C <= min(u,v)` is verified on every composed
matrix. With `S_M = 1` every copula reduces to the time filter, so
time-only runs need no separate code path.

### Impedance specifications (`segments.specs`, `--spec`)

| Spec | Gate | Notes |
|---|---|---|
| M1 | `exp(-(t + c/VoT)/beta)` | benchmark: exponential, `beta` = mean acceptable time of the Weibull (per mode and job type), one VoT per mode; no atom |
| M1' | `exp(-t/beta) exp(-c/mu_s)` | as M1 but the VoT of each segment is set so that the mean acceptable cost `mu_s = beta VoT_s` equals the segment's mean envelope `(1-pi)(low+high)/2` |
| M2 | `S_T(t) S_M(c)` | Weibull time margin, uniform cost margin with an atom (default) |
| M3 | Gumbel-Hougaard copula of the M2 margins | `--theta`; 1 is M2, infinity is the comonotone limit |

M1' and M2 share the first moment of both margins, so the M1' to M2 contrast
is the shape of the thresholds and M2 to M3 is dependence.

**M1 values of time** (EUR per hour, national value-of-time study, LMS/NRM):
car driver 12.05, train 15.10, bus/tram/metro 10.80 (bicycle 10.50-11.00 and
walking 12.50-13.00 are not needed: no cost). Public transport is priced per
journey at the rail value for the rail share of its kilometres and the
bus/tram/metro value for the rest, `VoT_ij = s_ij 15.10 + (1 - s_ij) 10.80`
with `s_ij = rail_km / (rail_km + other_km)`. The cost is rescaled to the rail
value, `c' = c 15.10/VoT_ij`, so that the single exponential cost gate equals
the generalised-cost form. Implied mean acceptable cost `beta VoT`: about
EUR 8 by car and EUR 10-11 by public transport, for every segment, against
envelope means of EUR 14-586 (D2 about 20, D10 about 475). The consequence is
in `pipeline.md` (M1 is lower than M2 because of the cost gate, not the time
shape).

**Policy reading of the dependence parameter.** Stronger positive dependence
between the time and money thresholds raises the joint survival
`C(S_T, S_M)` and so shrinks the group that fails the gates: people who
tolerate long trips also tolerate high costs, so fewer are stopped by either.
The consequence is not only smaller numbers. A smaller, more concentrated
non-surviving group is easier to miss in an aggregate or headcount policy
analysis, and it lies in the low deciles where the money gate binds (the
D2 range in the checks of `pipeline.md`: -28% to +19% around M2 at one-way
budgets, -38% to +46% with halved budgets). An analysis that assumes
dependence (or a generalised cost, which corresponds to the comonotone
extreme) therefore reports fewer excluded people than one that assumes
independence, without any change in their circumstances. Results for the
lowest deciles should be reported over the dependence range, not at a
single theta.

## 5. Diagnostics of a margin

For any margin, `core.families` computes the hazard `h = -d log S/dz`, the
elasticity `z h(z)`, the implied value of time `h_T / h_M` (the rate at which
the two gates trade off, which for the product form is not constant, unlike a
generalised-cost VOT), and the total-time-on-test transform. Used for the
diagnostics in `families.md`.

## 6. Competition (Shen)

The Hansen measure ignores that other people want the same jobs. The
package also implements the Shen (1998) competition-adjusted measure
(`core.competition`, `core.accessibility`, `engine.runner`): first
`V_j = sum_k P_k f(c_kj)` (demand at destination `j`), then
`A_i = sum_j D_j f(c_ij) / V_j`. Zero-competition destinations contribute
zero rather than `D/floor`. Competition is pooled by income class
(segments compete only for the jobs of their own class). The paper runs
described in `pipeline.md` use Hansen only; Shen is square (origins =
destinations) and stays so.

## 7. Where the inputs come from

| Quantity | How it is obtained | Doc |
|---|---|---|
| Segment populations | GSPREE: a Poisson GLM structure model on CBS household tables, then iterative proportional fitting to buurt margins | `segments.md` |
| Jobs by sector per buurt | LISA municipal jobs (2022) distributed over buurten with a log-linear model on municipal composition, KWB establishment counts, IPF | `data_lineage.md` |
| Jobs by income class | sectors ranked by CBS wages (81431NED); pools partition the jobs | `segments.md` |
| Admits home working | rough share per sector from CBS 85718NED and 82072NED | `pipeline.md` |
| Travel times | OSM (car, bicycle via r5py/R5; public transport via GTFS frequency model), peak load by road class | `skims.md` |
| Cost | car: per-km rates x distance + parking; PT: NS 2026 tariff + regional per-boarding and per-km | `skims.md` |

## 8. Robustness experiments

* **Impedance shape** (El-Geneidy-style): a hard 45-minute cut-off versus an
  exponential calibrated to the same cut-off (`cli.accessibility
  --time-shape --cutoff`, `run.compare`). Ranking of origins agrees (Spearman
  0.94 for car, 0.99 for bicycle) but levels differ (exponential/step:
  1.40 car, 0.74 bicycle), so conclusions about *levels and distribution*
  depend on the shape while the *ordering* does not.
* **Congestion**: car times under peak load rescale by 0.61 in accessibility
  with ranking preserved (Spearman 0.99).
* **Routed versus modelled distance** for car cost: no measurable effect.
* **PT router versus OpenTripPlanner**: `servers.md`.

## 9. Assumptions and limits

* Independent gates unless a copula is chosen; no parameters of the copula
  are estimated from data.
* Time margins are survey fits per mode; there is no separate margin per
  income class.
* Origin-destination travel times are between buurt centroids;
  far destinations may use a municipality point (`skims.md`).
* Wait time in public transport follows a frequency rule, not a timetable;
  no transfer penalty (a modelling choice, adjustable).
* Rail fares follow the published NS 2026 tariff table between tariff units,
  interpolated linearly; bus/tram/metro fares are a regional linear
  approximation.
* Job locations are imputed below the municipal level; the imputation is
  described and its limitations listed in `data_lineage.md`.
* Not implemented: shared-bicycle chains, scenarios S1-S4, specifications
  M1/M1', the interchangeability ratio, the reachability gap, NDW floating
  car data for congestion.
