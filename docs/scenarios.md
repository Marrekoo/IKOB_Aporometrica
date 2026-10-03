# Shared bicycles, scenarios and the paper's diagnostics

How the shared-bicycle chains are priced and combined, how the scenarios
S0-S4 are set up and costed, and how the interchangeability ratio R, the
reachability gap and the paper tables are computed.

## Shared-bicycle chains (`run.shared_bike`)

### Skim modes

Built by `cli.skims build-pt` (`skims.md`); names are store modes:

| Mode | Access | Egress |
|---|---|---|
| `pt` | walk | walk |
| `pt_bw` | bicycle | walk |
| `pt_wb_<kind>` | walk | bicycle from a hub of `<kind>` |
| `pt_bb_<kind>` | bicycle | bicycle from a hub of `<kind>` |

`<kind>` is a hub tariff (`lime`, `ovfiets`), one mode per kind because the
tariffs differ and a slower hub can be the affordable one. The kinds read are
`shared_bike.egress_hub_kinds`; with an empty list the run reads `pt_wb` and
`pt_bb` at the OV-fiets charge. `shared_bike.egress_suffix` maps a kind to a
mode suffix, so `{lime = "_s2"}` reads `pt_wb_lime_s2` and `pt_bb_lime_s2`
(scenario S2).

### Tariffs

Added to the PT fare of the same journey:

| Rental | Price | Parameter |
|---|---|---|
| OV-fiets (egress) | EUR 4.80 per rental | `shared_bike.ovfiets_eur` |
| Lime (dockless access, Lime hub egress) | EUR 3 / 4 / 5 for rentals up to 20 / 30 / 40 minutes (ride + fixed minutes); the last tier beyond | `shared_bike.lime_tiers`, `dockless_model = "lime_tiers"` |
| Dockless, per minute | EUR 1.00 + 0.20 per riding minute | `dockless_model = "unlock_per_minute"` |
| Dockless, flat | one price per rental | `dockless_model = "flat"`, `flat_eur` |
| Own bicycle | free | |

`shared_bike.lime_scale` multiplies every Lime price. A **price-scale table**
(`paths.lime_price_scales`, `--price-scales FILE`) multiplies the Lime part
of the journey cost per segment, for concessions: rows
`household_type,income_class,scale`, `*` for all, later rows overriding
earlier ones. A **PT fare-scale table** (`paths.pt_fare_scales`,
`--pt-fare-scales FILE`, same layout) does the same for the PT fare part.
Examples are in `inputs/tariffs/` (seeded from `data/tariffs/`;
`*_d2_d4_50.csv` halves deciles 2-4).

Bicycle legs ride at 16 km/h on 1.3 x the crow-fly distance, at most 20
minutes, plus 1 fixed minute (`[bike_leg]`).

### Variants (`--shared-bike`)

A variant is a person's set of alternatives; a pair is acceptable if any
option passes both gates (`model_theory.md`, section 5). The share with a
private bicycle is the buurt's ownership share.

| Variant | Output mode | Who | Options |
|---|---|---|---|
| v0 | `pt_v0` | owners (others: plain PT) | plain PT; own bicycle to the stop, walk egress |
| v1 | `pt_v1` | everyone | plain PT; walk access + shared bicycle egress |
| v2 | `pt_v2` | owners: own bicycle access; others: dockless access; everyone: shared egress | plain; bicycle access; bicycle access + shared egress; walk + shared egress |
| v3 | `pt_v3` | as v2 | the options of v2 with leg-wise time gates (bicycle margin on bicycle legs, PT margin on the PT leg), cost on the journey total; independent thresholds |
| v4 | `bike_v4` | owners ride their own bicycle; others a dockless bicycle door to door | needs `--modes bike` |

v0 is the reference for the other variants: plain PT understates PT
accessibility for bicycle owners. Every variant assumes unlimited supply
(no stock-outs) and dockless availability at every origin.

## Scenarios

| Scenario | Change | Set-up |
|---|---|---|
| S0 | baseline | Lime tiers, current hubs |
| S1 | Lime prices halved | `--lime-scale 0.5` (blanket), or a price-scale table for targeted concessions |
| S2 | extra Lime hubs: citywide (S2c) or in the target buurten (S2t) | `cli.hubs propose` (`--within` for S2t), the S2 skim modes, and `--set 'shared_bike.egress_suffix={lime="_s2c"}'` (or `"_s2t"`) |
| S3 | S1 and S2 together | both settings |
| S4 | Lime prices halved for the residents of target buurten | `--price-scales lime_price_scales_all_50.csv --price-zones lime_price_zones_overvecht_kanaleneiland.csv` |

Fare concessions on PT itself use `--pt-fare-scales`. A flat dockless price
(`--dockless-model flat`, calibrated for revenue neutrality below) is a
further tariff option.

### S4: a price cut by home address (`--price-zones`)

A **price zone** (`paths.lime_price_zones`, `--price-zones FILE`, a CSV with
a column `buurtcode`) restricts the price-scale table to the residents of
the listed buurten, as when a concession is granted through the address
entered in the operator's app: their whole journey (dockless access, hub
egress) is priced with the scales, everyone else pays the full price. Every
listed code must be an origin of the run. Since origins are evaluated
independently, S4 equals S1 at the zone's origins and S0 elsewhere (tested).
`data/tariffs/lime_price_zones_overvecht_kanaleneiland.csv` holds the 15
target buurten: the 10 of Overvecht and Kanaleneiland-Noord, -Zuid,
Bedrijvengebied Kanaleneiland, Transwijk-Noord and -Zuid. Its public cost is
the revenue foregone of the zone's residents at baseline volume
(`scenario.cost.by_origin_eur_year` in `run.json`).

### S2: siting extra hubs (`skims.hub_siting`, `cli.hubs`)

    python -m ikob2.cli.hubs propose --data-root <root> \
        --base-accessibility <root>/outputs/runs/s0/accessibility.csv

Origin buurten are ranked by the mean of two percentile ranks, lowest first:
baseline accessibility of `siting.access_mode` (population-weighted over
segments, weight `siting.access_weight`) and private-bicycle ownership. New
hubs are placed at buurt centroids in that order, skipping buurten within
`siting.min_spacing_m` (400 m) of any hub, until there are
`(siting.hub_density_factor - 1)` times as many new hubs as existing hubs of
`siting.kind`. With `--within FILE` (`siting.within`, a CSV with `buurtcode`,
e.g. the S4 price zone) only those buurten are candidates, ranked among
themselves, and as many hubs are placed as the spacing allows (or
`--n-new`). The result goes to `intermediate/hubs/utrecht_hubs_<label>.csv`
(`--label`, default `s2`; `--out` to change it):

    python -m ikob2.cli.hubs propose --data-root <root> --label s2c \
        --base-accessibility <root>/outputs/runs/s0/accessibility.csv
    python -m ikob2.cli.hubs propose --data-root <root> --label s2t \
        --within lime_price_zones_overvecht_kanaleneiland.csv \
        --base-accessibility <root>/outputs/runs/s0/accessibility.csv

S2c doubles the 27 municipal hubs citywide; S2t places 12 hubs in the 15
target buurten (the other three have a hub within 400 m of their centroid).
The S2 skims are built with the existing and the new Lime hubs, each hub file
given with its tariff kind:

    python -m ikob2.cli.skims build-pt <store> --kwb <gpkg> --gtfs <zip> \
        --data-root <root> --egress bike --egress-hubs file \
        --hub-file hubs/utrecht_hubs.csv:lime \
        --hub-file hubs/utrecht_hubs_s2c.csv:lime \
        --hub-kind lime --mode-name pt_wb_lime_s2c

(and the same with `--access bike` for `pt_bb_lime_s2c`, and for `s2t`). `--hub-file`
replaces `pt.hub_files` and `pt.hub_kinds` together; relative paths are
found under `<root>/inputs` or `<root>/intermediate` (`skims.md`).

### Flat price calibration (`run.scenarios`)

The model has acceptable opportunities, not trips. Volume stands in as the
number of acceptable pairs (jobs of the income class x persons of the
segment) whose chosen option contains a Lime rental, where a person uses the
fastest acceptable option. With independent thresholds the probability that
option `k` (in time order) is the one chosen is

    S_T(t_k) [ S_M(c_k) - S_M(max(c_k, cmin_(k-1))) ]

(`choice_terms`, checked against Monte Carlo in the tests). Without
`--flat-eur`, the flat price is calibrated for revenue neutrality:
`fixed_point` (price x rentals at that price = baseline revenue, default) or
`weighted_mean` (baseline volumes). Calibration requires M2. The calibrated
price is written to `run.json` under `scenario`.

### Usage and effectiveness

`--report-usage` records Lime revenue and rentals per segment and per origin
in `run.json`.
`cli.compare run_a run_b --mode pt_v2` then writes, by income class and
household type, each group's share of the gain and of the cost and their
ratio (`run.scenarios.effectiveness`).

## Public cost (`run.costs`, `[costs]`)

* **S2 hubs**: set-up cost per cluster of `hub_cluster_size` hubs spread
  over `horizon_years` (annuity at `discount_rate`), counted in full for the
  total public cost and net of `province_capex_share` for the municipality;
  plus yearly rebalancing and enforcement per cluster.
* **S1 price cut**: the municipality compensates the operator for revenue
  foregone at baseline volume. Model rentals are scaled to observed rides
  (`rides_observed` over `rides_months`, fleet `fleet`) at the model's mean
  price per rental. `hubs_for_budget` gives the number of hubs with the same
  annual cost, for comparisons at equal expenditure.
* **PT fare concessions**: revenue foregone at baseline volume,
  `sum_s persons_s x spend[decile_s] x (1 - scale_s)`, with `spend` the
  ODiN fare spending per person and year by decile (`cli.segments
  pt-spend`, `segments.pt_spend`); the local (shrunk) estimate is central,
  the national-only estimate the low bound. Reported by the run when the
  spending table exists.

The cost of S1 covers the whole concession area's rides; the gains are
those of the study area's residents.

## Interchangeability ratio R (`run.interchange`, `cli.interchange`)

    R[i, s] = da(A)[i, s] / da(B)[i, s]

the gain of intervention A over that of B for origin `i` and segment `s`,
both against the same baseline:

    python -m ikob2.cli.interchange --data-root <root> \
        --base s0 --a s1 --b s2 --mode pt_v2

Writes to `outputs/comparisons/<a>_over_<b>/`: `interchange_pairs.csv` (one
row per origin and segment), `interchange_origins.csv` (dispersion of R
across an origin's segments: CV, interquantile ratio and spread at
`--quantiles 0.9 0.1`) and `interchange_summary.csv` (population-weighted
pooling). R is undefined where `|da(B)| <= --tol`; those pairs stay in the
table with `defined = False` and are counted. Under a generalised cost with
common inputs R does not depend on the segment; under the gates it does,
and its dispersion is the diagnostic.

## Reachability gap (`run.gap`)

    G[i, s] = sum_j D[j, s] [ S_T(t_ij) - f_s(t_ij, c_ij) ]

the jobs that pass the time gate but fail the money gate: the difference
between a time-only run (`--no-cost-gate`) and the gated run of the same
mode, segment and origin. Under independence it equals
`sum_j D[j, s] S_T(t_ij) (1 - S_M(c_ij))`. The atom is reported with it:
the part of the gap no price cut can close.

## Paper tables (`cli.paper_tables`)

    python -m ikob2.cli.paper_tables --data-root <root> \
        --tags m0 m1 m1c m1p m2 m3t1.25 m3t1.5 m3t2 m3t4 m3tinf --mode pt_v2

Reads the runs `sp_<tag>_<s0|s1|s2>` (income-matched jobs), `spc_<tag>_<s0|s1|s2>`
(`--common-jobs`, the controlled comparison) and `spt_<tag>_<s0|s2>`
(time only; M3 uses the M2 time-only run), and writes to
`outputs/comparisons/specs/`:

| File | Content |
|---|---|
| `baseline_by_spec.csv` | accessibility, raw and normalised, and atom, per specification and income class |
| `incidence_by_spec.csv` | gains of S1 and S2 by income class and household type, per specification |
| `interchange_by_spec.csv` | R: pooled dispersion, median, undefined pairs, full and controlled |
| `gap_by_spec.csv` | the reachability gap before and after S1, and the atom |
| `correlation_by_spec.csv` | Pearson and Spearman correlations between specifications of levels and of the S1 and S2 gains |

The run names are a convention: produce them with `cli.accessibility --run
sp_m2_s0 --spec m2 ...` and so on.
