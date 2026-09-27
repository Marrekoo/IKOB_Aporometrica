# Data specification

Every input, intermediate and output file: content, source, format and the
code that reads or writes it. The folder layout is
`ikob2.utils.paths.DataLayout` under the data root (`--data-root`,
`$IKOB_DATA_ROOT` or `paths.data_root`).

    <root>/
      inputs/          source data, read only
      cache/statline/  CBS StatLine snapshots
      intermediate/    derived, rebuildable
      outputs/         runs and comparisons

`python -m ikob2.cli.layout --root <root> create` makes every folder below
and copies the reference files of the repository's `data/` into it.
File names inside `inputs/` that the commands look for by default are
`paths.*` parameters (`lisa`, `ikob_jobs`, `education_jobs`, `odin`,
`bike_ownership`, `osm_national`, `survey_margins`).

## Inputs (`inputs/`)

| Path | Content | Source | Used by |
|---|---|---|---|
| `kwb/wijkenbuurten_<year>_v3.gpkg` | CBS Kerncijfers wijken en buurten: buurt polygons and attributes, RD New (EPSG:28992); 2022 is the model year | CBS/PDOK | `data.geopackage.load_cbs_buurten`, `segments.kwb` |
| `lisa/LISA_Gemeenten_2025.xlsx` | jobs per municipality and LISA sector (15 sectors), 2016-2025 on 2025 boundaries | LISA | `segments.lisa` |
| `legacy_ikob/Alle_Zones_2030_2040.xlsx` | IKOB job table: jobs per 2022 buurt by income group and year (sheet `buurten-arbeidsplaatsen`), derived from NRM zone totals | IKOB model | `segments.jobs.parse_legacy_jobs` (buurt job totals) |
| `legacy_ikob/Ralph_Sahar_CBS_buurten_met_banen_naar_opleidingsniveau.xlsx` | LISA 2016 jobs per buurt by education level | Municipality of Amsterdam | `segments.jobs_impute.parse_education_shares` |
| `osm/*.osm.pbf` | OpenStreetMap: national extract (`paths.osm_national`) and province extracts (`paths.osm_regions`) | Geofabrik | R5, Valhalla, OTP, `skims.peak`, `skims.osm_walk` |
| `gtfs/gtfs-nl.zip` | national GTFS feed | OVapi / NDOV | `skims.gtfs_pt`, OTP |
| `survey/S_T_work.csv` | Weibull time margins per mode and job type: `wfh` (No home working / Home working), `mode`, `eta` (minutes), `k`, `median`, `class` | survey fit (seed: `data/margins/`) | `segments.time_margins` |
| `envelope/reference_budgets.csv` | reference budgets: `household_type, income_class (D1-D10), low, high` (EUR per journey), `km_low, km_high` (car km equivalents); D1 rows blank (censored, atom 1) | the paper's envelope (seed: `data/envelope/`) | `segments.bridge.load_reference_budgets` |
| `envelope/sources/*.csv` | the inputs of the reference-budget envelope: Nibud basket, Warnaar anchors, CBS income percentiles, car costs, ...; each row with its source (`envelope/README.md`) | seed: `data/envelope/sources/` | `ikob2.envelope` |
| `tariffs/lime_price_scales*.csv`, `tariffs/pt_fare_scales*.csv` | multipliers on the Lime price and on the PT fare by segment: `household_type, income_class, scale` (`*` = all, later rows override) | scenario definitions (seed: `data/tariffs/`) | `run.shared_bike.load_price_scales` |
| `odin/ODIN_22_23_clean.csv` | pooled ODiN 2022-23 (persons, households, tours, legs) | CBS / RWS | `segments.car_availability`, `segments.pt_spend` |
| `veh_owners/bike_ownership_buurten.csv` | one row per Utrecht buurt: `buurtcode, buurtnaam, wijkcode, wijknaam, aantal_inwoners, buurtteam, pct_with_bicycle` (0-100), `mapping_confidence`; the Utrecht buurtteam survey 2025 assigned to buurten by hand | Municipality of Utrecht | `segments.ownership.load_bike_ownership` |
| `hubs/utrecht_hubs.csv` | the municipal shared-bicycle hubs: `hub, lat, lon, precision, source` | Municipality of Utrecht, PDOK | `skims.hubs` (kind `lime`) |
| `ovfiets/locaties.json` | OV-fiets locations, `{"locaties": {code: {lat, lng, name, ...}}}` | fiets.openov.nl | `skims.hubs` (kind `ovfiets`) |

## Files in the repository (`data/`, `src/`)

`data/` holds the reference files that `cli.layout create` copies into a new
data folder (it never overwrites): `envelope/`, `margins/`, `tariffs/`,
`statline/` and `calibration/`. The runs read the copies in the data folder,
not these.

| File | Content |
|---|---|
| `data/envelope/reference_budgets.csv` | reference budgets (see `inputs/envelope/` above) |
| `data/margins/S_T_work.csv` | Weibull time margins (see `inputs/survey/`) |
| `data/tariffs/*.csv` | price and fare scale tables (see `inputs/tariffs/`) |
| `data/statline/*.csv` | StatLine snapshots: 86161NED (households by type and income decile, 2022), 71487ned (households with children), 81431ned (jobs and hourly wage by sector, 2022), 85318NED (KWB establishments per buurt in 8 SBI groups), 85718NED (working from home by education, 2024), 82072NED (education of employee jobs by sector, 2010) |
| `data/calibration/car_detour.json` | detour factor (route / crow-fly distance) by distance band |
| `src/ikob2/skims/ns_2026_2e_klas.csv` | NS single fare, second class, full tariff, from 1 January 2026: `te` (tariff units) -> `eur` |
| `src/ikob2/defaults.toml` | all parameters |

## Intermediate (`intermediate/`)

| Path | Content | Written by |
|---|---|---|
| `segments/nl_segments.gpkg` | persons per buurt in 44 segments (inspection only; runs compute segments on the fly) | `cli.segments run` |
| `jobs/sector_jobs_<year>.csv` | `buurtcode, L01..L15`: jobs per LISA sector per buurt | `cli.segments jobs` |
| `ownership/car_availability_<study>.csv` | `household_type, income_class, n_national, share_national, n_local, share_local, share` | `cli.segments car-availability` |
| `ownership/pt_spend_<study>.csv` | per income decile: PT trips per person and year, mean fare, spending (local shrunk and national) | `cli.segments pt-spend` |
| `envelope/odin/*.csv` | ODiN aggregates for the envelope (tour rates, composition, commuting) | `cli.envelope aggregates` |
| `envelope/*.csv` | the re-derived envelope: anchors, residuals, tour bounds, scenario grid, `reference_budgets.csv` | `cli.envelope build` |
| `hubs/utrecht_hubs_s2.csv` | extra Lime hubs of scenario S2: `hub, lat, lon, precision, source, buurtcode, access, bike_share, score` | `cli.hubs propose` |
| `skims/<study>/` | skim store (below) | `cli.skims build`, `build-distance`, `build-pt` |
| `calibration/car_detour.json` | detour model (seeded from `data/calibration/`) | `cli.skims calibrate-detour` |
| `osm_peak/*.peak.osm.pbf` | OSM with peak-load speeds | `cli.skims make-peak` |
| `osm_walk/*.walk.osm.pbf` | OSM reduced to the pedestrian network | `skims.osm_walk.make_walk_extract` |
| `valhalla/` | Valhalla tiles, config, logs | `cli.servers valhalla build` |
| `otp/` | OTP jar, `graph/` (inputs, `graph.obj`), GTFS subsets, logs | `cli.servers otp prepare/build` |

### Skim store (`skims.store.SkimStore`)

`manifest.json` (origin ids, layers, metadata, block progress) and
`<layer>/<mode>/<variable>.npy`, each an origins x layer-destinations
`float32` matrix opened as a memory map. NaN = unreachable or not computed.

| Layer | Meaning |
|---|---|
| `near` | destinations within `--near-km` of an origin, routed individually (car, bike, walk) |
| `far` | the remaining destinations, represented by their municipality's point |
| `all` | every destination at full resolution (public transport) |

| Mode | Variables |
|---|---|
| `car` | `time` (minutes), `distance` (km: Valhalla within 30 km, detour model beyond) |
| `bike` | `time` |
| `walk` | `time` |
| `pt` (layer `all`) | `time`, `rail_km`, `other_km`, `other_boardings` |
| `pt_bw`, `pt_wb[_<kind>]`, `pt_bb[_<kind>]` (layer `all`) | as `pt`, plus `access_min` (riding minutes of a bicycle access) and `egress_min` (riding minutes of a bicycle egress) |

Money costs are not stored; they are computed at run time
(`skims.car.CarCostModel`, `skims.pt_fare.PtFareModel`,
`run.shared_bike.SharedBikeTariffs`).

## Outputs (`outputs/runs/<run>/`)

| File | Content |
|---|---|
| `accessibility.csv` | one row per origin x mode x segment: `buurtcode, mode, segment, household_type, income_class, population, accessibility` (acceptable jobs, conditional on having the mode), `atom`, `accessibility_normalised`; with `--ownership` also `availability` and `accessibility_expected` |
| `summary_income.csv`, `summary_household.csv` | population-weighted means by mode and income class / household type; `*_expected.csv` for `accessibility_expected` |
| `run.json` | resolved parameters, arguments, specification, scenario (calibrated S4 price, Lime usage, public cost), and `input_files`: path and SHA-256 of every input file |
| `segments.csv` | segment definitions: budget interval, atom, Lime price scale |
| `time_margins.csv` | Weibull margins used |
| `hubs.csv` | shared-bicycle hubs used (name, kind, lat, lon) |
| `origins.gpkg` | buurt polygons with mean accessibility per mode (`acc_<mode>`, `accx_<mode>`) and money-gate summary (`mg_*`), and hub points; EPSG:28992 |
| `money_gate_curves.csv`, `money_gate_ttt.csv`, `money_gate_summary.csv` | aggregated cost-threshold survival, hazard and TTT per origin (`outputs.diagnostics`) |
| `accessibility.parquet` | `accessibility.csv` as Parquet (when pyarrow is installed) |
| `dictionary.csv`, `README.md` | every column with description and unit; how to join the files (R snippet) |

`outputs/comparisons/` holds `cli.compare` (`<a>__vs__<b>/`),
`cli.interchange` (`<a>_over_<b>/`) and `cli.paper_tables` (`specs/`)
tables (`pipeline.md`, `scenarios.md`).

## Conventions

* Buurt codes are strings `BUnnnnnnnn`; municipality codes `GMnnnn`;
  segments `<household_type>_<income_class>`.
* Time in minutes, distance in km, money in euro at the price level of the
  source (NS 2026, CBS 2022; not deflated to a common year).
* Populations and jobs are non-integer after IPF and imputation.
* Every negative value in a CBS numeric attribute is a suppression code and
  is read as NaN.
