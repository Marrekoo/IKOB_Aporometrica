# Data specification

Every input, intermediate and output file: what it is, where it comes from,
its format and which code reads or writes it. The folder layout is
`ikob2.utils.paths.DataLayout` (default root `/home/marco/IKOB data`, or
`--data-root`). Nothing under `inputs/` is ever written by the code.

## Inputs (external, read only)

| Path | Content | Source | Used by |
|---|---|---|---|
| `inputs/kwb/wijkenbuurten_2022_v3.gpkg` (also 2018 and 2024 vintages) | CBS Kerncijfers wijken en buurten 2022, buurt polygons and attributes, RD New (EPSG:28992) | CBS/PDOK | `data.geopackage.load_cbs_buurten` (zones, centroids, KWB covariates) |
| `inputs/lisa/LISA_Gemeenten_2025.xlsx` | Jobs per municipality and LISA sector (15 sectors L01-L15), used for the year 2022 | LISA | `segments.lisa` |
| `inputs/legacy_ikob/` | `Alle_Zones_2030_2040.xlsx`, `Buurt naar inkomensverdeling arbeidsplaatsen.xlsx`, `Ralph_Sahar_CBS_buurten_met_banen_naar_opleidingsniveau.xlsx`: legacy IKOB job totals per buurt and education mix | earlier IKOB runs | `segments.jobs`, `segments.jobs_impute` |
| `inputs/osm/*.osm.pbf` | OpenStreetMap extracts: national file (`netherlands-260822`) and Geofabrik province files | Geofabrik | routing |
| `inputs/gtfs/gtfs-nl.zip` | National GTFS feed (public transport timetables) | OVapi/NDOV | `skims.gtfs_pt`, OTP |
| `inputs/survey/S_T_work.csv` (copy in `data/margins/`) | Weibull time margins per mode and job type | survey fit by the authors | `segments.time_margins` |
| `inputs/odin/` | ODiN tables and codebook (trip counts behind the reference budgets) | CBS/RWS | background of Table 6 |
| `ns-prijslijst-2026-nl.pdf` | Official NS 2026 price list | NS | transcribed to `skims/ns_2026_2e_klas.csv` |

## Files shipped in the repository (`data/`)

| File | Columns / content |
|---|---|
| `data/envelope/reference_budgets.csv` | `household_type, income_class (D1-D10), low, high` (euro per trip), `km_low, km_high` (equivalent car km). D1 rows are blank (censored, atom = 1). Table 6 of the paper. |
| `data/margins/S_T_work.csv` | `wfh` (No home working / Home working), `mode`, `eta` (Weibull scale, minutes), `k` (shape), `median`, `class` |
| `data/calibration/car_detour.json` | `km` grid and detour factor: routed distance / crow-fly distance, calibrated with OSRM |
| `data/statline/*.csv` | Snapshots of CBS StatLine tables (copied to `cache/statline/`) so runs need no network: `86161NED` (households by type and income decile, 2022), `71487ned` (households with children by age), `81431ned` (jobs and hourly wage by sector), `85318NED` (KWB establishments per buurt in 8 SBI groups), `85718NED` (working from home), `82072NED` (education by sector) |
| `src/ikob2/skims/ns_2026_2e_klas.csv` | NS single-fare table: tariff units (km) -> euro, second class, 2026 |

## Intermediate (derived; delete and rebuild)

| Path | Format | Produced by |
|---|---|---|
| `intermediate/segments/nl_segments.gpkg` (optional) | one row per buurt, population per household type x income decile; `cli.accessibility` computes the segments on the fly from KWB and the StatLine snapshots, so this file is only needed for inspection | `cli.segments run` |
| `intermediate/jobs/sector_jobs_2022.csv` | `buurtcode, L01..L15`: jobs per LISA sector per buurt | `cli.segments jobs` |
| `intermediate/skims/<study>/` | skim store (below) | `cli.skims build`, `build-distance`, `build-pt` |
| `intermediate/calibration/` | detour model | `cli.skims calibrate-detour` |
| `intermediate/osm_peak/*.peak.osm.pbf` | OSM with peak-load speeds | `cli.skims make-peak` |
| `intermediate/osm_walk/*.walk.osm.pbf` | OSM reduced to the pedestrian network | `skims.osm_walk` |
| `intermediate/valhalla/` | Valhalla tiles, config, logs | `cli.servers valhalla build` |
| `intermediate/otp/` | OTP jar, `graph/` (inputs, `graph.obj`), GTFS subsets, logs | `cli.servers otp prepare/build` |

### Skim store (`skims.store.SkimStore`)

`manifest.json` (origin ids, layers, metadata, block progress) and
`<layer>/<mode>/<variable>.npy`, each an origins x destinations `float32`
matrix opened as a memory map. NaN = not reachable or not computed.

| Layer | Meaning |
|---|---|
| `near` | destinations within `--near-km` of an origin, routed individually (car, bike, walk) |
| `far` | the remaining destinations mapped to their municipality's point |
| `all` | full-resolution layer added by `add_layer`; used for public transport |

`SkimStore.combined(mode, variable, ...)` assembles a full-resolution matrix
from `near` and `far`.

| Mode | Stored variables |
|---|---|
| car | `time` (minutes, free-flow; a separate store `<study>_peak` holds peak-load times), `distance` (km, routed within 30 km and detour-modelled beyond) |
| bike | `time` (16 km/h) |
| walk | `time` (4 km/h from zone size) |
| pt (layer `all`) | `time` (minutes, frequency model), `rail_km`, `other_km`, `other_boardings` |

Money costs are not stored: they are computed at run time from the stored
distances by `skims.car.CarCostModel` (car: per-km rate x distance +
parking) and `skims.pt_fare.PtFareModel` (PT: NS 2026 table on `rail_km` +
regional boarding and per-km fare on `other_km`, `other_boardings`).

## Outputs (`outputs/runs/<run>/`)

| File | Content |
|---|---|
| `accessibility.csv` | long table, one row per origin x mode x segment: `buurtcode, mode, segment, household_type, income_class, population, accessibility` (acceptable jobs), `atom` (share for whom no priced trip is acceptable), `accessibility_normalised` |
| `summary_household.csv`, `summary_income.csv` | population-weighted means by mode and household type / income class |
| `run.json` | all parameters of the run: modes, time shape, cut-off, cost gate, fare model, skim store, data versions |

`outputs/comparisons/` holds run-versus-run tables (`cli.compare`): rank
correlation, level ratios and their distribution over segments.

## Conventions

* Buurt codes are strings `BUnnnnnnnn`; municipality codes `GMnnnn`.
* Time in minutes, distance in km, money in euro (price level of the
  source; NS 2026 and CBS 2022 are not deflated to a common year).
* Population counts are non-integer after IPF; jobs are non-integer after
  imputation.
* CBS suppression sentinel -99999999 is converted to NaN on load.
