# Skims (`ikob2.skims`)

Origin x destination travel times, distances and PT journey attributes.
Origins are the buurten of a study area (Utrecht: 111 buurten); destinations
are all Dutch buurten, so matrices are rectangular (111 x 14,318 float32 is
6 MB per matrix). Money costs are computed from these at run time.

## Skim store (`skims.store.SkimStore`)

One directory: `manifest.json` (origins, layers, metadata, progress) and one
`<layer>/<mode>/<variable>.npy` per matrix, opened as a memory map so reading
a few rows or columns touches only those. NaN means unreachable within the
routing limit or not yet computed.

**Near and far layers.** Destinations within `skims.near_km` (50 km) of any
origin are routed individually (layer `near`); every destination is also
represented by its municipality's point (layer `far`, one job-weighted point
per municipality). `SkimStore.combined(mode, variable, destinations,
near=..., far=..., fill=...)` assembles a full-resolution matrix: the near
value where it exists, otherwise the municipality value. `--far-cells none`
routes everything individually. Public transport is stored at full
resolution in layer `all`.

**Resumable.** Builds write blocks of origins and record them in the
manifest; a rerun continues where an interrupted one stopped. Finished blocks
are never recomputed, so a mode built with other settings needs a new name.

    python -m ikob2.cli.skims inspect <store>

## Car, bicycle and walking (`cli.skims build`)

    python -m ikob2.cli.skims build --kwb <wijkenbuurten_2022_v3.gpkg> \
        --study GM0344 --osm netherlands-260822.osm.pbf \
        --modes car bike walk --out <store> --max-memory 11G

Needs Java 21 and the `routing` extra.

* **car, bike**: R5 (r5py) on the OSM network between buurt centroids snapped
  to the street network; car at OSM free-flow speeds, bicycle at 16 km/h
  (`skims.cycle_kmh`). Limits `skims.max_minutes` (car 120, bike 90).
* **walk**: from zone geometry by default (`--walk-model zone`): crow-fly
  distance x 1.3 at 4.8 km/h between zones, and within a zone the mean
  distance between two random points, `0.52 sqrt(area)`. It ignores barriers.
  `--walk-model router` routes with R5 instead.
* **pt** (R5, optional): R5 with GTFS, median over a departure window
  (`skims.departure`, `window_minutes`), at most `max_rides` rides. The
  accessibility runs use the frequency model below instead.

The national OSM extract (1.4 GB) builds in R5 in about 8 minutes with an
11 GB heap.

## Car distances

R5 matrices carry no distance. The store's car `distance` comes from:

* **Valhalla** within `distance.radius_km` (30 km, crow-fly) of an origin
  (`skims.distance`, `servers.md`);
* the **detour model** beyond it, for the far layer, and where Valhalla finds
  no route: crow-fly distance x a detour factor per distance band
  (`skims.car.DetourModel`).

The detour model is calibrated on routed pairs (`cli.skims calibrate-detour`,
OSRM table service, `skims.osrm`): the median route/crow-fly ratio per band.
The seeded calibration (`intermediate/calibration/car_detour.json`, 10,840 Utrecht
pairs) gives 2.05 below 1.5 km, 1.6 at 2-6 km, 1.5 at 11 km, 1.33 at 37 km
and 1.23 from 90 km. Without a model the constant `car.detour_constant`
(1.3) is used.

    python -m ikob2.cli.skims calibrate-detour --kwb <gpkg> --study GM0344 --out <json>
    python -m ikob2.cli.skims build-distance <store> --kwb <gpkg> --detour <json>

## Car cost and parking (`skims.car`)

`car_time_and_cost` returns time and money separately. Time adds a parking
search: arrival search at the destination by KWB urbanisation class
(`car.parking_arrival_min`: 12 / 8 / 4 / 0 / 0 minutes for classes 1-5) and a
departure search at the origin of `car.departure_factor` (0.25) of the
origin's arrival value. Money is the variable cost per km x distance, plus an
optional per-km road charge and per-zone parking cost. Models
(`car.models`, `--car-model`): `fossil` 0.16 EUR/km (default), `electric`
0.05, `shared` 0.33 EUR/km + 0.05 EUR/min, `taxi` 2.40 EUR/km + 0.40 EUR/min.

## Peak load by road class (`skims.peak`)

A peak-load OSM extract divides `maxspeed` by a congestion factor by road
class, so travel times on that class are multiplied by it and route choice
reacts as well:

| Road class | Factor | Basis |
|---|---|---|
| motorway, trunk (and links) | 1.40 | TomTom Traffic Index |
| primary, secondary (and links) | 1.20 | Monitor Nationale Omgevingsvisie, Indicatoren Bereikbaarheid |
| tertiary, residential, unclassified, living street | 1.05 | minor interactions |

It is uniform in space and time, without bottlenecks. Speeds are rounded to
whole km/h (realised factors 1.39-1.41, 1.19-1.20, 1.03-1.05) and floored at
`peak.minimum_kmh`. Only ways with a numeric `maxspeed` change; in the Dutch
data nearly all have one (`peak.coverage`).

    python -m ikob2.cli.skims make-peak --osm <free-flow.pbf> --out <peak.pbf>
    python -m ikob2.cli.skims build ... --osm <peak.pbf> --modes car --out <peak store>
    python -m ikob2.cli.accessibility ... --study <peak store> --distance-study <free-flow store>

`--distance-study` takes the routed distances from the free-flow store.

## Public transport: a frequency model (`skims.gtfs_pt`)

One weekday of the GTFS feed (`pt.date`, 2026-09-15) is reduced to a peak
window (`pt.window_h`, 07:00-09:00): per line (route x direction) a headway
per stop (window / departures) and a median in-vehicle time between
consecutive stops. Rules:

* **Waiting** at every boarding: `min(headway / 2, pt.wait_cap_min)`, i.e.
  half the headway up to a 7.5-minute cap (travellers time their arrival to
  infrequent services).
* **Transfers**: no penalty (`pt.boarding_penalty_min` = 0), only the
  boarding wait and the walk between stops within `pt.transfer_radius_m`
  (300 m).
* **Walking** (access, egress, transfers): crow-fly x `pt.walk_detour` (1.3)
  at `pt.walk_kmh` (4 km/h); access and egress up to `pt.max_access_min`
  (20 minutes).
* Every journey has at least one boarding (separate before-boarding,
  after-alighting, boarded and riding nodes).

Dijkstra shortest paths (scipy) give door-to-door minutes up to
`pt.max_minutes` (180) from each origin to every buurt, in resumable blocks
of `pt.block_size` origins. Along the time-optimal journey the router also
accumulates in-vehicle kilometres by rail and by other lines and the number
of boardings onto other lines (`rail_km`, `other_km`, `other_boardings`):
crow-fly distance between consecutive stops x `pt.rail_detour` (1.15) or
`pt.other_detour` (1.25).

    python -m ikob2.cli.skims build-pt <store> --kwb <gpkg> --gtfs <zip> \
        [--date 2026-09-15 --window 7 9 --walk-kmh 4]

Limits: headways are per line and stop (parallel lines are not combined into
a higher frequency), and all route types share one waiting rule. The model
is compared with OpenTripPlanner in `servers.md`.

## PT fares (`skims.pt_fare`)

Applied when a run is set up, so fare assumptions change without rerouting:

* **Rail**: the NS single fare, second class, full tariff incl. VAT, valid
  from 1 January 2026 (`src/ikob2/skims/ns_2026_2e_klas.csv`): EUR 3.00 up to
  8 tariff units, 4.60 at 15, 8.00 at 30, 12.40 at 50, 19.10 at 80, 22.70 at
  100, 28.80 at 150 and 33.30 at 200, held beyond (`pt_fare.rail_beyond_table
  = "cap"`). One tariff unit is taken as one rail kilometre, read linearly
  between whole units. All rail operators use this table.
  `--pt-rail-discount` applies a share off (off-peak discounts do not apply
  to the 7-9 h peak); `--pt-rail-table km,eur.csv` uses another table;
  `--pt-rail-anchors` a tapering power law through two anchors
  (`pt_fare.rail_eur_per_km_at_1km`, `_at_100km`).
* **Bus, tram, metro, ferry**: EUR 1.08 boarding + 0.18 per km, the boarding
  charged once per journey (`--pt-boardings count` charges every boarding).

The fare is the PT cost matrix of the run, gated by the segments' cost
margins; a fare-scale table can scale it per segment (`scenarios.md`).

## Bicycle access and egress legs

`build-pt` stores further PT modes with bicycle legs (`gtfs_pt.LegSpec`):

    python -m ikob2.cli.skims build-pt <store> ... --access bike --mode-name pt_bw
    python -m ikob2.cli.skims build-pt <store> ... --egress bike --egress-hubs file \
        --hub-kind lime --mode-name pt_wb_lime

A bicycle leg rides at `bike_leg.kmh` (16 km/h, `--bike-kmh`) on crow-fly x
`bike_leg.detour` (1.3), at most `bike_leg.max_minutes` (20), plus
`bike_leg.fixed_minutes` (1) to unlock or park. An access leg can reach any
stop. An egress leg starts at a hub (`--egress-hubs`):

* `file` (default): hubs from `pt.hub_files`, each with its tariff kind from
  the parallel list `pt.hub_kinds` (`hubs/utrecht_hubs.csv` = `lime`,
  `ovfiets/locaties.json` = `ovfiets`); on the command line
  `--hub-file FILE:KIND` (repeatable) replaces both lists. The traveller alights at a stop within
  `pt.hub_walk_radius_m` (300 m) of a hub, walks to it and rides to the
  destination; the fastest hub per stop counts. `--hub-kind` keeps the hubs of
  one kind, so each kind is its own mode;
* `rail`: every rail stop; `all`: every stop.

The chosen journey's riding minutes are stored as `access_min` and
`egress_min` (ride only) for duration-priced tariffs. Within a kind the
fastest hub is taken; a slower but cheaper hub of the same kind (a Lime ride
just below a tier bound) is not an option.

## Hub files (`skims.hubs`)

* CSV with `lat`, `lon` and optionally `hub` (name): the municipal hubs
  (`inputs/hubs/utrecht_hubs.csv`: station hubs from the OV-fiets
  coordinates, street hubs from PDOK road geometries) and the extra hubs of
  scenarios S2c and S2t (`intermediate/hubs/utrecht_hubs_s2c.csv`, `_s2t.csv`; `cli.hubs propose`);
* the OV-fiets feed (`http://fiets.openov.nl/locaties.json`, a JSON object
  `{"locaties": {code: {"lat", "lng", "name", ...}}}`).

A relative path is looked up in the current folder, then under
`<root>/inputs` and `<root>/intermediate`; a path found in both is an error,
so a stale copy cannot shadow the file meant.

## Limits

* Car times are free flow or road-class peak load; no measured congestion.
  Measured speeds (NDW, including floating-car data) are available per route
  through an NDW account, not as a bulk national download.
* Travel times are between buurt centroids; the far layer approximates
  distant destinations by municipality points.
* PT waiting follows the headway rule, not the timetable.
