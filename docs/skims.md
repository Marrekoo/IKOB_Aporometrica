# Skims (`ikob2.skims`)

Origin x destination travel times for the accessibility runs, computed
with **r5py** (the R5 engine, the same as r5r) from OpenStreetMap and,
for public transport, GTFS. Origins are the buurten of a study area
(first: Utrecht city, 111 buurten); destinations are all buurten
nationally. The matrices are therefore rectangular and small (111 x
14,318 float32 = 6 MB per matrix).

## Store (`skims.store.SkimStore`)

One directory: `manifest.json` (origins, layers, metadata, progress)
and one `<layer>/<mode>/<variable>.npy` per matrix. Matrices are opened
as memory maps, so reading a few origin rows or destination columns
touches only those; NaN means not reachable within the routing limit
(or not computed yet).

**Near and far layers.** Destinations within `--near-km` of any origin
are routed individually (layer `near`); every destination is also
mapped to its municipality's point (layer `far`, one job-weighted
centroid per municipality). `SkimStore.combined(mode, variable,
destinations, near=..., far=...)` assembles a full-resolution matrix on
demand: the near value where the destination has one, otherwise the
value of its municipality point. Routing and storage cost scale with
the near destinations plus the cells, not with all buurten. The far
value is a municipality-level approximation; whether that is good enough
for the far tail of a 45-minute threshold is a modelling choice (set
`--far-cells none` to route everything individually).

**Resumable.** Builds write origin blocks and record them; rerunning
continues where an interrupted run stopped.

## Building

    python -m ikob2.cli.skims build --kwb <wijkenbuurten_2022_v3.gpkg> \
        --study GM0344 --osm netherlands.osm.pbf --gtfs gtfs-nl.zip \
        --modes car bike walk pt --out data/skims/utrecht --max-memory 11G
    python -m ikob2.cli.skims inspect data/skims/utrecht

Needs Java 21 and the `routing` extra (`pip install -e .[routing]`).

* **car, bike:** R5 on the OSM network; free-flow car speeds (no
  congestion), 16 km/h cycling (confirmed as the project's cycling speed). Times are between buurt centroids
  snapped to the street network.
* **walk:** by default from zone geometry (`--walk-model zone`):
  crow-fly centroid distance x 1.3 at 4.8 km/h, and within a zone the
  mean distance between two random points from its land area
  (0.52 sqrt(A)). Rough; ignores barriers.
* **pt:** R5 with GTFS, median over a departure window (default 08:00
  plus 60 minutes), walking access and egress, at most 4 rides.
  Not yet exercised on the national feed.

## Using a store with the engine

    t = store.combined("car", "time", dest_codes, near="near", far="far",
                       fill=9999.0)                    # (origins, dests)
    a = runner.run_hansen(None, segments,
                          cost_matrices={"time": t, fare.matrix_id: c},
                          opportunities=pools_over_dest_codes)

`dest_codes` and the job pools must use the same destination order.
`fill` replaces unreachable pairs by a large travel time (weight 0).

## Trial results (development extract)

Utrecht province OSM extract, 111 origins x 2,579 near + 346 far
destinations, three modes: network build 32 s, 3.5 minutes in total,
store 4 MB. Median car time 27 minutes; median bike/car time ratio 2.6.
(Reachability is low only because the extract stops at the province
edge.)

## Idea: congestion from measured traffic data (NDW)

R5 routes with static speeds, so car times are free-flow. NDW data could
correct them by time of day. What NDW offers (read from opendata.ndw.nu,
the Dexter web application and docs.ndw.nu, September 2026):

* **Real-time files** (opendata.ndw.nu, no login): `trafficspeed`,
  `traveltime`, speeds and intensities per measurement site, DATEX II
  (v2.3 stops 9 February 2027, v3 profiles replace it). Snapshots only,
  no history.
* **Dexter, historical data** (dexter.ndw.nu; docs: "Historische data"):
  explorer (hourly values, public), export (finest available
  aggregate, Excel/CSV) and reports. Export types include intensity and
  speed, travel time, traffic jams (files), incidents and **Floating Car
  Data (FCD)**, listed as available to "everyone". The export API answers
  401 without login, so exports need an NDW account; scripted access is
  not documented.
* **Dexter open data** (no login): a limited set from the export part
  (loop-based intensity and speed, bicycle counts). It is a web form
  with a captcha and caps on the number of days and locations, so it is
  not a bulk or scriptable source.
* **FCD specifics** (docs): the network is cut into segments of at most
  50 m; hourly means per segment; results are per ROUTE that the user
  draws in the tool. NDW states that the FCD explorer is indicative only
  and that only the FCD export is a reference source.

Consequences: there is no bulk national FCD download. A usable design is
to export hourly speeds for a limited set of representative routes (the
motorway and main-road corridors around Utrecht) and derive a
congestion index per hour and day type (mean travel time / free-flow
travel time), then apply it to the R5 times. Applying it needs a rule for
which OD pairs use those corridors: either factors by distance band and
region, or the road-class share of each route (needs detailed
itineraries). Time-of-day speeds per OSM way would need a match between
NDW locations or segments and OSM ways. An alternative without an
account is to archive the real-time `trafficspeed` feed ourselves for a
few weeks and build the profile from that (loop sites only, and only from
the start of collection). Open questions for NDW: bulk or API access to
FCD exports, licence terms for publication, and whether historical
minute data can be requested for many routes.

## Peak load by road class (`skims.peak`)

Free-flow times understate peak travel. As a first correction the OSM
extract is rewritten so that the routers see peak speeds: `maxspeed` is
divided by a congestion factor by road class, so travel times on those
classes are multiplied by it and route choice reacts as well.

| Road class | Factor | Basis |
|---|---|---|
| motorway, trunk (and links) | 1.40 | TomTom Traffic Index |
| primary, secondary (and links) | 1.20 | Monitor Nationale Omgevingsvisie, Indicatoren Bereikbaarheid |
| tertiary, residential, unclassified, living street | 1.05 | minor extra interactions in the streets |

Enough to mimic a peak load, not a congestion model: uniform in space and
time, no bottlenecks. Speeds are rounded to whole km/h, so realised
factors are 1.39 to 1.41, 1.19 to 1.20 and 1.03 to 1.05. Only ways with a
numeric `maxspeed` are changed; in the Dutch data nearly all are (motorway
100%, residential 99%, checked on the Utrecht extract). Tested on the
Utrecht extract, car times from the Dom Tower rise 7% to 19% (Amersfoort 28
to 32 minutes).

    python -m ikob2.cli.skims make-peak --osm <free-flow.pbf> --out <peak.pbf>
    python -m ikob2.cli.skims build ... --osm <peak.pbf> --modes car --out <peak store>
    python -m ikob2.cli.accessibility ... --study utrecht_nl_peak \
        --distance-study utrecht_nl

`--distance-study` reuses the routed distances of the free-flow store
(distances barely change with speeds). The national peak extract is built
in the data folder under `intermediate/osm_peak`.

## Public transport from GTFS: a frequency model (`skims.gtfs_pt`)

Not R5, and no averaging over departure times. One weekday of the GTFS
feed (default Tuesday 2026-09-15) is reduced to a peak window (default
07:00-09:00): per line (route x direction) a headway per stop (window /
departures) and a median in-vehicle time between consecutive stops.
Rules:

* **Waiting** at every boarding: `min(headway / 2, 7.5)` minutes, i.e.
  half the headway below 15 minutes and the same 7.5-minute average
  above it (infrequent services are used by timing the arrival).
* **Transfers** are not penalised: no extra transfer penalty, only the
  boarding wait and the walk between stops (`--boarding-penalty-min`
  exists, default 0).
* **Walking** (access, egress, transfers): crow-fly distance x detour 1.3
  at an adjustable speed, default **4 km/h**; access and egress up to 20
  minutes; transfers between stops within 300 m.
* A trip always contains at least one boarding (the graph has separate
  before-boarding, after-alighting, boarded and riding nodes).

Shortest paths over the graph (scipy Dijkstra) give door-to-door minutes
for the 111 origins to all 14,318 destination buurten; the result is the
store layer `all` (`pt/time`).

    python -m ikob2.cli.skims build-pt <store> --kwb ... --gtfs ... \
        [--date 2026-09-15 --window 7 9 --walk-kmh 4]

Limits: headways are per line and stop (parallel lines are not combined
into a higher frequency) and all route types have the same wait rule.

### PT fares (`skims.pt_fare`)

`build-pt` also stores, for the time-optimal journey of every pair, the
in-vehicle kilometres by rail and by other lines and the number of
boardings onto other lines (`rail_km`, `other_km`, `other_boardings`). The
kilometres are the crow-fly distance between consecutive stops times a
detour (rail 1.15, other 1.25; `--rail-detour`, `--other-detour`),
accumulated along the shortest-path tree. The fare is computed when a run
is set up, so fare assumptions change without new routing:

* **Rail**: the official NS single fare, second class, full tariff incl.
  VAT, valid from 1 January 2026 (`src/ikob2/skims/ns_2026_2e_klas.csv`,
  from the price list "NS Tarieven Consumenten"): 3.00 EUR up to 8 tariff
  units, 4.60 at 15, 8.00 at 30, 12.40 at 50, 19.10 at 80, 22.70 at 100,
  28.80 at 150 and 33.30 at 200, held there beyond; one tariff unit is one
  tariff kilometre, read linearly between whole units. `--pt-rail-discount
  0.2` / `0.4` applies NS's discounts, `--pt-rail-table km,eur.csv` a
  different table, `--pt-rail-anchors` the tapering power law through the
  paper's anchors. (Fares you had noted earlier, 2.70 up to 8 km, 4.40 at
  15, 21.30 at 100 and 29.40 at 200, correspond to an earlier year: the
  2026 list is about 5 to 13% higher.) Other rail operators are priced
  with the same table.
* **Bus, tram, metro, ferry**: 1.08 EUR boarding + 0.18 EUR per km, the
  boarding charged once per journey (`--pt-boardings count` charges every
  boarding).

Example fares from Leidsche Rijn: Amsterdam (36 km rail + 8 km bus) 11.8
EUR, Amersfoort 9.5, Rotterdam 15.7, Arnhem 17.5, Groningen 33.2 (rail
only, 199 km); median over reachable pairs 19.8 EUR. The fare goes to the
run as the PT cost matrix, gated by the segments' cost margins like the
car cost.

## Bicycle access and egress for public transport

`cli.skims build-pt --mode-name pt_bw --access bike` (and `pt_wb --egress
bike`, `pt_bb` both) store further PT modes in the same store; the plain mode
is `pt`. A bicycle leg (`gtfs_pt.LegSpec`) rides at `--bike-kmh` (16) with
crow-fly detour `--bike-detour` (1.3), at most `--bike-max-min` (20) minutes,
plus `--bike-fixed-min` (1) for unlocking or parking. A bicycle egress starts
only at hub stops (`--egress-hubs rail`: rail stops, standing in for OV-fiets
stations; `all` for every stop); an access can use any stop. With a bicycle
access the riding minutes of the chosen access are stored as `access_min` for
metered tariffs. The router still applies the frequency model unchanged.

## National network

The national OSM extract (`netherlands-260822.osm.pbf`, 1.4 GB) builds
in 453 s with an 11 GB heap and peaks at 10.6 GB resident on the 15 GB
development machine, so a national OSM-only network works for R5 and
Valhalla. Free-flow car times from Utrecht: Amsterdam 41, Rotterdam 48,
Groningen 126, Maastricht 128 minutes. National public transport is
computed by the GTFS frequency model (no street graph needed beyond
walking access); OpenTripPlanner is used as an independent check on it and
runs on a walking-network extract, see `servers.md`.

## Not yet built

* **List-type skim** for cheap/slow versus expensive/fast options per
  OD pair (the shared-bicycle case).
* **Shared-bicycle legs** (hubs, dockless supply) and the scenario
  variants S1-S4, which need their own networks.
* The choice of departure windows and percentile for PT (the frequency
  model uses none; OTP validation is in `servers.md`).
