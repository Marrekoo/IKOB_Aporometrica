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
  congestion), 16 km/h cycling. Times are between buurt centroids
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

R5 routes with static speeds, so car times are free-flow. To correct for
congestion by time of day, NDW open data (https://opendata.ndw.nu/) is a
candidate. That page offers real-time snapshots (loop-based
`trafficspeed`, `traveltime`, speeds and intensities; DATEX II, moving
to v3 profiles by February 2027); historical and aggregated data is on
the linked Dexter portal (https://dexter.ndw.nu/opendata/). Not yet
examined: whether Dexter has floating-car data or only roadside
measurements, its time coverage and its licence. Two ways to use it:
(a) speed-ratio factors (measured / free-flow) per period and region or
distance band, applied to the OD times; (b) time-of-day speed profiles
per road link written into the OSM extract before routing, which needs
a match between NDW measurement sites and OSM ways.

## National network (feasibility)

The national OSM extract (`netherlands-260822.osm.pbf`, 1.4 GB) builds
in 453 s with an 11 GB heap and peaks at 10.6 GB resident on the 15 GB
development machine, so a national OSM-only network works (GTFS on top
is untested). Free-flow car times from Utrecht: Amsterdam 41, Rotterdam
48, Groningen 126, Maastricht 128 minutes.

## Not yet built

* **Distances and fares.** r5py's travel-time matrix has no distance.
  Fares by distance (NS tariff units, regional per-boarding plus
  per-km) need leg distances from detailed itineraries; the existing R
  project (`PT skim generator`) plans this for the reachable coarse
  pairs only.
* **List-type skim** for cheap/slow versus expensive/fast options per
  OD pair (the shared-bicycle case).
* **Shared-bicycle legs** (hubs, dockless supply) and the scenario
  variants S1-S4, which need their own networks.
* PT on the national GTFS feed (memory), and the choice of departure
  windows and percentile.
