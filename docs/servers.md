# Local routing servers (OSM and GTFS)

Both servers run from the project environment, without Docker or root
access. Their files live in `<root>/intermediate/valhalla` and
`<root>/intermediate/otp`; ports and memory settings are in `[servers]`.

    python -m ikob2.cli.servers valhalla build       # once: admin db + tiles
    python -m ikob2.cli.servers valhalla start [--concurrency 3]
    python -m ikob2.cli.servers valhalla status|stop
    python -m ikob2.cli.servers otp prepare          # links the pbf and GTFS zip, writes the build config
    python -m ikob2.cli.servers otp build            # graph (OSM + GTFS)
    python -m ikob2.cli.servers otp start|status|stop

On a 15 GB machine only one of the two servers (or an R5 skim build) fits
while a graph is being built: stop Valhalla before building the OTP graph.

## Valhalla (car route distances)

`pyvalhalla` ships the Valhalla executables (`pip install -e .[routing]`).
Tiles for the national extract build in about 3 minutes and take 0.9 GB.
The server answers `/route` and `/sources_to_targets` (a matrix with distance
and time for car, bicycle and pedestrian costings) on port 8002.

Long-distance matrices are memory-hungry, so the configuration rejects
matrix pairs beyond 80 km crow-fly (`valhalla_max_matrix_distance`) and more
than 20,000 pairs per request; batches of about 10 x 100 run at roughly 500
pairs per second using 2-4 GB. `cli.skims build-distance` therefore uses
Valhalla within 30 km and the detour model beyond (`skims.md`). A check
against the OSRM demo server (Utrecht Dom to Amsterdam / Groningen, car):
45.8 / 188.3 km against 45.9 / 188.4 km.

Valhalla's multimodal matrix is not used; public transport comes from the
frequency model.

## OpenTripPlanner (PT itineraries, validation)

OTP 2.8.1 (`otp-2.8.1-shaded.jar`, Java 21). `otp prepare` links the OSM
extract and the GTFS zip into the graph folder and writes a build config that
limits the transit service window (`otp_service_start`, `otp_service_end`);
`otp build` builds and saves `graph.obj`; `otp start` serves it on port 8080.
OTP answers itinerary requests through GraphQL (`POST /otp/gtfs/v1`),
wrapped by `otp_server.plan()` and `journey_summary()` (rail km, other km,
boardings, walk km). It has no origin-destination matrix service, so it
serves as an independent check on the frequency-model router.

**National graph.** Two reductions make a national graph fit in memory
without changing what PT routing needs:

1. `skims.osm_walk.make_walk_extract` writes a walking-network extract: ways
   a pedestrian can use (no motorway, trunk or service roads; `foot=no` and
   `foot=private` excluded), only the nodes those ways use, no relations
   (2.1 of 18.6 million ways; 152 MB instead of 1.4 GB).
2. `skims.gtfs_subset.subset_gtfs` keeps the services of the chosen dates
   (optionally within a bounding box): for one weekday nationally, 66,332
   stops, 120,546 trips and 2.2 million stop times.

The national graph builds in about 12 minutes with a 12 GB heap (`graph.obj`
651 MB) and is served with an 8 GB heap:

    python -m ikob2.cli.servers otp prepare --osm <walk.osm.pbf> \
        --gtfs <gtfs-day.zip> --service-start 2026-09-14 --service-end 2026-09-18
    python -m ikob2.cli.servers otp build --build-heap 12G
    python -m ikob2.cli.servers otp start --heap 8G

Because the walking graph has no roads, OTP is used only for walk + transit
itineraries.

### Validation of the frequency-model PT router

238 random pairs from Utrecht origins to destinations across the country;
OTP itineraries departing at 07:00, 07:30, 08:00, 08:30 and 09:00 on
2026-09-15 (averaged), compared with the store's `time`, `rail_km` and
`other_km`. Script and samples: `validation/pt_router_vs_otp.py` and
`validation/results/` (`validation/README.md`).

| | Frequency model vs OTP |
|---|---|
| Time correlation | 0.95 |
| Mean difference (model minus OTP) | -3.7 min |
| Median / mean absolute difference | -2.1 / 8.1 min |
| Rail km: correlation, mean difference | 0.98, +9.8 km |
| Other-transit km correlation | 0.82 |

By OTP travel time: 0-60 min +2.6 min (n = 12), 60-120 min -1.1 (118),
120-180 min -6.1 (100), beyond 180 min -20 (8). The frequency model is
faster on long journeys: it waits a fixed half-headway (capped at 7.5
minutes) and has no transfer penalty, while OTP follows the timetable,
including missed connections and minimum transfer times. Agreement is best
under two hours, where most jobs are. Rail kilometres are 10-15% longer than
OTP's, which suggests the rail detour factor (1.15) is high for the intercity
corridors of long journeys; since fares follow `rail_km`, long-distance rail
fares are biased upward accordingly.
