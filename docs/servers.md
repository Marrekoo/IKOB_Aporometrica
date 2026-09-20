# Local routing servers (OSM and GTFS)

Everything runs from the project environment: no Docker and no sudo
(neither is available on the development machine). Files live in
`<data>/intermediate/valhalla` and `<data>/intermediate/otp`.

    python -m ikob2.cli.servers valhalla build       # once: admin db + tiles
    python -m ikob2.cli.servers valhalla start --concurrency 3
    python -m ikob2.cli.servers valhalla status|stop
    python -m ikob2.cli.servers otp prepare          # links the pbf and gtfs zip
    python -m ikob2.cli.servers otp build            # graph (OSM + GTFS)
    python -m ikob2.cli.servers otp start|status|stop

## Valhalla (OSM: car, bicycle, pedestrian)

`pyvalhalla` ships the Valhalla executables (`pip install -e .[routing]`).
Tiles for the national extract build in about 3 minutes and take 0.9 GB.
The server answers `/route` and `/sources_to_targets` (matrix with
distance and time) on port 8002. The default service limits are raised
for local use, but a matrix over long distances is memory-hungry: a
25 x 1000 national request took the server to 10 GB and it was killed by
the operating system, and so did 10 x 200 pairs within 150 km. The
configuration therefore rejects matrix pairs beyond 80 km crow-fly and
20,000 pairs per request; batches of about 10 x 100 run at roughly 500
pairs per second using 2-4 GB. Distances further out come from the detour
model.

Check against the OSRM demo server (Utrecht Dom to Amsterdam / Groningen,
car): 45.8 / 188.3 km against 45.9 / 188.4 km.

`python -m ikob2.cli.skims build-distance <store> --kwb ... --detour ...`
fills the store's car `distance` variable: Valhalla within 30 km (crow-fly)
of an origin, the detour model beyond and for the municipality layer, and
the detour estimate wherever Valhalla finds no route. The accessibility
run uses stored distances automatically.

Valhalla's multimodal (transit) matrix is not supported, so public
transport is not served here.

## OpenTripPlanner (OSM + GTFS: public transport itineraries)

OTP 2.8.1 is a jar (`otp-2.8.1-shaded.jar`, Java 21; 2.9 and later need Java 25). `otp prepare` links
the OSM extract and the GTFS zip into the graph folder and writes a
build config that limits the transit service window (default September
2026) to keep memory down; `otp build` builds and saves `graph.obj`;
`otp start` serves it on port 8080. OTP answers itinerary requests (legs,
modes, distances, times), which is what fares by distance need. It has no
origin-destination matrix service, so the bulk PT matrices come from the
frequency-model router (`skims.gtfs_pt`, see `skims.md`) and OTP serves as
its independent check.

**National coverage.** A graph from the full national OSM extract does not
fit in 15 GB (the build was killed at 10.5 GB). Two reductions make the
national graph fit, without changing what public transport routing needs:

1. `skims/osm_walk.py` writes a *walking-network* extract: ways a pedestrian
   can use (no motorway, trunk, service roads; `foot=no/private` excluded),
   only the nodes those ways use, no relations. 2.1 million of 18.6 million
   ways remain (152 MB instead of 1.4 GB).
2. `skims/gtfs_subset.py` keeps the services of one weekday (2026-09-15)
   for the whole country: 66,332 stops, 120,546 trips, 2.2 million stop
   times.

The national graph then builds in about 12 minutes at a 12G heap
(`graph.obj` 651 MB) and is served with an 8G heap:

    python -m ikob2.cli.servers otp prepare --osm <walk.osm.pbf> \
        --gtfs <gtfs-day.zip> --service-start 2026-09-14 --service-end 2026-09-18
    python -m ikob2.cli.servers otp build --build-heap 12G
    python -m ikob2.cli.servers otp start --heap 8G

The same commands with province extracts and `--bbox` give the earlier
regional graph. The REST plan endpoint is absent in 2.8.1; use GraphQL
(`POST /otp/gtfs/v1`), wrapped by `otp_server.plan()` and
`journey_summary()` (rail km, other km, boardings, walk km). Because the
walk graph has no roads, OTP is only used for public transport itineraries
(walk + transit), never for car or bicycle.

### Validation of the frequency-model PT router against OTP

Random Utrecht origins and destinations, OTP itineraries departing
07:00, 07:30, 08:00, 08:30, 09:00 on 2026-09-15 (average), compared with
the skim store's `time`, `rail_km` and `other_km`
(the comparison script is not part of the package).

| | Regional graph (124 pairs, destinations within 3.6-5.85E, 51.75-52.95N) | National graph (238 pairs, whole country) |
|---|---|---|
| Time correlation | 0.97 | 0.95 |
| Frequency model minus OTP, mean | -2.0 min | -3.7 min |
| Median / mean absolute difference | -1.3 / 5.6 min | -2.1 / 8.1 min |
| Rail km correlation, mean difference | 0.92, +7.6 km | 0.98, +9.8 km |
| Other-transit km correlation | 0.80 | 0.82 |

By OTP travel time (national): 0-60 min +2.6 min (n=12), 60-120 min -1.1
(118), 120-180 min -6.1 (100), beyond 180 min -20 (8). The frequency model
is faster on long trips. Likely causes (not isolated): the model waits a
fixed half-headway (capped at 7.5 min) and charges no transfer penalty,
while OTP follows the real timetable, including missed connections and
minimum transfer times. Agreement is best under two hours, where most
jobs are. Rail km in the model are on average 10-15% longer than OTP's; the model
scales crow-fly distances by a rail detour factor (`--rail-detour`, default
1.15), which may be too high for the intercity corridors dominating long
trips. Fares follow `rail_km`, so they are biased upward accordingly.

Memory: only one of the two servers (or an R5 skim run) fits next to the
other on a 15 GB machine while a graph is being built; stop Valhalla
before building the OTP graph.
