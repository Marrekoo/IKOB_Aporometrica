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

OTP 2.10 is a jar (`otp-2.10.0-shaded.jar`, Java 21). `otp prepare` links
the OSM extract and the GTFS zip into the graph folder and writes a
build config that limits the transit service window (default September
2026) to keep memory down; `otp build` builds and saves `graph.obj`;
`otp start` serves it on port 8080. OTP answers itinerary requests (legs,
modes, distances, times), which is what fares by distance need. It has no
origin-destination matrix service, so bulk PT travel times stay with R5
(r5py).

Memory: only one of the two servers (or an R5 skim run) fits next to the
other on a 15 GB machine while a graph is being built; stop Valhalla
before building the OTP graph.
