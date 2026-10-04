# Extra hubs of scenarios S2c and S2t

`utrecht_hubs_s2c.csv` (27 hubs, citywide) and `utrecht_hubs_s2t.csv` (12 hubs
in the 15 target buurten) are the output of `cli.hubs propose`
(`[siting]` parameters; `--within lime_price_zones_overvecht_kanaleneiland.csv
--label s2t --n-new 12` for S2t; `docs/scenarios.md`). They are seeded into
`intermediate/hubs/`, where the S2 skim modes read them, so the paper's skims
can be rebuilt with exactly these hubs. Columns: `hub, lat, lon, precision,
source` as in `data/hubs/`, and the siting inputs of the buurt the hub is
placed in: `buurtcode`, `access` (baseline PT accessibility), `bike_share`
(share of residents with a bicycle, `data/veh_owners/`) and `score`.
