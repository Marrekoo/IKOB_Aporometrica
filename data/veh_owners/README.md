# Bicycle ownership per buurt (Utrecht)

`bike_ownership_buurten.csv`: one row per buurt of the municipality of
Utrecht (111): `buurtcode, buurtnaam, wijkcode, wijknaam, aantal_inwoners,
buurtteam, pct_with_bicycle` (0-100) and `mapping_confidence`.

Source: the Utrecht buurtteam survey 2025 (Municipality of Utrecht), which
reports the share of respondents with a bicycle per buurtteam area; each
buurt takes the share of its buurtteam area, assigned by hand
(`mapping_confidence` high / medium / low). Seeded into `inputs/veh_owners/`;
read by `segments.ownership.load_bike_ownership` (`--ownership`) and by the
hub siting (`cli.hubs propose`).
