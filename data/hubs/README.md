# Shared-bicycle hubs of Utrecht

`utrecht_hubs.csv`: the 27 municipal shared-bicycle hubs of Utrecht (Lime
egress points in scenario S0), one row per hub: `hub` (name), `lat`, `lon`
(WGS84), `precision` (how the point was located: `station`, `intersection`,
...) and `source` (the evidence for the location: an OV-fiets location, a
PDOK road intersection, ...). Seeded into `inputs/hubs/`; read by
`skims.hubs` with the tariff kind `lime` (`pt.hub_files`, `pt.hub_kinds`).

Source: the hub list of the Municipality of Utrecht, located on PDOK road
geometries and OV-fiets locations (column `source`).

The extra hubs of scenarios S2c and S2t are in `data/hubs_scenarios/`
(README there).
