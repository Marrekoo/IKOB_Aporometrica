# OV-fiets locations

`locaties.json`: the OV-fiets rental locations, retrieved from the openOV
feed http://fiets.openov.nl/locaties.json on 2026-09-20 and reduced to the
fields the model reads (`name`, `city`, `lat`, `lng`, keyed by location
code). Seeded into `inputs/ovfiets/`; read by `skims.hubs` with the tariff
kind `ovfiets`. The reduced file gives the same 301 hubs as the full feed.
