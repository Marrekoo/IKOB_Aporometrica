# Legacy IKOB data: where it comes from, what it is used for

Reconstructed from the `ikob-scripts` repository (scripts and the
`Beschrijving`/`Toelichting` documents in it) and from inspecting its
files. The scripts were read, not re-run; that repository states that
they are not maintained and may be broken.

## Skims (travel times, distances)

* Source: NRM (Nationaal Regionaal Model) zone-to-zone skims from
  Rijkswaterstaat, per region (Noord, Oost, West, Zuid) and year (2018,
  2030H, 2040L, 2040H). Files per mode: car (free-flow, morning and
  evening peak, plus distance), bike (morning/evening peak), public
  transport (morning/evening peak, plus `ov_afstanden`). OV travel
  times/distances are, per the description, best computed by Goudappel.
* `Van lijst naar matrix.py` turns the list format (`from;to;value`)
  into NRM-zone matrices; `Van NRM naar IKOB nw.py` maps NRM zones to
  IKOB zones through the `IKOBxx_Omnummer.csv` tables (26 IKOB areas).
* `national_skim.py` builds one buurt x buurt matrix over 2018 buurt
  codes (13,404 buurten; 33,668 matrix rows because a buurt can occur
  once per region). It has been run for car only: morning-peak travel
  time and free-flow distance.
* **Not used here.** The skims will be rebuilt from OSM (car, bike) and
  GTFS (public transport); walking from zone size; fares from GTFS by
  distance. The legacy skims are 2018-coded: 11,459 of the 14,412 2022
  buurten have a row (767 of 869 in a 23-municipality Utrecht proxy).

## Jobs (opportunities)

* Totals per NRM zone (2018, 2030H, 2040L, 2040H) are assigned to
  buurten by centroid: a zone containing several buurt centroids splits
  its jobs in proportion; a zone containing none adds its jobs to the
  buurt containing the zone centroid.
* Split into four income groups (`laag`, `middellaag`, `middelhoog`,
  `hoog`) with shares derived from a **2016 LISA** distribution of jobs
  by education level (Ralph and Sahar, Municipality of Amsterdam; newer
  LISA is expensive). The 2016 -> 2022 buurt mapping is in
  `Buurt naar inkomensverdeling arbeidsplaatsen.xlsx`.
* Result: `Alle_Zones_2030_2040.xlsx`, sheet `buurten-arbeidsplaatsen`,
  jobs per buurt per income group and year. Its codes match the 2022 KWB
  buurten exactly (14,327 rows); the 2018 total is about 8.66 million
  jobs. This is what `ikob2.segments.jobs` reads.
* Caveat: income group of a job is inferred from education level, not
  observed; the job groups are not the same thing as population income
  deciles.

## Population segments

* Legacy: four income classes per zone from KWB 2022 (household income
  distribution, urbanisation, inhabitants; labour force derived), in
  `Beroepsbevolking_inkomensgroep` files. Car ownership and
  preferences follow from urbanisation class through five fixed
  tables. Replaced here by the 44 household-type x income segments
  (`docs/segments.md`).

## What replaces what

| Input | Legacy | In `ikob2` |
|---|---|---|
| Segment populations | 4 income classes, KWB 2022 | 44 segments (`ikob2.segments`) |
| Jobs | 4 income groups, NRM + LISA 2016 education shares | `ikob2.segments.jobs`, interim |
| Job-to-segment matching | one group per income class | quantile matching (below) |
| Skims | NRM 2018, 2018 buurt codes | to be rebuilt (OSM/GTFS) |
| Fares | fare model on distance skim | unchanged (`FareModel`) |

### Quantile matching (interim assumption)

The four job groups partition the income-rank axis [0, 1] in proportion
to their national job shares, `laag` lowest. Income decile Dk covers the
rank interval [(k-1)/10, k/10]; the share of that interval inside job
group g is the weight W[k, g], and the decile's opportunity vector is
`sum_g W[k, g] * jobs_g` (rows of W sum to 1). `onbekend` has no rank
and sees all jobs. With the legacy national totals D1-D2 map almost
entirely to `laag` and D9-D10 to `hoog`; deciles on a group boundary are
split. This is replaced when LISA by industry (SBI) is available.

## Planned: refit jobs to current marginals by SBI

The intention is to actualise the jobs by refitting them to current
marginals for SBI codes that match the job types. Not implemented; the
open points to settle and document are: which marginals (year, spatial
level), which SBI-to-income-group correspondence, and how the LISA 2016
education shares are superseded.
