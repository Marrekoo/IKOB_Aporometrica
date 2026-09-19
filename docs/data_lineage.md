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
| Jobs | 4 income groups, NRM + LISA 2016 education shares | LISA 2022 sectors imputed on buurten (`jobs_impute`) |
| Job-to-segment matching | one group per income class | sectors ranked by wage, partition over deciles (below) |
| Skims | NRM 2018, 2018 buurt codes | to be rebuilt (OSM/GTFS) |
| Fares | fare model on distance skim | unchanged (`FareModel`) |

## Sector jobs from LISA (`ikob2.segments.lisa`, `jobs_impute`)

LISA municipal data (BIJ12; peildatum 1 April; jobs rounded to tens;
2016-2025 on 2025 municipal boundaries; municipalities identified by
name; **15 LISA sectors**, not a finer SBI split) replaces the legacy
education-based income split. The year is a parameter (first run: 2022,
so jobs and the KWB 2022 segment populations describe the same year).

**Imputation onto buurten** (GSPREE-style, `python -m ikob2.cli.segments
jobs ...`):

1. Buurt job totals: a blend of the legacy 2018 shares and the KWB
   establishment shares (weight 0.25 on establishments, see below),
   rescaled so each municipality's buurten add up to the LISA total of
   the target year.
2. Structure model: per sector, a Poisson log-linear model with a
   log-total offset of the municipal sector composition (LISA 2016) on
   jobs-weighted municipal means of buurt covariates: the 2016 education
   mix of jobs (praktisch, hoger; Amsterdam LISA file), urbanisation
   class and ln house value (KWB 2022). Covariates are in natural units;
   buurt values are clipped to the range of the municipal means the model
   was fitted on (buurt values vary far more, and the model is
   exponential).
3. Seed: each LISA sector is weighted by the buurt's establishments in
   its KWB group (below); the covariate model only divides a group over
   its LISA sectors. Then IPF per municipality to the buurt totals
   (rows) and the LISA sector totals (columns). Without establishments
   the seed is the covariate model alone.

First run (2022, all 342 municipalities): 9,426,120 jobs imputed = LISA
total; every municipality converges; largest marginal error 1e-8 jobs;
14,411 buurten (one 'Buitenland' buurt has no municipality). The model
reduces the job-weighted total-variation distance to the true municipal
composition from 0.184 (national average) to 0.132. 5,330 buurten had
incomplete covariates (filled with the municipal or national mean) and
3,614 were clipped to the fitted range.

### Establishments per buurt (KWB)

The KWB gpkg only has the total number of establishments; the split by
SBI group is in the StatLine KWB tables (85318NED = 2022) in eight
groups: A, B-F, G+I, H+J, K-L, M-N, O-Q, R-U (`cli.segments fetch`
stores a snapshot). They are rounded, and the group cells of about 14% of
buurten are suppressed while the total is known; those cells are filled
from the municipality's composition (`establishments.complete_group_counts`).

*Compatibility.* National KWB/LISA establishment ratios for the groups
that map cleanly are 0.97 (A), 1.05 (B-F), 1.09 (G+I), 1.01 (H+J), 1.08
(O-Q) and 1.02 (R-U); KWB M-N against LISA L10 is 1.03. KWB's K-L group
is 182k establishments but LISA's finance (L09) only 17.7k: real estate
(L) is essentially absent from LISA, so K-L is a weak proxy for L09 jobs,
and L10 is taken to be M+N only (`lisa.SECTOR_TO_SBI`).

*Validation of the buurt totals.* LISA 2016 jobs per buurt exist (the
education file, 2016 codes; 10,137 buurten in 304 municipalities in
common with the 2016 KWB and the legacy table). Job-weighted total
variation distance between predicted and true buurt job shares within
municipalities (`jobs_impute.within_municipality_tv`, lower is better):

| Predictor of buurt job share | distance |
|---|---|
| uniform | 0.454 |
| legacy 2018 (NRM) | 0.301 |
| KWB 2016 establishments, total | 0.338 |
| establishments weighted by job size per group | 0.365 |
| 75% legacy + 25% establishments (total) | **0.277** |
| 50% / 50% | 0.280 |
| 25% legacy + 75% establishments | 0.301 |

The table comes from a one-off analysis (KWB 2016 establishments via
`fetch --kwb-table 83487NED`, which lacks the O-Q group; it was derived as
total minus the other groups), not from a script in the repository.

So the legacy shares are better within municipalities than raw
establishment counts, but the blend beats both (8% lower distance than
legacy alone). Weighting establishments by a job size per group made it
worse (the O-Q group, with about 65 jobs per establishment, dominates),
so it is not used. The weight 0.25 is the default (`--establishment-
weight`); the flat optimum lies between 0.25 and 0.5. Caveat: the
validation compares 2016 truth with 2018 legacy shares and 2016
establishments, and 17% of buurten had suppressed group cells (filled with
zero in the test).

*Not validated:* the sector placement. No buurt-level sector truth exists;
the establishment-weighted seed is plausible (it moves, for example, a
sector's jobs to the buurten that have its establishments) but unchecked.
On the 2022 run the buurt totals shift on average by 138 jobs (95th
percentile 447) relative to the covariate-only seed, and all marginals stay
exact.

**Assumptions and limits (read before using the result):**
* Only the municipal marginals are data. The split of sectors over
  buurten inside a municipality is the model's transfer of a
  between-municipality relation (ecological inference); no buurt-level
  truth exists to check it. 2016 education shares are assumed to
  describe 2022.
* 80% of 2022 buurten have education data (codes changed since 2016).
* Municipality names are matched exactly plus an alias table
  (`GEMEENTE_ALIASES`: '(L.)'-type suffixes, the 2023 merger into Voorne
  aan Zee, Weesp into Amsterdam, a truncated Nuenen name).
* LISA rows absent for a municipality x sector are treated as 0 (tiny
  sectors in small municipalities; rows add up to the total within
  rounding).

## Sector -> income level (`jobs.sector_income_weights`)

Wage per LISA sector comes from CBS 81431NED (employee jobs and mean
hourly wage by SBI2008 section, 2022), the job-weighted mean over the
SBI sections of each sector (`lisa.SECTOR_TO_SBI`; **assumed** to match
LISA's sector definitions, verify against LISA documentation; L10 is
M+N, see above). Sectors
are ranked by wage and laid along the income-rank axis in proportion to
their national jobs; decile Dk covers [(k-1)/10, k/10]; each sector's
jobs are spread over the deciles it overlaps in proportion to the
overlap. The decile pools therefore **partition** the jobs (they add up
to the total), which is the D_{j,s} of the paper. `onbekend` sees all
jobs.

(An earlier four-group quantile matching gave overlapping pools, as every
decile inside a group saw the whole group; it was removed.)

Limits: within-sector wage dispersion is ignored (every job of a sector
sits at its mean-wage rank), so e.g. financial services is entirely at
the top and hospitality entirely at the bottom. A wage distribution per
sector (rather than a mean) would soften this; 81431NED only has means.
