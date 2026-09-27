# Jobs: sources, imputation and matching

How the opportunities `D[j, c, w]` are built: jobs per LISA sector per buurt
(`segments.lisa`, `segments.jobs_impute`, `segments.establishments`), their
income class (`segments.jobs`) and their job type (`segments.wfh`).

## Sources

| Source | Content | Role |
|---|---|---|
| LISA municipal data (`LISA_Gemeenten_2025.xlsx`) | jobs per municipality in 15 LISA sectors, peildatum 1 April, rounded to tens, 2016-2025 on 2025 municipal boundaries, municipalities identified by name | the only observed job counts; column margins of the imputation |
| IKOB job table (`Alle_Zones_2030_2040.xlsx`) | jobs per 2022 buurt, derived from NRM zone totals assigned to buurten by centroid (2018 values used) | buurt job totals |
| KWB establishments (StatLine 85318NED, 2022) | establishments per buurt in 8 SBI groups | where each sector's establishments are |
| LISA 2016 jobs by education (Amsterdam file) | jobs per 2016 buurt by education level | covariate of the sector model |
| KWB 2022 | urbanisation class, mean house value | covariates of the sector model |
| CBS 81431NED (2022) | employee jobs and mean hourly wage per SBI section | sector -> income rank |
| CBS 85718NED (2024), 82072NED (2010) | working from home by education; education mix per SBI section | sector -> job type |

The year is a parameter (`accessibility.jobs_year`, `--year`); 2022 matches
the KWB 2022 segment populations.

## Imputation onto buurten (`cli.segments jobs`)

Known: jobs per municipality x sector (LISA) and a job total per buurt.
Unknown: how each municipality's sector jobs are spread over its buurten.

1. **Buurt totals.** A blend of the IKOB table's buurt shares and the KWB
   establishment shares (weight `jobs.establishment_weight` = 0.25 on
   establishments), rescaled so that each municipality's buurten add up to
   its LISA total of the target year.
2. **Structure model.** Per sector, a Poisson log-linear model with a
   log-total offset, fitted on municipalities: the 2016 LISA sector
   composition on jobs-weighted municipal means of buurt covariates (the 2016
   education mix of jobs, praktisch and hoger; urbanisation class; ln house
   value). Covariates are in natural units; buurt values are clipped to the
   range of the municipal means the model was fitted on.
3. **Seed and IPF.** Each LISA sector's seed in a buurt is the buurt's
   establishments in the sector's KWB group; the covariate model divides a
   group over its LISA sectors. Without establishments the seed is the
   covariate model alone. IPF per municipality then fits the seed to the buurt
   totals (rows) and the LISA sector totals (columns) (`jobs.tol`,
   `jobs.max_iter`).

Output: `buurtcode, L01..L15`. The imputed total equals the LISA total, all
municipalities converge, and marginal errors are below 1e-8 jobs. Buurten
with incomplete covariates get the municipal or national mean.

### Establishments (`segments.establishments`)

KWB counts establishments in eight SBI groups: A, B-F, G+I, H+J, K-L, M-N,
O-Q, R-U. Counts are rounded, and in about 14% of buurten the group cells are
suppressed while the total is known; those cells are filled from the
municipality's composition (`complete_group_counts`).

National KWB/LISA establishment ratios are 0.97 (A), 1.05 (B-F), 1.09 (G+I),
1.01 (H+J), 1.08 (O-Q), 1.02 (R-U), and 1.03 for KWB M-N against LISA L10.
KWB's K-L group (182k establishments) is far larger than LISA finance (L09,
17.7k), because real estate (L) is essentially absent from LISA; K-L is
therefore a weak proxy for L09, and L10 is taken to be M+N only
(`lisa.SECTOR_TO_SBI`).

### Validation of the buurt totals

LISA 2016 jobs per buurt exist in the education file (10,137 buurten in 304
municipalities in common with KWB 2016 and the IKOB table). The job-weighted
total-variation distance between predicted and true buurt job shares within
municipalities (`jobs_impute.within_municipality_tv`, lower is better):

| Predictor of the buurt job share | Distance |
|---|---|
| uniform | 0.454 |
| IKOB table (NRM-based) | 0.301 |
| KWB 2016 establishments, total | 0.338 |
| establishments weighted by job size per group | 0.365 |
| 75% IKOB table + 25% establishments | **0.277** |
| 50% / 50% | 0.280 |
| 25% IKOB table + 75% establishments | 0.301 |

The blend beats both components; weighting establishments by job size makes
it worse (the O-Q group, about 65 jobs per establishment, dominates), so it
is not used. The optimum is flat between 0.25 and 0.5. The comparison uses
2016 truth against 2018 totals and 2016 establishments, and was computed
with a one-off script (KWB 2016 via `fetch --kwb-table 83487NED`) that is not
part of the package.

The placement of *sectors* within a municipality has no buurt-level truth
and is not validated.

### Assumptions and limits

* Only the municipal margins are data; the split of sectors over buurten is
  the model's transfer of a between-municipality relation (ecological
  inference). The 2016 education shares are assumed to describe 2022.
* About 80% of 2022 buurten have education data (codes changed since 2016).
* Municipality names are matched exactly, plus an alias table
  (`GEMEENTE_ALIASES`: '(L.)'-type suffixes, the Voorne aan Zee merger, Weesp
  into Amsterdam, a truncated Nuenen name).
* LISA rows absent for a municipality x sector are 0.
* One buurt ('Buitenland') has no municipality and gets no jobs.

## Sector -> income class (`jobs.sector_income_weights`, `sector_pools`)

The wage of a LISA sector is the job-weighted mean hourly wage (81431NED) of
its SBI sections (`lisa.SECTOR_TO_SBI`, an assumed correspondence; L10 is
M+N). Sectors are ranked by wage and laid along the income-rank axis in
proportion to their national jobs; decile `Dk` covers `[(k-1)/10, k/10]`, and
each sector's jobs are spread over the deciles it overlaps in proportion to
the overlap. The decile pools therefore **partition** the jobs: they add up
to the total. `onbekend` sees all jobs. `--common-jobs` gives every segment
all jobs instead (the controlled comparison).

Limit: within-sector wage dispersion is ignored; every job of a sector sits
at its mean-wage rank (financial services entirely at the top, hospitality
entirely at the bottom). 81431NED publishes means only.

## Job type: admits working from home (`segments.wfh`)

The time margins differ by whether the job admits working from home. Per SBI
section:

    share_wfh(section) = sum_e P(e | section) x incidence(e)

with `incidence(e)` the share of employed people with education `e` who at
least sometimes work from home (85718NED, 2024) and `P(e | section)` the
education mix of employee jobs (82072NED, 2010, the only year published);
a LISA sector averages its sections. `split_jobs_by_wfh` splits every
sector's jobs into `no_wfh` and `wfh_possible`.

The derivation compresses the range (33% for agriculture and hospitality to
68% for education; 45% overall against 52% of workers), ignores occupation,
and reads "at least sometimes" as "admits".
