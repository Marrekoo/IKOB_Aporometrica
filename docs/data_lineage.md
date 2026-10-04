# Jobs: sources, imputation and matching

How the opportunities `D[j, c, w]` are built: jobs per LISA sector per buurt
(`segments.lisa`, `segments.jobs_impute`, `segments.establishments`), and
their income class and job type (`segments.occupations`).

## Sources

| Source | Content | Role |
|---|---|---|
| LISA municipal data (`LISA_Gemeenten_2025.xlsx`) | jobs per municipality in 15 LISA sectors, peildatum 1 April, rounded to tens, 2016-2025 on 2025 municipal boundaries, municipalities identified by name | the only observed job counts; column margins of the imputation |
| IKOB job table (`Alle_Zones_2030_2040.xlsx`, Stichting CROW, ikob-scripts) | jobs per 2022 buurt, derived from NRM zone totals assigned to buurten by centroid (2018 values used) | buurt job totals |
| KWB establishments (StatLine 85318NED, 2022) | establishments per buurt in 8 SBI groups | where each sector's establishments are |
| Jobs by education level per buurt, 2016 (`data/jobs_education`, Municipality of Amsterdam) | jobs per 2016 buurt by education level | covariate of the sector model |
| KWB 2022 | urbanisation class, mean house value | covariates of the sector model |
| Eurostat LFS `lfsa_eisn2` (NL, 2022) | employed persons by NACE section x ISCO-08 major group | occupations within a sector |
| Eurostat SES 2022 `earn_ses22_47` (NL) | mean hourly earnings by NACE section x ISCO-08 major group | wage of a sector x occupation cell |
| CBS 85517NED (2022) | employees and hourly wage quartiles per BRC 2014 occupation group | wage spread within a cell |
| CBS ISCO 2008 - BRC 2014 correspondence | ISCO unit groups per BRC occupation group | quartiles -> ISCO major groups |
| Sostero et al. (2020), Zenodo 10.5281/zenodo.7716456 | technical teleworkability per ISCO 3-digit group | job type (home working) |
| Eurostat LFS `lfsa_egai2d` (NL, 2022) | employed persons by ISCO-08 2-digit group | weights of the teleworkability |

The tables are in `data/occupations` (README there: sources, licences,
retrieval) and are seeded into `inputs/occupations`.

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
2016 truth against 2018 totals and 2016 establishments. The script is
`validation/jobs_buurt_totals.py` (KWB 2016 via 83487NED), which follows
this description; `validation/README.md` gives how far it reproduces the
table.

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

## Income class and job type (`segments.occupations`)

**Cells.** Each LISA sector (`lisa.SECTOR_TO_NACE`, the sections of
`lisa.SECTOR_TO_SBI`) is split into the nine ISCO-08 major groups (armed
forces excluded) in proportion to their national employment in its NACE
sections (LFS). A cell's jobs are that share of the sector's national LISA
jobs. Its mean hourly wage is the employment-weighted SES mean over the
sector's sections; a cell no section publishes takes the occupation's
employment-weighted mean over all sections (agriculture is outside the SES).

**Wages within a cell** are lognormal with the cell's mean and one common log
standard deviation `sigma` (0.380). `sigma` comes from the 85517NED quartiles:
each BRC group is a lognormal with its median and interquartile range
(`sd = ln(P75/P25)/1.349`; a group without quartiles takes those of the level
above), its employees split evenly over its ISCO unit groups; per ISCO major
group the spread of the mixture of its groups, averaged over the major groups
by employees.

**Deciles.** The job-weighted mixture of all cells is the national wage
distribution of jobs. Its deciles (EUR 10.0, 12.7, 15.0, 17.3, 19.8, 22.6,
25.9, 30.4, 37.6 per hour) partition every cell:
`W[k, c] = Phi((ln b_k - mu_c)/sigma) - Phi((ln b_{k-1} - mu_c)/sigma)`,
`mu_c = ln(mean wage) - sigma^2/2`. A sector's weight in decile `k` is the
sum over its cells, weighted by their share of its jobs, so every sector
reaches several deciles and each decile pool holds a tenth of the jobs. The
pools **partition** the jobs; `onbekend` sees all jobs; `--common-jobs` gives
every segment all jobs (the controlled comparison).

**Job type.** A job admits working from home with the teleworkability of its
occupation: the 'physical interaction' indicator of Sostero et al. (2020), the
share of an ISCO 3-digit group that can potentially work remotely, averaged to
2-digit groups and weighted to major groups by Dutch employment (OC1 0.76,
OC2 0.77, OC3 0.51, OC4 0.83, OC5 0.12, OC6-OC9 0.00-0.01). The share
therefore varies by decile through the occupations in it, from 14% of the D1
jobs to 70% of the D10 jobs. The weights by job type are written to
`job_weights.csv` of a run.

Limits: income deciles of households are matched to wage deciles of jobs
(two-earner and part-time households loosen that link); the occupation mix of
a sector and the wage spread are national; the LFS counts all employed
persons, the SES employees; teleworkability is a European estimate of what is
technically possible, not of practice.

Why cells and a spread: a sector's jobs cover a wide range of wages (in
manufacturing the mean hourly wage runs from EUR 13.50 for elementary
occupations to EUR 42.50 for managers), so placing every job of a sector at
one wage would give each decile one to three sectors and pools that jump from
decile to decile with the sectors' sizes. Within-cell spread lets every cell
straddle several deciles.
