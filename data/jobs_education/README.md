# Jobs by education level per buurt (2016)

`jobs_by_education_buurt_2016.csv`: the jobs of 2016 per CBS buurt split by the
education level of the job, for 12,704 buurten of the 2016 buurt classification.
The model uses it as a covariate of the sector model that imputes jobs per buurt
and sector (`segments.jobs_impute`, `docs/data_lineage.md`): the share of jobs at
practical and at higher education level describes the kind of work a buurt holds.

## Origin and acknowledgement

Ralph and Sahar of the Municipality of Amsterdam determined, for all of the
Netherlands, the distribution of jobs by education level in 2016 from the LISA
data of that year. We thank them for this work and for making it available. The
table is published in the IKOB scripts repository of Stichting CROW
(https://github.com/Stichting-CROW/ikob-scripts, `segs/Databronnen SEGS
compleet/`, commit f90ac72 of 2026-04-08), whose `segs/README.md` describes it.
The file here is a lossless conversion of that spreadsheet,
`Ralph_Sahar_CBS_buurten_met_banen_naar_opleidingsniveau.xlsx` (sheet
`NL_CBS_buurten_met_banen_oplein`), to CSV; the MapInfo row identifier is dropped
and nothing else is changed. Read it with `float_precision="round_trip"`
(pandas) to get the spreadsheet's values to the last bit.

## Columns

| Column | Content |
|---|---|
| `BU_CODE`, `Gemeente`, `Buurtnaam` | CBS buurt code (2016), municipality, buurt name |
| `Aantal_Praktisch`, `Aantal_Middelbaar`, `Aantal_Theoretisch`, `Aantal_Onbekend` | jobs at practical, intermediate, theoretical (higher) and unknown education level; estimates, so not whole numbers |
| `Praktisch_Percentage`, `Middelbaar_Percentage`, `Theoretisch_Percentage`, `Onbekend_Percentage` | the same as percentages of all jobs of the buurt |
| `Praktisch`, `Middelbaar`, `Hoger` | shares of the jobs with a known education level (they sum to 1) |
| `Aaandeel_laag`, `Aandeel_middellaag`, `Aandeel_middelhoog`, `Aandeel_hoog` | the jobs of the buurt by income quartile, from the education shares and the matrix below (not used by the model) |

Income quartile by education level, as shares of all jobs (the matrix of the
spreadsheet, below the table):

| Income quartile | Practical, or benefit | Intermediate | Higher | Total |
|---|---|---|---|---|
| Low | 0.20 | 0.04 | 0.01 | 0.25 |
| Lower middle | 0.06 | 0.16 | 0.03 | 0.25 |
| Upper middle | 0.02 | 0.12 | 0.11 | 0.25 |
| High | 0.01 | 0.04 | 0.20 | 0.25 |
| Total | 0.29 | 0.36 | 0.35 | |

## Notes

* 49 buurt codes appear twice, with different buurt names and job counts (in
  Amsterdam, Amstelveen and other municipalities of Noord-Holland). The model
  reads the first record of each code (`segments.jobs_impute.parse_education_shares`).
  The choice is immaterial: adding the two records together instead moves 0.09%
  of the jobs of the municipality of Utrecht between its buurten (at most 11
  jobs in one buurt) and changes accessibility by less than 1e-5 relative (M2,
  scenarios S0 and S1), well below the spread from the rounding of the input
  data (`docs/scenarios.md`, precision).
* Licence: the CROW repository publishes the spreadsheet without a licence
  statement; it is redistributed here unchanged in content, with its source.
