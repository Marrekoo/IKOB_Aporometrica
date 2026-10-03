# Occupation tables of the job matching

Inputs of `ikob2.segments.occupations`, which places the jobs of each LISA
sector in the income-decile pools and splits them by whether they admit
working from home (method: `docs/data_lineage.md`). `cli.layout create`
copies them to `inputs/occupations/` of a data folder. Retrieved 3 October
2026 (`sources.json`).

| File | Content | Source |
|---|---|---|
| `lfs_nace_isco_nl.csv` | `nace_r2, isco08, employed_thousands`: employed persons aged 15 or over, Netherlands, 2022, by NACE Rev. 2 section and ISCO-08 major group (OC0 armed forces is in the file and not used) | Eurostat, `lfsa_eisn2` |
| `ses_hourly_nace_isco_nl.csv` | `nace_r2, isco08, mean_hourly_eur`: mean gross hourly earnings, Netherlands, 2022, all enterprise sizes, both sexes; sections B-S (no agriculture); blank where not published | Eurostat, Structure of Earnings Survey 2022, `earn_ses22_47` |
| `lfs_isco2_nl.csv` | `isco08, employed_thousands`: employed persons aged 15 or over, Netherlands, 2022, by ISCO-08 2-digit group | Eurostat, `lfsa_egai2d` |
| `brc_wage_quartiles.csv` | `brc, title, employees_thousands, p25, p50, p75`: employees aged 15-75 and quartiles of their gross hourly wage (EUR), 2022, per BRC 2014 occupation (class, segment and group levels); blank where not published | CBS StatLine 85517NED |
| `isco_brc_crosswalk.csv` | `isco08_unit, isco08_unit_label, brc_group, brc_group_label, brc_class`: the BRC 2014 occupation group of every ISCO-08 unit group | CBS, Schakelschema ISCO2008 - BRC2014 (sheet "BRC 2014 variabelen") |
| `teleworkability_isco3.csv` | `isco08, title, physical_interaction, social_interaction` per ISCO-08 3-digit group; `physical_interaction` is the share of the group that can potentially work remotely (technical teleworkability) | Sostero et al. (2020), Zenodo |

## Reproducing

    python -m ikob2.cli.segments occupations --out data/occupations

downloads every table from its public API or file and writes `sources.json`:
per file the exact query URL, the time of retrieval (UTC) and the SHA-256 of
the CSV. Reading the CBS correspondence (an Excel 97 file) needs `xlrd`
(extra `legacy`). A refetch on 2026-10-03 reproduced the files here
byte for byte. The queries:

| File | Query |
|---|---|
| `lfs_nace_isco_nl.csv` | https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data/lfsa_eisn2?geo=NL&sex=T&age=Y_GE15&unit=THS_PER&time=2022&format=JSON&lang=EN |
| `ses_hourly_nace_isco_nl.csv` | https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data/earn_ses22_47?geo=NL&sex=T&indic_se=ERN&sizeclas=TOTAL&unit=EUR&format=JSON&lang=EN |
| `lfs_isco2_nl.csv` | https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data/lfsa_egai2d?geo=NL&sex=T&age=Y_GE15&unit=THS_PER&time=2022&format=JSON&lang=EN |
| `brc_wage_quartiles.csv` | https://opendata.cbs.nl/ODataApi/odata/85517NED/TypedDataSet?%24select=Beroep%2CPerioden%2CWerknemer_1%2Ck_25ePercentiel_2%2Ck_50ePercentielMediaan_3%2Ck_75ePercentiel_4&%24format=json&%24filter=Perioden%20eq%20%272022JJ00%27 |
| `isco_brc_crosswalk.csv` | https://www.cbs.nl/-/media/imported/onze-diensten/methoden/classificaties/documents/2015/09/schakelschema-isco2008-brc2014.xls |
| `teleworkability_isco3.csv` | https://zenodo.org/api/records/7716456/files/Telework%20ISCO%20indices.csv/content |

Eurostat and CBS revise published figures now and then; compare the SHA-256
in `sources.json` after a refetch. A new data folder gets the tables through
`python -m ikob2.cli.layout --root <root> create`, which never overwrites a
file: replace `inputs/occupations` of an existing folder by hand.

## Licences

Eurostat data may be reused with acknowledgement of the source (Commission
Decision 2011/833/EU). CBS StatLine tables and the CBS correspondence are
published under CC BY 4.0. The teleworkability indices are published under
CC BY 4.0.

## References

Eurostat (2026a). *Employed persons by occupation and economic activity
(NACE Rev. 2)* [lfsa_eisn2]. https://doi.org/10.2908/lfsa_eisn2

Eurostat (2026b). *Employed persons by detailed occupation (ISCO-08 two digit
level)* [lfsa_egai2d]. https://doi.org/10.2908/lfsa_egai2d

Eurostat (2026c). *Mean hourly earnings by sex, occupation and economic
activity (2022)* [earn_ses22_47], Structure of Earnings Survey 2022.
https://doi.org/10.2908/earn_ses22_47

Statistics Netherlands (2025). *Werknemers; uurloon en beroep (2013-2024)*
[85517NED]. StatLine. https://opendata.cbs.nl/statline/#/CBS/nl/dataset/85517NED

Statistics Netherlands (2015). *Schakelschema ISCO2008 - BRC2014*. CBS,
Den Haag / Heerlen.

ROA & CBS (2015). *Beroepenindeling ROA-CBS 2014 (BRC 2014)*. ROA Technical
Report ROA-TR-2015/5. Research Centre for Education and the Labour Market,
Maastricht University.

Sostero, M., Milasi, S., Hurley, J., Fernández-Macías, E., & Bisello, M.
(2020). *Teleworkability and the COVID-19 crisis: a new digital divide?* JRC
Working Papers Series on Labour, Education and Technology 2020/05, JRC121193.
European Commission, Seville.

Sostero, M., Fernández-Macías, E., Milasi, S., Bisello, M., & Hurley, J.
(2020). *Occupation teleworkability indices* [Data set]. Zenodo.
https://doi.org/10.5281/zenodo.7716456
