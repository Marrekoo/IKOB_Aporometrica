"""
CBS StatLine access for the segment pipeline.

The tables are small (86161NED, 71487ned, 81431ned, the KWB establishments
table, 85718NED, 82072NED), so they are downloaded once and kept as CSV
snapshots (<data root>/cache/statline, or data/statline/ in the
repository). Runs then read
the snapshots and are offline and reproducible; the snapshot files are
the exact inputs a reviewer can inspect. `fetch_*` is the only network
code and is only called by the `fetch` CLI subcommand.

The classic OData "TypedDataSet" endpoint is used rather than the
`cbsodata` package because it returns dimension KEYS (e.g. 'GM0363',
'2022JJ00', '1050015'), which is what the pipeline joins on;
`cbsodata` returns display titles ('Amsterdam', '2022') instead.
Keys come back space-padded ('GM0363  '), so every key is stripped.
"""

from __future__ import annotations

import json
import logging
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

ODATA_ROOT = "https://opendata.cbs.nl/ODataApi/odata"
# The Feed endpoint pages large result sets with nextLink instead of
# refusing queries that could return >= 10 000 rows.
ODATA_FEED_ROOT = "https://opendata.cbs.nl/ODataFeed/odata"

INCOME_TABLE = "86161NED"
CHILDREN_TABLE = "71487ned"

WAGE_TABLE = "81431ned"     # jobs, wages, working hours by SBI2008 section

INCOME_SNAPSHOT = "{table}_{period}.csv"
WAGE_SNAPSHOT = "{table}_{period}.csv"
KWB_ESTABLISHMENTS_SNAPSHOT = "kwb_establishments_{table}.csv"
HOME_WORK_TABLE = "85718NED"      # home working by person characteristics
SECTOR_EDUCATION_TABLE = "82072NED"  # employee jobs by SBI2008 and education
HOME_WORK_SNAPSHOT = "home_working_{table}_{period}.csv"
SECTOR_EDUCATION_SNAPSHOT = "sector_education_{table}_{period}.csv"

# KWB reports establishments in these SBI2008 groups (title regex ->
# canonical group code). The topic KEYS differ per KWB vintage, the
# titles do not, so keys are resolved from the table's DataProperties.
ESTABLISHMENT_GROUPS: dict[str, str] = {
    "total": r"^Bedrijfsvestigingen totaal",
    "A": r"^A Landbouw",
    "B-F": r"^B-F Nijverheid",
    "G+I": r"^G\+I Handel",
    "H+J": r"^H\+J Vervoer",
    "K-L": r"^K-L Financi",
    "M-N": r"^M-N Zakelijke",
    "O-Q": r"^O-Q ",
    "R-U": r"^R-U Cultuur",
}
CHILDREN_SNAPSHOT = "{table}_{period}.csv"


def _odata_get(table: str, select: list[str], filter_expr: str | None,
               timeout: float = 120.0, root: str = ODATA_ROOT) -> pd.DataFrame:
    params = {"$select": ",".join(select), "$format": "json"}
    if filter_expr:
        params["$filter"] = filter_expr
    url = (f"{root}/{table}/TypedDataSet?"
           f"{urllib.parse.urlencode(params, quote_via=urllib.parse.quote)}")
    rows: list[dict] = []
    while url:
        logger.info("GET %s", url)
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            payload = json.load(resp)
        rows.extend(payload["value"])
        url = payload.get("odata.nextLink") or payload.get("@odata.nextLink")
    df = pd.DataFrame(rows)
    for col in df.select_dtypes(include="object"):
        df[col] = df[col].str.strip()
    return df


def fetch_income_seed(
    period: str,
    *,
    population_key: str,
    total_col: str,
    decile_cols: list[str],
    household_keys: list[str],
    table: str = INCOME_TABLE,
) -> pd.DataFrame:
    """Gemeente x household-type income-decile table (86161NED)."""
    select = ["Populatie", "KenmerkenVanHuishoudens", "RegioS", "Perioden",
              total_col, *decile_cols]
    # StatLine refuses queries that could return >= 10 000 rows, so the
    # household types must be in the server-side filter, not applied after.
    keys = " or ".join(f"KenmerkenVanHuishoudens eq '{k}'"
                       for k in household_keys)
    flt = (f"Perioden eq '{period}' and Populatie eq '{population_key}' "
           f"and ({keys})")
    return _odata_get(table, select, flt)


def fetch_single_parent_counts(
    period: str, *, age_total: str, table: str = CHILDREN_TABLE,
) -> pd.DataFrame:
    """Households with children / single-parent households (71487ned)."""
    select = ["LeeftijdKindEren", "Perioden", "RegioS",
              "TotaalHuishoudensMetKinderen_1",
              "TotaalEenouderhuishoudensMetKinderen_13"]
    return _odata_get(
        table, select,
        f"Perioden eq '{period}' and LeeftijdKindEren eq '{age_total}'")


def fetch_sector_wages(
    period: str, *, table: str = WAGE_TABLE, characteristic: str = "T001098",
) -> pd.DataFrame:
    """Jobs (x 1000) and mean hourly wage per SBI2008 section
    (81431NED, characteristic 'Totaal' = all employee jobs)."""
    select = ["KenmerkenBaanWerknemerBedrijf", "BedrijfstakkenBranchesSBI2008",
              "Perioden", "Banen_1", "Uurloon_3"]
    return _odata_get(
        table, select,
        f"Perioden eq '{period}' and "
        f"KenmerkenBaanWerknemerBedrijf eq '{characteristic}'")


def resolve_establishment_keys(table: str) -> dict[str, str]:
    """Group code -> topic key of the establishment counts in a KWB
    table. Groups a vintage does not publish are omitted."""
    import re

    url = f"{ODATA_ROOT}/{table}/DataProperties?$format=json"
    with urllib.request.urlopen(url, timeout=120) as resp:
        props = json.load(resp)["value"]
    topics = [(p["Key"], p.get("Title") or "") for p in props
              if p.get("Type") == "Topic"]
    keys: dict[str, str] = {}
    for group, pattern in ESTABLISHMENT_GROUPS.items():
        hits = [k for k, t in topics if re.search(pattern, t)]
        if len(hits) == 1:
            keys[group] = hits[0]
        elif len(hits) > 1:
            raise ValueError(f"{table}: title pattern for {group!r} matches "
                             f"several topics: {hits}")
    return keys


def fetch_kwb_establishments(table: str) -> pd.DataFrame:
    """Establishments per buurt and SBI group from a KWB table.

    Returns columns buurtcode, gemeentenaam and one column per group
    (`total`, `A`, `B-F`, ...); CBS rounds these counts and blanks
    suppressed cells (NaN). Uses the Feed endpoint, which pages results.
    """
    keys = resolve_establishment_keys(table)
    if "total" not in keys:
        raise ValueError(f"{table}: no establishment total found.")
    select = ["Codering_3", "Gemeentenaam_1", "SoortRegio_2", *keys.values()]
    out = _odata_get(table, select, "SoortRegio_2 eq 'Buurt'",
                     root=ODATA_FEED_ROOT)
    out = out.rename(columns={"Codering_3": "buurtcode",
                              "Gemeentenaam_1": "gemeentenaam",
                              **{v: k for k, v in keys.items()}})
    # The Feed endpoint ignores the region filter, so select buurten here.
    out = out[out["buurtcode"].str.match(r"^BU\d+")]
    if out["buurtcode"].duplicated().any():
        raise ValueError(f"{table}: duplicate buurt codes in the feed.")
    return out.drop(columns="SoortRegio_2").reset_index(drop=True)


def fetch_home_working_by_education(
    period: str, *, table: str = HOME_WORK_TABLE,
) -> pd.DataFrame:
    """Employed persons (x 1000) by education level and home-working
    category (85718NED): all education levels' totals, 'meestal of soms
    thuiswerken', 'meestal', 'soms' and 'niet'. Education keys:
    2018700 low (basisonderwijs, vmbo, mbo1), 2018740 middle (havo, vwo,
    mbo2-4), 2018790 high (hbo, wo)."""
    edu = ["2018700", "2018740", "2018790", "T009002"]
    cats = ["T001205", "A027929", "A027930", "A027931", "A027934"]
    f_edu = " or ".join(f"Persoonskenmerken eq '{k}'" for k in edu)
    f_cat = " or ".join(f"Thuiswerken eq '{k}'" for k in cats)
    select = ["Persoonskenmerken", "Thuiswerken", "PositieInDeWerkkring",
              "Geslacht", "Perioden", "WerkzameBeroepsbevolking_1"]
    df = _odata_get(
        table, select,
        f"Perioden eq '{period}' and PositieInDeWerkkring eq 'T001095' "
        f"and Geslacht eq 'T001038' and ({f_edu}) and ({f_cat})")
    return df


def fetch_sector_education_jobs(
    period: str = "2010JJ00", *, table: str = SECTOR_EDUCATION_TABLE,
) -> pd.DataFrame:
    """Employee jobs (x 1000) by SBI2008 section and education level
    (82072NED, published for 2010 only): total and low/middle/high."""
    edu = ["10000", "18700", "18740", "18790"]
    f_edu = " or ".join(f"Onderwijsniveau eq '{k}'" for k in edu)
    select = ["Geslacht", "Persoonskenmerken", "Onderwijsniveau",
              "BedrijfstakkenSBI2008", "Perioden", "BanenVanWerknemers_1"]
    return _odata_get(
        table, select,
        f"Perioden eq '{period}' and Geslacht eq '1100' and "
        f"Persoonskenmerken eq '10000' and ({f_edu})")


def snapshot_path(root: str | Path, template: str, table: str,
                  period: str) -> Path:
    return Path(root) / template.format(table=table, period=period)


def read_snapshot(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"StatLine snapshot not found: {path}. Run "
            f"`python -m ikob2.cli.segments fetch` once (needs network)."
        )
    return pd.read_csv(path, dtype={"RegioS": str,
                                    "KenmerkenVanHuishoudens": str,
                                    "Populatie": str, "Perioden": str,
                                    "LeeftijdKindEren": str})
