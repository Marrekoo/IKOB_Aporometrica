"""
CBS StatLine access for the segment pipeline.

Only two tables are needed, and both are tiny, so they are downloaded
once and cached as CSV snapshots under data/statline/. Runs then read
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

INCOME_TABLE = "86161NED"
CHILDREN_TABLE = "71487ned"

INCOME_SNAPSHOT = "{table}_{period}.csv"
CHILDREN_SNAPSHOT = "{table}_{period}.csv"


def _odata_get(table: str, select: list[str], filter_expr: str | None,
               timeout: float = 120.0) -> pd.DataFrame:
    params = {"$select": ",".join(select), "$format": "json"}
    if filter_expr:
        params["$filter"] = filter_expr
    url = (f"{ODATA_ROOT}/{table}/TypedDataSet?"
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
