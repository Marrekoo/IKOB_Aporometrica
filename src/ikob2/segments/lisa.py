"""
LISA jobs by municipality and sector.

LISA (BIJ12, peildatum 1 April) publishes jobs ('banen') per
municipality in 15 LISA sectors, rounded to tens, for 2016-2025. Every
year is classified on the 2025 municipal boundaries and municipalities
are identified by NAME only. Municipal data does not come in a finer
SBI split than these 15 sectors.

Each LISA sector is a set of SBI2008 / NACE Rev. 2 sections
(`SECTOR_TO_NACE`), the link to the Eurostat occupation tables
(segments.occupations). The correspondence is an ASSUMPTION about the
LISA sector definitions (the file does not spell it out); L10 is M+N
without real estate, which follows from comparing establishment counts
(below).
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from ikob2.params import DEFAULTS

logger = logging.getLogger(__name__)

# LISA sector code (prefix of the sector label) -> NACE Rev. 2 sections
# (= SBI2008 sections). Order = LISA order.
SECTOR_TO_NACE: dict[str, tuple[str, ...]] = {
    "L01": ("A",),         # Landbouw, bosbouw en visserij
    "L02": ("B", "C"),     # Delfstoffen, industrie
    "L03": ("D", "E"),     # Energie, water en afval
    "L04": ("F",),         # Bouw
    "L05": ("G",),         # Handel
    "L06": ("H",),         # Vervoer en opslag
    "L07": ("I",),         # Horeca
    "L08": ("J",),         # Informatie en communicatie
    "L09": ("K",),         # Financiele dienstverlening
    "L10": ("M", "N"),     # Zakelijke diensten (see below)
    "L11": ("O",),         # Openbaar bestuur
    "L12": ("P",),         # Onderwijs
    "L13": ("Q",),         # Zorg
    "L14": ("R",),         # Cultuur, sport, recreatie
    "L15": ("S",),         # Overige diensten
}
SECTORS = tuple(SECTOR_TO_NACE)

# LISA sector -> KWB establishment group (CBS Kerncijfers wijken en
# buurten reports establishments in these SBI groups). Cross-checked
# against LISA's own establishment counts (2022): KWB/LISA is 0.97-1.09
# for A, B-F, G+I, H+J, O-Q, R-U and 1.03 for M-N against L10, which is
# why L10 is M+N only: real estate (L), most of KWB's K-L group, is
# essentially absent from LISA. L09 (finance, 17.7k LISA establishments)
# is therefore only a small part of KWB's K-L (182k) and that group is a
# weak proxy for L09 jobs.
SECTOR_TO_KWB_GROUP: dict[str, str] = {
    "L01": "A",
    "L02": "B-F", "L03": "B-F", "L04": "B-F",
    "L05": "G+I", "L07": "G+I",
    "L06": "H+J", "L08": "H+J",
    "L09": "K-L",
    "L10": "M-N",
    "L11": "O-Q", "L12": "O-Q", "L13": "O-Q",
    "L14": "R-U", "L15": "R-U",
}

# KWB 2022 municipality name -> LISA 2025 name where they differ
# (disambiguating suffixes, truncation, and mergers since 2022).
GEMEENTE_ALIASES = {
    "Beek": "Beek (L.)",
    "Hengelo": "Hengelo (O.)",
    "Laren": "Laren (NH.)",
    "Middelburg": "Middelburg (Z.)",
    "Rijswijk": "Rijswijk (ZH.)",
    "Stein": "Stein (L.)",
    "Nuenen, Gerwen en Nederwetten": "Nuenen, Gerwen en Nederwet",
    "Brielle": "Voorne aan Zee",           # merged 2023
    "Hellevoetsluis": "Voorne aan Zee",
    "Westvoorne": "Voorne aan Zee",
    "Weesp": "Amsterdam",                   # merged 2022
}


def sector_code(label: str) -> str:
    """'L05. Handel' -> 'L05'."""
    return str(label).split(".")[0].strip()


def read_lisa_sectors(path: str | Path,
                      year: int = DEFAULTS.accessibility.jobs_year) -> pd.DataFrame:
    """Jobs per municipality x LISA sector for one year (needs openpyxl).

    Returns a frame indexed by LISA municipality name with the 15 sector
    codes as columns; values are jobs (rounded to tens in the source).
    """
    raw = pd.read_excel(path, sheet_name="LISA Gemeenten per sector")
    return parse_lisa_sectors(raw, year)


def parse_lisa_sectors(raw: pd.DataFrame, year: int) -> pd.DataFrame:
    df = raw[raw["Jaar"] == year].copy()
    if df.empty:
        raise ValueError(f"No LISA rows for year {year}; available: "
                         f"{sorted(raw['Jaar'].unique())}")
    df["sector"] = df["LISA_sector"].map(sector_code)
    df["Banen"] = pd.to_numeric(df["Banen"], errors="coerce")
    if df["Banen"].isna().any() or (df["Banen"] < 0).any():
        raise ValueError("LISA jobs contain missing or negative values.")
    wide = df.pivot_table(index="Gemeente", columns="sector", values="Banen",
                          aggfunc="sum")
    unknown = set(wide.columns) - set(SECTORS)
    missing = set(SECTORS) - set(wide.columns)
    if unknown or missing:
        raise ValueError(f"LISA sectors differ from the expected 15: "
                         f"unknown {sorted(unknown)}, missing "
                         f"{sorted(missing)}.")
    # Absent rows are tiny sectors in small municipalities (e.g. utilities
    # in Blaricum); the sector rows add up to the municipal total within
    # rounding (max 40 jobs), so absent means zero.
    n_gap = int(wide.isna().sum().sum())
    if n_gap:
        logger.info("LISA %s: %d municipality x sector cells absent; "
                    "treated as 0.", year, n_gap)
    return wide[list(SECTORS)].fillna(0.0)


def lisa_gemeente_of_buurten(kwb: pd.DataFrame,
                             lisa_names) -> pd.Series:
    """LISA municipality name per buurt (indexed by buurtcode).

    kwb needs `buurtcode` and `gemeentenaam` (KWB 2022). Names are
    matched exactly, then through GEMEENTE_ALIASES. Buurten that map to
    no LISA municipality (e.g. 'Buitenland') get NaN and are logged.
    """
    names = set(lisa_names)
    mapped = kwb["gemeentenaam"].map(
        lambda n: n if n in names else GEMEENTE_ALIASES.get(n))
    mapped = mapped.where(mapped.isin(names))
    out = pd.Series(mapped.to_numpy(), index=kwb["buurtcode"].to_numpy(),
                    name="lisa_gemeente")
    lost = kwb.loc[mapped.isna().to_numpy(), "gemeentenaam"]
    if len(lost):
        logger.warning("%d buurten map to no LISA municipality: %s",
                       len(lost), dict(lost.value_counts().head(5)))
    return out

