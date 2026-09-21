"""
Reader for CBS "Kerncijfers wijken en buurten" GeoPackages.

CBS ships this product under two conventions depending on vintage and
export path: lowercase Dutch column names in the modern GeoPackage
releases ("buurtcode", "buurtnaam", ...), and uppercase truncated
names in shapefile-derived exports ("BU_CODE", "BU_NAAM", ...).
Column resolution below tries both, case-insensitively, and fails
loudly with the actual column list rather than a bare KeyError, since
this is the first thing a new CBS vintage breaks.

Zone geometry is reduced to a centroid: the accessibility engine
consumes zone-to-zone skims, not polygons. Centroids are computed in
a projected CRS (native RD New, EPSG:28992, unless the file says
otherwise) so "centre" means metres, not degrees.

CBS suppresses small-count figures with negative sentinel codes
("geheim", secret; not applicable) rather than leaving cells empty. The
code differs between vintages and tables (-99999999 in older files,
-99997 and -99995 in the 2022 Wijken en Buurten file). All attributes
used here are counts, shares and averages, which are never negative, so
every negative value in a numeric attribute column is treated as
suppressed and set to NaN; the distinct codes found are reported so a
silent sentinel never leaks into a population or income figure.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable

import numpy as np

from ikob2.data.validation import ValidationReport
from ikob2.domain.zones import ZoneSet

logger = logging.getLogger(__name__)

CBS_SECRET_SENTINEL = -99999999  # the older code; any negative value counts as suppressed

# Candidate column names, in priority order, case-insensitive.
# Left entry wins when several are present in the same file.
_CODE_CANDIDATES = ("buurtcode", "bu_code", "buurtcd", "bu_naam_code")
_NAME_CANDIDATES = ("buurtnaam", "bu_naam")
_MUN_CODE_CANDIDATES = ("gemeentecode", "gm_code")
_MUN_NAME_CANDIDATES = ("gemeentenaam", "gm_naam")
_WATER_CANDIDATES = ("water",)


def _resolve_column(columns: Iterable[str], candidates: tuple[str, ...]) -> str | None:
    lower_map = {c.lower(): c for c in columns}
    for candidate in candidates:
        if candidate in lower_map:
            return lower_map[candidate]
    return None


def list_layers(path: str | Path) -> list[str]:
    import pyogrio
    return pyogrio.list_layers(str(path))[:, 0].tolist()


def _pick_layer(path: Path, layer: str | None) -> str:
    layers = list_layers(path)
    if not layers:
        raise ValueError(f"{path}: GeoPackage contains no layers.")
    if layer is not None:
        if layer not in layers:
            raise ValueError(
                f"{path}: layer '{layer}' not found; available layers: "
                f"{layers}"
            )
        return layer
    if len(layers) == 1:
        return layers[0]
    buurt_layers = [l for l in layers if "buurt" in l.lower()]
    if len(buurt_layers) == 1:
        return buurt_layers[0]
    raise ValueError(
        f"{path}: {len(layers)} layers found ({layers}) and none "
        f"uniquely named '*buurt*'; pass layer=... explicitly."
    )


def load_cbs_buurten(
    path: str | Path,
    *,
    layer: str | None = None,
    attribute_columns: tuple[str, ...] | None = None,
    skip_water: bool = True,
    target_crs: str = "EPSG:28992",
) -> tuple[ZoneSet, ValidationReport]:
    """
    Load a CBS Kerncijfers wijken en buurten GeoPackage as a ZoneSet.

    Parameters
    ----------
    path : path to the .gpkg file.
    layer : layer name, or None to auto-detect (single layer, or the
        unique layer whose name contains "buurt").
    attribute_columns : extra numeric columns to carry through as
        ZoneSet.attributes (e.g. ("aantal_inwoners",
        "gemiddeld_inkomen_per_inwoner")). None means: carry every
        numeric column found, since the caller rarely knows the exact
        CBS vintage's column names up front and can subset later via
        ZoneSet.attribute().
    skip_water : drop rows whose "water" column marks them as water
        area ("JA"/"ja"/True) rather than a populated neighbourhood.
        Purely cosmetic zones (open water) are not accessibility
        origins. No-op if the file has no water column.
    target_crs : CRS to project into before computing centroids. CBS
        data ships in RD New (EPSG:28992) natively; this only takes
        effect if the file is in something else.

    Returns
    -------
    (ZoneSet, ValidationReport) — the report is warn-and-collect
    (see data.validation); call `.raise_if_failed()` if you want
    load errors to abort instead of being merely logged.
    """
    import geopandas as gpd
    import pandas as pd

    path = Path(path)
    report = ValidationReport()

    if not path.exists():
        report.error("GeoPackage not found: %s", path)
        report.raise_if_failed()

    resolved_layer = _pick_layer(path, layer)
    gdf = gpd.read_file(path, layer=resolved_layer)

    if gdf.crs is None:
        report.warning(
            "%s: layer '%s' has no CRS set; assuming %s.",
            path, resolved_layer, target_crs,
        )
    elif str(gdf.crs) != target_crs:
        gdf = gdf.to_crs(target_crs)

    code_col = _resolve_column(gdf.columns, _CODE_CANDIDATES)
    if code_col is None:
        raise KeyError(
            f"{path} layer '{resolved_layer}': no buurt-code column "
            f"found among {list(gdf.columns)}; expected one of "
            f"{_CODE_CANDIDATES}."
        )
    name_col = _resolve_column(gdf.columns, _NAME_CANDIDATES)
    mun_code_col = _resolve_column(gdf.columns, _MUN_CODE_CANDIDATES)
    mun_name_col = _resolve_column(gdf.columns, _MUN_NAME_CANDIDATES)
    water_col = _resolve_column(gdf.columns, _WATER_CANDIDATES)

    n_before = len(gdf)
    empty_geom = gdf.geometry.isna() | gdf.geometry.is_empty
    if empty_geom.any():
        report.warning(
            "%s: %d zone(s) with missing/empty geometry dropped.",
            path, int(empty_geom.sum()),
        )
        gdf = gdf.loc[~empty_geom]

    if skip_water and water_col is not None:
        is_water = gdf[water_col].astype(str).str.strip().str.upper().eq("JA")
        if is_water.any():
            report.warning(
                "%s: %d water zone(s) dropped (column '%s').",
                path, int(is_water.sum()), water_col,
            )
            gdf = gdf.loc[~is_water]

    dup_mask = gdf[code_col].duplicated(keep=False)
    if dup_mask.any():
        dupes = sorted(gdf.loc[dup_mask, code_col].unique())
        report.error(
            "%s: %d duplicate buurt code(s), e.g. %s.",
            path, len(dupes), dupes[:5],
        )
    report.raise_if_failed()

    if len(gdf) < n_before:
        logger.info(
            "%s: %d of %d zones retained after filtering.",
            path, len(gdf), n_before,
        )

    codes = gdf[code_col].astype(str).to_numpy()
    names = (
        gdf[name_col].astype(str).to_numpy() if name_col is not None
        else codes.copy()
    )
    municipality_code = (
        gdf[mun_code_col].astype(str).to_numpy() if mun_code_col is not None
        else None
    )
    municipality_name = (
        gdf[mun_name_col].astype(str).to_numpy() if mun_name_col is not None
        else None
    )

    centroids = gdf.geometry.centroid
    centroid_x = centroids.x.to_numpy(dtype=np.float64)
    centroid_y = centroids.y.to_numpy(dtype=np.float64)

    reserved = {code_col, name_col, mun_code_col, mun_name_col,
                water_col, gdf.geometry.name}
    reserved.discard(None)

    if attribute_columns is None:
        # pandas/pyogrio nullable dtypes (Int64, Float64, boolean — the
        # ones CBS integer columns get promoted to as soon as one row is
        # NULL) and the geometry dtype all crash np.issubdtype, which
        # expects a plain numpy dtype. is_numeric_dtype handles the
        # extension-array cases directly and returns False (not an
        # exception) for geometry.
        candidate_cols = [
            c for c in gdf.columns
            if c not in reserved and pd.api.types.is_numeric_dtype(gdf[c])
        ]
    else:
        missing = [c for c in attribute_columns if c not in gdf.columns]
        if missing:
            raise KeyError(
                f"{path} layer '{resolved_layer}': requested attribute "
                f"column(s) not found: {missing}; available: "
                f"{list(gdf.columns)}"
            )
        candidate_cols = list(attribute_columns)

    attributes: dict[str, np.ndarray] = {}
    for col in candidate_cols:
        values = gdf[col].to_numpy(dtype=np.float64)
        negative = values < 0  # NaN compares False
        n_secret = int(negative.sum())
        if n_secret:
            codes_found = sorted({int(v) for v in values[negative]})
            report.warning(
                "%s: column '%s' has %d suppressed CBS value(s) "
                "(negative codes %s) set to NaN.",
                path, col, n_secret, codes_found[:5],
            )
            values = np.where(negative, np.nan, values)
        attributes[col] = values

    zones = ZoneSet(
        codes=codes,
        names=names,
        centroid_x=centroid_x,
        centroid_y=centroid_y,
        crs=str(gdf.crs) if gdf.crs is not None else target_crs,
        municipality_code=municipality_code,
        municipality_name=municipality_name,
        attributes=attributes,
    )

    report.raise_if_failed()
    return zones, report
