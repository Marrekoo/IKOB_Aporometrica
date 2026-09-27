"""Read the Kerncijfers wijken en buurten inputs the pipeline needs."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from ikob2.segments.config import SegmentConfig

logger = logging.getLogger(__name__)

_REQUIRED_ROLES = (
    "zonecode", "zonename", "citycode", "cityname", "inhabitants",
    "households", "p_hh_single", "p_hh_no_child", "p_hh_with_child",
    "p_hh_low_income", "stedelijkheid", "avg_house_value",
)


def clean_pct(x: pd.Series) -> pd.Series:
    """Percentages: CBS suppression/not-applicable codes are negative
    and anything outside 0..100 is invalid -> NaN."""
    x = pd.to_numeric(x, errors="coerce").astype(float)
    return x.where((x >= 0) & (x <= 100))


def _nonneg(x: pd.Series) -> pd.Series:
    x = pd.to_numeric(x, errors="coerce").astype(float)
    return x.where(x >= 0)


def read_kwb(path: str | Path, cfg: SegmentConfig) -> pd.DataFrame:
    """Buurt table (no geometry) with cleaned KWB inputs.

    Suppressed counts (negative sentinel) become NaN; this cannot change
    the result because a buurt without households is skipped either way.
    """
    import pyogrio

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"KWB GeoPackage not found: {path}")
    layer = cfg.kwb_layer
    if layer is None:
        layer = pyogrio.list_layers(str(path))[0, 0]
        logger.info("kwb_layer not set; using first layer: %s", layer)

    v = cfg.kwb_vars
    missing_roles = [r for r in _REQUIRED_ROLES if r not in v]
    if missing_roles:
        raise KeyError(f"Missing cfg.kwb_vars entries: {missing_roles}")
    want = [v[r] for r in _REQUIRED_ROLES]

    available = list(pyogrio.read_info(str(path), layer=layer)["fields"])
    missing = [c for c in want if c not in available]
    if missing:
        raise KeyError(
            f"{path} layer '{layer}': configured KWB columns missing: "
            f"{missing}; available: {available}")

    raw = pyogrio.read_dataframe(str(path), layer=layer, columns=want,
                                 read_geometry=False)

    hh = _nonneg(raw[v["households"]])
    out = pd.DataFrame({
        "buurtcode": raw[v["zonecode"]].astype(str).str.strip(),
        "buurtnaam": raw[v["zonename"]].astype(str),
        "gemeentecode": raw[v["citycode"]].astype(str).str.strip(),
        "gemeentenaam": raw[v["cityname"]].astype(str),
        "inwoners": _nonneg(raw[v["inhabitants"]]),
        "huishoudens": hh,
        "p_hh_single": clean_pct(raw[v["p_hh_single"]]),
        "p_hh_no_child": clean_pct(raw[v["p_hh_no_child"]]),
        "p_hh_with_child": clean_pct(raw[v["p_hh_with_child"]]),
        "p_laag40": clean_pct(raw[v["p_hh_low_income"]]),
    })
    out["hh_eenpersoons"] = hh * out["p_hh_single"] / 100
    out["hh_zonder_kind"] = hh * out["p_hh_no_child"] / 100
    out["hh_met_kind"] = hh * out["p_hh_with_child"] / 100

    for name, role in (("stedelijkheid", "stedelijkheid"),
                       ("gem_woz", "avg_house_value")):
        # CBS suppression codes are negative sentinels (-99999999, -99997, -99995 by vintage);
        # left in, they would enter the municipal means and z-scores.
        out[name] = _nonneg(raw[v[role]])

    keep = (out["buurtcode"].str.match(r"^BU")
            & out["gemeentecode"].str.match(r"^GM[0-9]{4}$"))
    out = out[keep]
    if cfg.gemeente_codes is not None:
        out = out[out["gemeentecode"].isin(cfg.gemeente_codes)]
    if out.empty:
        raise ValueError("No buurten left after filtering; check "
                         "cfg.gemeente_codes and the KWB gemeentecode.")
    return out.reset_index(drop=True)


def diagnose_kwb(kwb: pd.DataFrame) -> dict:
    """Log the two input diagnostics of the R script and return them."""
    comp = kwb["p_hh_single"] + kwb["p_hh_no_child"] + kwb["p_hh_with_child"]
    diag = {
        "n_buurten": len(kwb),
        "median_p_comp_sum": float(comp.median()),
        "n_missing_comp_sum": int(comp.isna().sum()),
        "p_laag40_median": float(kwb["p_laag40"].median()),
        "p_laag40_na": int(kwb["p_laag40"].isna().sum()),
    }
    logger.info("KWB composition/income diagnostic: %s", diag)
    if diag["median_p_comp_sum"] > 120:
        logger.warning(
            "KWB composition percentages sum above 100 (median %.1f): "
            "p_hh_no_child may include one-person households; consider "
            "kwb_no_child_includes_single=True.",
            diag["median_p_comp_sum"])
    return diag
