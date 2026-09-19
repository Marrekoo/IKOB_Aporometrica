"""End-to-end segment pipeline: KWB + StatLine snapshots -> 44 segment
columns per buurt (household-based and population-scaled)."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ikob2.segments import statline
from ikob2.segments.config import LOW_INCOME_CLASSES, SegmentConfig
from ikob2.segments.ipf import ipf_batch
from ikob2.segments.kwb import diagnose_kwb, read_kwb
from ikob2.segments.marginals import (
    build_gemeente_seed,
    decile_marginals,
    household_marginals,
    single_parent_shares,
)
from ikob2.segments.structure import (
    fit_structure_model,
    gemeente_covariates,
    std_vec,
)

logger = logging.getLogger(__name__)

_ID_COLS = ["buurtcode", "buurtnaam", "gemeentecode", "gemeentenaam",
            "inwoners", "huishoudens"]


@dataclass(frozen=True)
class SegmentResult:
    household_based: pd.DataFrame     # persons = households x hh_size
    population_scaled: pd.DataFrame   # rows rescaled to KWB inwoners
    diagnostics: pd.DataFrame         # per buurt: status, iters, max_diff
    hh_marginals: pd.DataFrame
    dec_marginals: pd.DataFrame
    report: dict


def compute_segments(kwb: pd.DataFrame, income_raw: pd.DataFrame,
                     children_raw: pd.DataFrame,
                     cfg: SegmentConfig) -> SegmentResult:
    """Pure core: takes already-loaded tables, touches no files."""
    diagnose_kwb(kwb)

    seed_study = build_gemeente_seed(income_raw, cfg, cfg.gemeente_codes)
    seed_ref = build_gemeente_seed(income_raw, cfg,
                                   cfg.reference_gemeente_codes)
    model = fit_structure_model(seed_ref, gemeente_covariates(kwb), cfg)

    sp = single_parent_shares(children_raw, cfg, cfg.gemeente_codes)
    hh = household_marginals(kwb, sp, cfg)
    dec = decile_marginals(kwb, seed_study, cfg)

    types, classes = list(cfg.household_types), list(cfg.income_classes)
    row_t = hh[types].to_numpy(float)
    row_t = np.where(np.isnan(row_t) | (row_t < 0), 0.0, row_t)
    total = row_t.sum(axis=1)
    ok = total > 0

    share = dec[classes].to_numpy(float)
    col_t = np.where(np.isnan(share) | (share < 0), 0.0, share) * total[:, None]
    col_sum = col_t.sum(axis=1)
    uniform = col_sum <= 0
    col_t = np.where(
        uniform[:, None], (total / len(classes))[:, None],
        col_t * (total / np.where(uniform, 1.0, col_sum))[:, None])

    sted = std_vec(kwb["stedelijkheid"]).to_numpy()
    woz = std_vec(kwb["gem_woz"]).to_numpy()
    seeds = model.predict(sted, woz, cfg)

    n, T, C = len(kwb), len(types), len(classes)
    table = np.zeros((n, T, C))
    iters = np.full(n, -1)
    max_diff = np.full(n, np.nan)
    status = np.array(["skipped_zero_households"] * n, dtype=object)
    if ok.any():
        res = ipf_batch(seeds[ok], row_t[ok], col_t[ok],
                        tol=cfg.ipf_tol, max_iter=cfg.ipf_max_iter)
        table[ok] = res.table
        iters[ok] = res.iters
        max_diff[ok] = res.max_diff
        status[ok] = np.where(res.converged, "ok", "not_converged")

    hh_size = np.array([cfg.hh_size[t] for t in types])
    persons = table * hh_size[None, :, None]
    # column-major over (class, type): D1 for every type, then D2, ...
    seg = persons.transpose(0, 2, 1).reshape(n, C * T)
    seg_df = pd.DataFrame(seg, columns=cfg.segment_columns)

    base = kwb[_ID_COLS].reset_index(drop=True)
    household_based = pd.concat([base, seg_df], axis=1)
    scaled = pd.concat([base, _scale_to_population(seg, kwb["inwoners"]
                                                   .to_numpy(float),
                                                   cfg.segment_columns)],
                       axis=1)
    diagnostics = pd.DataFrame({"buurtcode": kwb["buurtcode"].to_numpy(),
                                "status": status, "iters": iters,
                                "max_diff": max_diff})
    report = _report(household_based, diagnostics, hh, dec, cfg)
    return SegmentResult(household_based, scaled, diagnostics, hh, dec,
                         report)


def _scale_to_population(seg: np.ndarray, pop: np.ndarray,
                         columns: list[str]) -> pd.DataFrame:
    x = np.where(np.isnan(seg) | (seg < 0), 0.0, seg)
    s = x.sum(axis=1)
    bad = np.isnan(pop) | (pop <= 0) | (s <= 0)
    factor = np.where(bad, 0.0, pop / np.where(s > 0, s, 1.0))
    return pd.DataFrame(x * factor[:, None], columns=columns)


def _report(hb: pd.DataFrame, diag: pd.DataFrame, hh: pd.DataFrame,
            dec: pd.DataFrame, cfg: SegmentConfig) -> dict:
    seg_total = hb[cfg.segment_columns].sum(axis=1)
    pop = hb["inwoners"]
    mad = float((abs(seg_total - pop) / np.maximum(pop, 1) * 100).mean())
    low = dec[list(LOW_INCOME_CLASSES)].sum(axis=1)
    report = {
        "n_buurten": len(hb),
        "converged": int((diag["status"] == "ok").sum()),
        "not_converged": int((diag["status"] == "not_converged").sum()),
        "skipped": int(diag["status"].str.startswith("skipped").sum()),
        "persons_vs_inwoners_mean_abs_pct_diff": mad,
        "hh_marginal_max_abs_diff": float(
            (hh["check_total"] - hh["huishoudens"]).abs().max()),
        "calibration_sources": dec["calibration_source"].value_counts()
        .to_dict(),
        "d1_d4_max_abs_diff_to_target": float(
            (low - dec["low_target_used"]).abs().max()),
    }
    logger.info("Segment report: %s", report)
    return report


def run_pipeline(kwb_path: str | Path, statline_dir: str | Path,
                 cfg: SegmentConfig | None = None) -> SegmentResult:
    cfg = cfg or SegmentConfig()
    kwb = read_kwb(kwb_path, cfg)
    income = statline.read_snapshot(statline.snapshot_path(
        statline_dir, statline.INCOME_SNAPSHOT, statline.INCOME_TABLE,
        cfg.income_period))
    children = statline.read_snapshot(statline.snapshot_path(
        statline_dir, statline.CHILDREN_SNAPSHOT, statline.CHILDREN_TABLE,
        cfg.children_period))
    return compute_segments(kwb, income, children, cfg)


def write_gpkg(result: SegmentResult, kwb_path: str | Path,
               out_path: str | Path, cfg: SegmentConfig,
               layer: str = "buurt_segments") -> None:
    """Write both variants as layers, with the KWB buurt geometry."""
    import geopandas as gpd

    geom = gpd.read_file(kwb_path, layer=cfg.kwb_layer,
                         columns=[cfg.kwb_vars["zonecode"]])
    geom = geom.rename(columns={cfg.kwb_vars["zonecode"]: "buurtcode"})
    geom["buurtcode"] = geom["buurtcode"].astype(str).str.strip()
    geom = geom.drop_duplicates("buurtcode")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        out_path.unlink()
    for suffix, df in (("household_based", result.household_based),
                       ("population_scaled", result.population_scaled)):
        gdf = gpd.GeoDataFrame(
            df.merge(geom, how="left", on="buurtcode"), geometry="geometry",
            crs=geom.crs)
        gdf.to_file(out_path, layer=f"{layer}_{suffix}", driver="GPKG")
