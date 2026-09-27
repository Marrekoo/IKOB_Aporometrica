"""
Build a tiny, fully synthetic data folder and run the accessibility model on it.

    python examples/tiny/make_data.py <root>          # build the data folder
    python -m ikob2.cli.accessibility --data-root <root> \
        --study tiny --run example --modes car bike pt

The folder has the layout of a real one (utils.paths.DataLayout):

  * 12 synthetic municipalities (GM9000..GM9011) of 4 buurten each, on a grid
    with 5 km between buurten, as a KWB-style GeoPackage with the attributes
    the segment pipeline reads;
  * synthetic StatLine snapshots for the two municipal tables (households by
    type and income decile, households with children); the sector-level tables
    (wages, home working, education by sector) are the real snapshots of the
    repository, which do not depend on the municipalities;
  * the reference files of the repository (budgets, time margins, tariffs,
    detour calibration), copied by `DataLayout.seed`;
  * synthetic jobs per LISA sector per buurt;
  * a skim store `intermediate/skims/tiny` with the 4 buurten of GM9000 as
    origins: car time and route distance, bicycle time (layers `near` and
    `far`), and public transport time, rail km, other km and boardings
    (layer `all`), all derived from crow-fly distances.

Every number comes from a seeded random generator or a closed formula, so the
folder, and the run on it, are the same on every machine.
`expected/accessibility.csv` is the output of the command above;
tests/test_example.py rebuilds the folder, reruns the model and compares.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from ikob2.segments import statline
from ikob2.segments.config import SegmentConfig
from ikob2.segments.lisa import SECTORS
from ikob2.skims.store import SkimStore
from ikob2.utils.paths import REPO_DATA, DataLayout

N_GM, PER_GM, SPACING_M = 12, 4, 5000.0
X0, Y0 = 130_000.0, 450_000.0          # RD New, around the centre of the country
STUDY = "tiny"


def municipalities() -> list[str]:
    return [f"GM{9000 + g:04d}" for g in range(N_GM)]


def buurten(rng: np.random.Generator):
    """Buurt table with polygons: municipality g is column g of the grid."""
    import geopandas as gpd
    from shapely.geometry import box

    recs, geoms = [], []
    for g, gm in enumerate(municipalities()):
        for b in range(PER_GM):
            x, y = X0 + g * SPACING_M, Y0 + b * SPACING_M
            hh = float(rng.integers(300, 1500))
            p_single = float(rng.integers(20, 55))
            p_no = float(rng.integers(20, 40))
            recs.append({
                "buurtcode": f"BU{9000 + g:04d}{b:04d}",
                "buurtnaam": f"Buurt {g}-{b}",
                "gemeentecode": gm, "gemeentenaam": f"Gemeente {g}",
                "water": "NEE",
                "aantal_inwoners": round(hh * rng.uniform(1.8, 2.6)),
                "aantal_huishoudens": hh,
                "percentage_eenpersoonshuishoudens": p_single,
                "percentage_huishoudens_zonder_kinderen": p_no,
                "percentage_huishoudens_met_kinderen": 100.0 - p_single - p_no,
                "stedelijkheid_adressen_per_km2": float(rng.integers(1, 6)),
                "gemiddelde_woningwaarde": float(rng.integers(150, 600)),
                "percentage_huishoudens_met_laag_inkomen":
                    float(rng.integers(10, 60)),
            })
            geoms.append(box(x - 400, y - 400, x + 400, y + 400))
    return gpd.GeoDataFrame(recs, geometry=geoms, crs="EPSG:28992")


def income_snapshot(rng: np.random.Generator) -> pd.DataFrame:
    """CBS 86161NED layout: households per type and income decile (per cent)."""
    cfg = SegmentConfig()
    rows = []
    for gm in municipalities():
        for key in cfg.household_key_map.values():
            pct = np.round(rng.dirichlet(np.ones(10) * 8) * 100)
            row = {"Populatie": cfg.income_population_key,
                   "KenmerkenVanHuishoudens": key, "RegioS": gm,
                   "Perioden": cfg.income_period,
                   cfg.income_total_col: float(rng.uniform(2, 60))}
            row.update(dict(zip(cfg.income_decile_cols.values(), pct)))
            rows.append(row)
    return pd.DataFrame(rows)


def children_snapshot(rng: np.random.Generator) -> pd.DataFrame:
    """CBS 71487ned layout: households with children, single-parent ones."""
    cfg = SegmentConfig()
    return pd.DataFrame([{
        "LeeftijdKindEren": cfg.children_age_total, "Perioden":
            cfg.children_period, "RegioS": gm,
        "TotaalHuishoudensMetKinderen_1": float(rng.integers(500, 5000)),
        "TotaalEenouderhuishoudensMetKinderen_13": float(rng.integers(80, 900))}
        for gm in municipalities()])


def sector_jobs(codes, rng: np.random.Generator) -> pd.DataFrame:
    jobs = rng.gamma(0.8, 60.0, (len(codes), len(SECTORS)))
    return pd.DataFrame(np.round(jobs, 1), index=pd.Index(codes, name="buurtcode"),
                        columns=list(SECTORS))


def write_store(root: Path, kwb) -> None:
    """Skims from crow-fly distances: car 45 km/h + 2 min, detour 1.3; bicycle
    16 km/h, detour 1.3; public transport 30 km/h + 8 min, rail beyond 10 km."""
    codes = list(kwb["buurtcode"])
    xy = np.column_stack([kwb.geometry.centroid.x, kwb.geometry.centroid.y])
    origins = [c for c, gm in zip(codes, kwb["gemeentecode"]) if gm == "GM9000"]
    o = xy[[codes.index(c) for c in origins]]
    gms = municipalities()
    gm_xy = np.array([xy[(kwb["gemeentecode"] == gm).to_numpy()].mean(axis=0)
                      for gm in gms])
    store = SkimStore.create(
        root, origins, {"near": codes, "far": gms, "all": codes},
        cell_of={"far": dict(zip(codes, kwb["gemeentecode"]))},
        meta={"source": "synthetic example (examples/tiny/make_data.py)"})

    def km(a, b):
        return np.hypot(a[:, None, 0] - b[None, :, 0],
                        a[:, None, 1] - b[None, :, 1]) / 1000.0

    for layer, dest in (("near", xy), ("far", gm_xy)):
        d = km(o, dest)
        store.write_rows(layer, "car", "time", 0, 2.0 + d * 1.3 / 45.0 * 60.0)
        store.write_rows(layer, "car", "distance", 0, d * 1.3)
        store.write_rows(layer, "bike", "time", 0, d * 1.3 / 16.0 * 60.0)
    d = km(o, xy)
    rail = np.where(d > 10.0, d * 1.15, 0.0)
    other = np.where(d > 10.0, 3.0, d * 1.25)
    store.write_rows("all", "pt", "time", 0, 8.0 + d * 1.2 / 30.0 * 60.0)
    store.write_rows("all", "pt", "rail_km", 0, rail)
    store.write_rows("all", "pt", "other_km", 0, other)
    store.write_rows("all", "pt", "other_boardings", 0, np.ones_like(d))


def build(root: str | Path, seed: int = 2026) -> DataLayout:
    rng = np.random.default_rng(seed)
    lay = DataLayout(Path(root))
    lay.ensure()
    cfg = SegmentConfig()
    # the municipal tables first: seed() then leaves them alone
    income_snapshot(rng).to_csv(statline.snapshot_path(
        lay.statline(), statline.INCOME_SNAPSHOT, statline.INCOME_TABLE,
        cfg.income_period), index=False)
    children_snapshot(rng).to_csv(statline.snapshot_path(
        lay.statline(), statline.CHILDREN_SNAPSHOT, statline.CHILDREN_TABLE,
        cfg.children_period), index=False)
    lay.seed(REPO_DATA)
    kwb = buurten(rng)
    kwb.to_file(lay.kwb(2022), driver="GPKG", layer="buurten")
    sector_jobs(list(kwb["buurtcode"]), rng).to_csv(lay.sector_jobs(2022))
    write_store(lay.skim_dir(STUDY), kwb)
    return lay


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    print(f"Synthetic data folder written to {build(sys.argv[1]).root}")
