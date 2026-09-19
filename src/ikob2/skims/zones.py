"""Zone points for routing, and coarse cells for far destinations."""

from __future__ import annotations

import numpy as np
import pandas as pd


def zone_points(codes, x, y, crs: str = "EPSG:28992",
                area_m2=None) -> pd.DataFrame:
    """Zones as routing points: id, lon, lat (WGS84), plus the projected
    x, y and optionally the zone area."""
    from pyproj import Transformer

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    lon, lat = Transformer.from_crs(crs, "EPSG:4326", always_xy=True) \
        .transform(x, y)
    df = pd.DataFrame({"id": [str(c) for c in codes], "lon": lon, "lat": lat,
                       "x": x, "y": y})
    if area_m2 is not None:
        df["area_m2"] = np.asarray(area_m2, dtype=float)
    if df["id"].duplicated().any():
        raise ValueError("Zone codes must be unique.")
    return df


def coarse_cells(codes, x, y, group, weights=None,
                 crs: str = "EPSG:28992") -> tuple[pd.DataFrame, dict]:
    """Representative points of coarse cells (e.g. municipalities).

    A cell's point is the weighted mean of its members' centroids
    (weights e.g. jobs, so the point sits where the opportunities are;
    unweighted if None or if a cell's weights are all zero).
    Returns (cells frame like zone_points, {member code: cell code}).
    """
    df = pd.DataFrame({"code": [str(c) for c in codes],
                       "x": np.asarray(x, float), "y": np.asarray(y, float),
                       "group": [str(g) for g in group]})
    df["w"] = 1.0 if weights is None else np.asarray(weights, float)
    df["w"] = df["w"].clip(lower=0.0)
    rows = []
    for g, d in df.groupby("group", sort=True):
        w = d["w"] if d["w"].sum() > 0 else pd.Series(1.0, index=d.index)
        rows.append((g, float(np.average(d["x"], weights=w)),
                     float(np.average(d["y"], weights=w))))
    cells = zone_points([r[0] for r in rows], [r[1] for r in rows],
                        [r[2] for r in rows], crs=crs)
    return cells, dict(zip(df["code"], df["group"]))
