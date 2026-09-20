"""Shared-bicycle hub locations from files.

A hub is a point where a shared bicycle can be picked up for the egress leg
of a public-transport journey (an OV-fiets station, a municipal hub). They
come from input files, not from the code:

  * CSV with columns `lat`, `lon` (and optionally `hub`, the name), for
    example `inputs/hubs/utrecht_hubs.csv`;
  * the OV-fiets feed (http://fiets.openov.nl/locaties.json): a JSON object
    `{"locaties": {code: {"lat", "lng", "name", ...}}}`.

Relative paths are looked up in the current folder first, then under
`<data root>/inputs`.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def resolve_path(path: str | Path, inputs: Path | None = None) -> Path:
    p = Path(path).expanduser()
    if p.is_absolute() or p.exists():
        return p
    if inputs is not None and (Path(inputs) / p).exists():
        return Path(inputs) / p
    raise FileNotFoundError(f"Hub file {path!r} not found"
                            f"{'' if inputs is None else f' (also tried under {inputs})'}.")


def read_hub_file(path: str | Path) -> pd.DataFrame:
    """One file -> frame with columns hub, lat, lon, source."""
    path = Path(path)
    if path.suffix.lower() == ".json":
        raw = json.loads(path.read_text())
        loc = raw.get("locaties", raw)
        df = pd.DataFrame([{"hub": v.get("name", k), "lat": v.get("lat"),
                            "lon": v.get("lng", v.get("lon"))}
                           for k, v in loc.items()])
    else:
        df = pd.read_csv(path)
        if "hub" not in df.columns:
            df["hub"] = [f"{path.stem}-{i}" for i in range(len(df))]
        if not {"lat", "lon"} <= set(df.columns):
            raise ValueError(f"{path}: needs columns lat and lon, has "
                             f"{list(df.columns)}.")
    df = df[["hub", "lat", "lon"]].copy()
    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    df["lon"] = pd.to_numeric(df["lon"], errors="coerce")
    bad = df[df[["lat", "lon"]].isna().any(axis=1)]
    if len(bad):
        raise ValueError(f"{path}: {len(bad)} hub(s) without coordinates, "
                         f"e.g. {list(bad['hub'][:3])}.")
    df["source"] = path.name
    return df.reset_index(drop=True)


def load_hubs(paths: Iterable[str | Path], inputs: Path | None = None,
              kinds: Iterable[str] | None = None,
              tariffs: Iterable[str] | None = None) -> pd.DataFrame:
    """All hubs of the listed files, and their RD New coordinates (x, y).
    `kinds` (one per file) names the tariff of a file's hubs and `tariffs`
    the known tariff names (a kind outside them is an error)."""
    from pyproj import Transformer

    paths = list(paths)
    frames = [read_hub_file(resolve_path(p, inputs)) for p in paths]
    if not frames:
        raise ValueError("No hub files given.")
    kind_list = list(kinds) if kinds is not None else None
    if kind_list is not None and len(kind_list) != len(frames):
        raise ValueError(f"{len(frames)} hub file(s) but {len(kind_list)} "
                         f"hub kind(s).")
    for i, f in enumerate(frames):
        f["kind"] = kind_list[i] if kind_list else "hub"
    df = pd.concat(frames, ignore_index=True)
    known = list(tariffs) if tariffs is not None else []
    if known:
        unknown = sorted(set(df["kind"]) - set(known))
        if unknown:
            raise ValueError(f"Unknown hub kind(s) {unknown}; tariffs: "
                             f"{known}.")
    x, y = Transformer.from_crs("EPSG:4326", "EPSG:28992",
                                always_xy=True).transform(
        df["lon"].to_numpy(), df["lat"].to_numpy())
    df["x"], df["y"] = x, y
    logger.info("%d hubs from %d file(s)", len(df), len(frames))
    return df


def hub_xy(hubs: pd.DataFrame) -> np.ndarray:
    return hubs[["x", "y"]].to_numpy(dtype=float)
