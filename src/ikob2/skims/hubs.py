"""Shared-bicycle hub locations from files.

A hub is a point where a shared bicycle can be picked up for the egress leg
of a public-transport journey (an OV-fiets station, a municipal hub). They
come from input files, not from the code:

  * CSV with columns `lat`, `lon` (and optionally `hub`, the name), for
    example `inputs/hubs/utrecht_hubs.csv`;
  * the OV-fiets feed (http://fiets.openov.nl/locaties.json): a JSON object
    `{"locaties": {code: {"lat", "lng", "name", ...}}}`.

Every file has a tariff kind (`pt.hub_kinds`, parallel to `pt.hub_files`;
on the command line `--hub-file FILE:KIND`). Relative paths are looked up
in the current folder first, then under `<data root>/inputs` and
`<data root>/intermediate` (hubs made by the model, e.g.
`intermediate/hubs/utrecht_hubs_s2.csv`).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def search_dirs(root: str | Path | None) -> list[Path]:
    """Folders of the data root in which relative hub paths are looked up:
    `inputs/` (source hub files) and `intermediate/` (hubs made by the
    model, such as the extra hubs of scenario S2)."""
    if not root:
        return []
    return [Path(root) / "inputs", Path(root) / "intermediate"]


def resolve_path(path: str | Path,
                 dirs: str | Path | Iterable[str | Path] | None = None) -> Path:
    """An absolute path, a path relative to the current folder, or a path
    relative to exactly one of `dirs` (a path found in several is an error,
    so a stale copy cannot shadow the file meant)."""
    p = Path(path).expanduser()
    if p.is_absolute() or p.exists():
        return p
    if dirs is None:
        dirs = []
    elif isinstance(dirs, (str, Path)):
        dirs = [dirs]
    dirs = [Path(d) for d in dirs]
    found = [d / p for d in dirs if (d / p).exists()]
    if len(found) > 1:
        raise ValueError(f"Hub file {path!r} exists in several folders "
                         f"({', '.join(map(str, found))}); remove the stale "
                         f"copy or give an absolute path.")
    if found:
        return found[0]
    tried = f" (also tried under {', '.join(map(str, dirs))})" if dirs else ""
    raise FileNotFoundError(f"Hub file {path!r} not found{tried}.")


def parse_hub_file_arg(value: str,
                       tariffs: Iterable[str] | None = None) -> tuple[str, str]:
    """`FILE:KIND` (a command-line hub file with its tariff kind) ->
    (file, kind). The kind is required: files and kinds are parallel
    lists, and a file without its kind would take another file's."""
    file, sep, kind = str(value).rpartition(":")
    if not sep or not file or not kind:
        raise ValueError(f"Hub file {value!r}: give FILE:KIND, e.g. "
                         f"hubs/utrecht_hubs.csv:lime.")
    known = list(tariffs) if tariffs is not None else []
    if known and kind not in known:
        raise ValueError(f"Hub file {value!r}: unknown kind {kind!r}; "
                         f"tariffs: {known}.")
    return file, kind


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


def load_hubs(paths: Iterable[str | Path],
              dirs: str | Path | Iterable[str | Path] | None = None,
              kinds: Iterable[str] | None = None,
              tariffs: Iterable[str] | None = None) -> pd.DataFrame:
    """All hubs of the listed files, and their RD New coordinates (x, y).
    Relative paths are resolved against `dirs` (`search_dirs(root)`).
    `kinds` (one per file) names the tariff of a file's hubs and `tariffs`
    the known tariff names (a kind outside them is an error)."""
    from pyproj import Transformer

    paths = list(paths)
    frames = [read_hub_file(resolve_path(p, dirs)) for p in paths]
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
