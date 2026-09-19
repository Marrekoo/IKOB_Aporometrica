"""
On-disk skim store: dense float32 matrices, read lazily.

Layout (one directory):

    manifest.json                       origins, layers, metadata, progress
    <layer>/<mode>/<variable>.npy       (n_origins, n_layer_destinations)

Matrices are plain .npy files opened as memory maps, so reading a block
of origin rows touches only those rows, and a study area (a few hundred
origins) against national destinations is a few megabytes per matrix.
Values are NaN where a pair is unreachable within the routing limit
(or has not been computed yet: see `pending_blocks`).

NEAR / FAR LAYERS. Far-away destinations do not need buurt resolution.
A store can hold a NEAR layer (destinations routed individually) and a
FAR layer (representative points of coarse cells, e.g. municipalities)
together with `cell_of`, which maps every destination code to its
coarse cell. `combined()` assembles a matrix for any destination list
on the fly: the near value where the destination is in the near layer,
otherwise the value of its coarse cell. That keeps routing and storage
cost roughly proportional to the number of near destinations plus the
number of cells.

Writes are block-wise and recorded in the manifest, so an interrupted
build resumes where it stopped.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

MANIFEST = "manifest.json"
DTYPE = np.float32


@dataclass(frozen=True)
class LayerInfo:
    destinations: tuple[str, ...]
    cell_of: dict | None = None      # destination code -> coarse cell code


class SkimStore:
    def __init__(self, root: str | Path, manifest: dict):
        self.root = Path(root)
        self._m = manifest
        self._mm_cache: dict = {}

    # ── creation / opening ───────────────────────────────────────────

    @classmethod
    def create(cls, root: str | Path, origins: Sequence[str],
               layers: dict[str, Sequence[str]],
               *, cell_of: dict[str, dict[str, str]] | None = None,
               meta: dict | None = None) -> "SkimStore":
        """New store. `layers` maps layer name -> destination codes;
        `cell_of[layer]` (optional) maps ANY destination code to a code of
        that layer, making it a coarse layer usable by `combined`."""
        root = Path(root)
        if (root / MANIFEST).exists():
            raise FileExistsError(f"A skim store already exists at {root}.")
        origins = [str(o) for o in origins]
        if len(set(origins)) != len(origins):
            raise ValueError("Origin codes must be unique.")
        cell_of = cell_of or {}
        manifest = {"version": 1, "origins": origins, "layers": {},
                    "meta": meta or {}, "progress": {}}
        for name, dests in layers.items():
            dests = [str(d) for d in dests]
            if len(set(dests)) != len(dests):
                raise ValueError(f"Layer '{name}': duplicate destinations.")
            if name in cell_of:
                unknown = set(cell_of[name].values()) - set(dests)
                if unknown:
                    raise ValueError(
                        f"Layer '{name}': cell_of points to codes not in "
                        f"the layer, e.g. {sorted(unknown)[:3]}.")
            manifest["layers"][name] = {
                "destinations": dests,
                "cell_of": {str(k): str(v)
                            for k, v in cell_of.get(name, {}).items()} or None,
            }
        root.mkdir(parents=True, exist_ok=True)
        store = cls(root, manifest)
        store._save()
        return store

    @classmethod
    def open(cls, root: str | Path) -> "SkimStore":
        path = Path(root) / MANIFEST
        if not path.exists():
            raise FileNotFoundError(f"No skim store at {root}.")
        return cls(root, json.loads(path.read_text()))

    def _save(self) -> None:
        tmp = self.root / (MANIFEST + ".tmp")
        tmp.write_text(json.dumps(self._m))
        os.replace(tmp, self.root / MANIFEST)

    def add_layer(self, name: str, destinations: Sequence[str]) -> None:
        """Add a destination layer to an existing store (e.g. 'all' for a
        mode computed at buurt level for every destination)."""
        if name in self._m["layers"]:
            raise ValueError(f"Layer '{name}' already exists.")
        dests = [str(d) for d in destinations]
        if len(set(dests)) != len(dests):
            raise ValueError(f"Layer '{name}': duplicate destinations.")
        self._m["layers"][name] = {"destinations": dests, "cell_of": None}
        self._save()

    def set_meta(self, key: str, value) -> None:
        self._m["meta"][key] = value
        self._save()

    # ── description ──────────────────────────────────────────────────

    @property
    def origins(self) -> list[str]:
        return self._m["origins"]

    @property
    def meta(self) -> dict:
        return self._m["meta"]

    @property
    def layer_names(self) -> list[str]:
        return list(self._m["layers"])

    def layer(self, name: str) -> LayerInfo:
        try:
            raw = self._m["layers"][name]
        except KeyError:
            raise KeyError(f"No layer '{name}'; have {self.layer_names}.") \
                from None
        return LayerInfo(tuple(raw["destinations"]), raw["cell_of"])

    def _path(self, layer: str, mode: str, variable: str) -> Path:
        return self.root / layer / mode / f"{variable}.npy"

    def arrays(self) -> list[tuple[str, str, str]]:
        """(layer, mode, variable) of every matrix present."""
        out = []
        for p in sorted(self.root.glob("*/*/*.npy")):
            out.append((p.parts[-3], p.parts[-2], p.stem))
        return out

    # ── writing ──────────────────────────────────────────────────────

    def allocate(self, layer: str, mode: str, variable: str) -> None:
        """Create the matrix (NaN-filled) if it does not exist."""
        path = self._path(layer, mode, variable)
        if path.exists():
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        shape = (len(self.origins), len(self.layer(layer).destinations))
        mm = np.lib.format.open_memmap(path, mode="w+", dtype=DTYPE,
                                       shape=shape)
        mm[:] = np.nan
        mm.flush()
        del mm

    def write_rows(self, layer: str, mode: str, variable: str, start: int,
                   values: np.ndarray) -> None:
        """Write origin rows [start, start + len(values)) and record them
        as done."""
        values = np.asarray(values, dtype=DTYPE)
        n_dest = len(self.layer(layer).destinations)
        if values.ndim != 2 or values.shape[1] != n_dest:
            raise ValueError(f"Rows must have shape (k, {n_dest}), got "
                             f"{values.shape}.")
        stop = start + values.shape[0]
        if start < 0 or stop > len(self.origins):
            raise ValueError(f"Rows [{start}, {stop}) outside "
                             f"{len(self.origins)} origins.")
        self.allocate(layer, mode, variable)
        mm = np.load(self._path(layer, mode, variable), mmap_mode="r+")
        mm[start:stop] = values
        mm.flush()
        del mm
        self._mm_cache.pop((layer, mode, variable), None)
        key = f"{layer}/{mode}/{variable}"
        done = self._m["progress"].setdefault(key, [])
        done.append([start, stop])
        self._m["progress"][key] = _merge_intervals(done)
        self._save()

    def done_rows(self, layer: str, mode: str, variable: str) -> list[list[int]]:
        return self._m["progress"].get(f"{layer}/{mode}/{variable}", [])

    def pending_blocks(self, layer: str, mode: str, variable: str,
                       block_size: int) -> list[tuple[int, int]]:
        """Origin row blocks [start, stop) still to compute."""
        done = self.done_rows(layer, mode, variable)
        n = len(self.origins)
        covered = np.zeros(n, dtype=bool)
        for a, b in done:
            covered[a:b] = True
        blocks = []
        i = 0
        while i < n:
            j = min(i + block_size, n)
            if not covered[i:j].all():
                blocks.append((i, j))
            i = j
        return blocks

    # ── reading ──────────────────────────────────────────────────────

    def array(self, layer: str, mode: str, variable: str) -> np.ndarray:
        """Read-only memory map of the whole matrix."""
        key = (layer, mode, variable)
        if key not in self._mm_cache:
            path = self._path(*key)
            if not path.exists():
                raise FileNotFoundError(
                    f"No matrix {layer}/{mode}/{variable} in {self.root}.")
            self._mm_cache[key] = np.load(path, mmap_mode="r")
        return self._mm_cache[key]

    def block(self, layer: str, mode: str, variable: str, *,
              origins: Iterable[str] | None = None,
              destinations: Iterable[str] | None = None,
              fill: float | None = None) -> np.ndarray:
        """Matrix for chosen origins x destinations of one layer (all by
        default), as an in-memory array. Only the requested origin rows
        are read from disk. NaN cells become `fill` if given."""
        mm = self.array(layer, mode, variable)
        rows = _positions(self.origins, origins, "origin")
        cols = _positions(self.layer(layer).destinations, destinations,
                          f"destination of layer '{layer}'")
        out = np.asarray(mm[rows] if rows is not None else mm)
        if cols is not None:
            out = out[:, cols]
        out = np.array(out, dtype=DTYPE, copy=True)
        if fill is not None:
            out[np.isnan(out)] = fill
        return out

    def combined(self, mode: str, variable: str, destinations: Sequence[str],
                 *, near: str, far: str | None = None,
                 origins: Iterable[str] | None = None,
                 fill: float | None = None) -> np.ndarray:
        """Matrix over `destinations` (any codes covered by the far
        layer's `cell_of`): the near-layer value where the destination is
        in the near layer, else the value of its far cell."""
        near_dests = self.layer(near).destinations
        if far is None:
            # everything must be in the near layer
            return self.block(near, mode, variable, origins=origins,
                              destinations=destinations, fill=fill)
        far_info = self.layer(far)
        if far_info.cell_of is None:
            raise ValueError(f"Layer '{far}' has no cell_of mapping.")
        near_index = {d: i for i, d in enumerate(near_dests)}
        far_index = {d: i for i, d in enumerate(far_info.destinations)}

        dests = [str(d) for d in destinations]
        in_near = np.array([d in near_index for d in dests])
        unknown = [d for d, n in zip(dests, in_near)
                   if not n and d not in far_info.cell_of]
        if unknown:
            raise KeyError(f"{len(unknown)} destination(s) are in neither "
                           f"layer, e.g. {unknown[:3]}.")

        near_block = self.block(near, mode, variable, origins=origins,
                                destinations=[d for d, n in zip(dests, in_near)
                                              if n])
        far_cols = [far_index[far_info.cell_of[d]]
                    for d, n in zip(dests, in_near) if not n]
        far_block = self.block(far, mode, variable, origins=origins)[
            :, far_cols] if far_cols else np.empty((near_block.shape[0], 0),
                                                   dtype=DTYPE)
        out = np.empty((near_block.shape[0], len(dests)), dtype=DTYPE)
        out[:, in_near] = near_block
        out[:, ~in_near] = far_block
        if fill is not None:
            out[np.isnan(out)] = fill
        return out


def _merge_intervals(intervals: list[list[int]]) -> list[list[int]]:
    out: list[list[int]] = []
    for a, b in sorted(intervals):
        if out and a <= out[-1][1]:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return out


def _positions(index: Sequence[str], wanted, what: str):
    if wanted is None:
        return None
    lookup = {c: i for i, c in enumerate(index)}
    wanted = [str(w) for w in wanted]
    missing = [w for w in wanted if w not in lookup]
    if missing:
        raise KeyError(f"{len(missing)} unknown {what} code(s), e.g. "
                       f"{missing[:3]}.")
    return np.array([lookup[w] for w in wanted], dtype=np.intp)
