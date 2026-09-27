"""
Zone geometry and CBS attribute container.

ZoneSet is the index every other loader (jobs, skims, segment shares)
joins onto: `codes[i]` is the identity of CBS buurt i everywhere
downstream, and `code_to_index` is the join key.

Centroids are stored rather than full geometries: the accessibility
engine works on zone-to-zone matrices (skims), not polygons, and a
centroid is all a network router or a nearest-hub lookup needs. Full
geometries stay in the source GeoPackage for anyone who needs them
for mapping.
"""

from dataclasses import dataclass, field
from typing import Any, Dict

import numpy as np


@dataclass(frozen=True)
class ZoneSet:
    codes: np.ndarray                      # (n,) str, e.g. "BU03449000"
    names: np.ndarray                      # (n,) str
    centroid_x: np.ndarray                 # (n,) float64, CRS units (metres)
    centroid_y: np.ndarray                 # (n,) float64, CRS units (metres)
    crs: str                               # e.g. "EPSG:28992"
    municipality_code: np.ndarray | None = None   # (n,) str
    municipality_name: np.ndarray | None = None   # (n,) str
    attributes: Dict[str, np.ndarray] = field(default_factory=dict)

    def __post_init__(self):
        n = len(self.codes)
        for name, arr in (
            ("names", self.names),
            ("centroid_x", self.centroid_x),
            ("centroid_y", self.centroid_y),
        ):
            if len(arr) != n:
                raise ValueError(
                    f"ZoneSet.{name} has length {len(arr)}, expected {n} "
                    f"(len(codes))"
                )
        for name, arr in self.attributes.items():
            if len(arr) != n:
                raise ValueError(
                    f"ZoneSet.attributes[{name!r}] has length {len(arr)}, "
                    f"expected {n} (len(codes))"
                )
        if len(set(self.codes)) != n:
            dupes = sorted({c for c in self.codes
                             if list(self.codes).count(c) > 1})
            raise ValueError(
                f"ZoneSet.codes contains {len(dupes)} duplicate code(s), "
                f"e.g. {dupes[:5]}; zone identity must be unique."
            )

    @property
    def n_zones(self) -> int:
        return len(self.codes)

    @property
    def code_to_index(self) -> Dict[str, int]:
        """Join key for other loaders: buurtcode -> row position."""
        return {code: i for i, code in enumerate(self.codes)}

    def attribute(self, name: str) -> np.ndarray:
        try:
            return self.attributes[name]
        except KeyError:
            raise KeyError(
                f"ZoneSet has no attribute '{name}'; available: "
                f"{sorted(self.attributes)}"
            ) from None

    def subset(self, mask: np.ndarray) -> "ZoneSet":
        """New ZoneSet restricted to codes[mask], preserving order."""
        mask = np.asarray(mask, dtype=bool)
        return ZoneSet(
            codes=self.codes[mask],
            names=self.names[mask],
            centroid_x=self.centroid_x[mask],
            centroid_y=self.centroid_y[mask],
            crs=self.crs,
            municipality_code=(
                None if self.municipality_code is None
                else self.municipality_code[mask]
            ),
            municipality_name=(
                None if self.municipality_name is None
                else self.municipality_name[mask]
            ),
            attributes={k: v[mask] for k, v in self.attributes.items()},
        )

    def reorder(self, codes: list[str] | np.ndarray) -> "ZoneSet":
        """New ZoneSet permuted to match an external ordering (e.g. a
        skim's zone order). Every code in `codes` must be present."""
        idx = self.code_to_index
        missing = [c for c in codes if c not in idx]
        if missing:
            raise KeyError(
                f"{len(missing)} code(s) not in this ZoneSet, e.g. "
                f"{missing[:5]}"
            )
        positions = np.array([idx[c] for c in codes], dtype=np.intp)
        return ZoneSet(
            codes=self.codes[positions],
            names=self.names[positions],
            centroid_x=self.centroid_x[positions],
            centroid_y=self.centroid_y[positions],
            crs=self.crs,
            municipality_code=(
                None if self.municipality_code is None
                else self.municipality_code[positions]
            ),
            municipality_name=(
                None if self.municipality_name is None
                else self.municipality_name[positions]
            ),
            attributes={k: v[positions] for k, v in self.attributes.items()},
        )
