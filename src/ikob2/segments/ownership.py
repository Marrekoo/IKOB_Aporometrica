"""
Private bicycle ownership per buurt.

Input: a CSV with one row per buurt and the percentage of residents with one
or more bicycles (`pct_with_bicycle`, 0-100), assigned from the
Utrecht buurtteam survey (2025). Extra columns (names, `buurtteam`,
`mapping_confidence`) are documentation and ignored.

The loader returns a share in [0, 1] per buurt: the probability that a
resident has a private bicycle, i.e. the availability of the bicycle mode
and the complement of the group for whom a shared bicycle is the option.
Missing or blank values are an error unless `fill` is given.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

COLUMN = "pct_with_bicycle"


def load_bike_ownership(path: str | Path,
                        buurtcodes: Sequence[str] | None = None,
                        *, fill: float | None = None) -> pd.Series:
    """Share of residents with a private bicycle per buurt, indexed by
    `buurtcode`, as a fraction in [0, 1].

    buurtcodes : if given, the result is reindexed to these codes and every
        code must be in the file (or `fill` is used).
    fill : fraction (0-1) for buurten that are blank or absent; None makes
        them an error.
    """
    df = pd.read_csv(path, dtype={"buurtcode": str})
    for col in ("buurtcode", COLUMN):
        if col not in df.columns:
            raise KeyError(f"{path}: column '{col}' missing; have "
                           f"{list(df.columns)}.")
    if df["buurtcode"].duplicated().any():
        dup = df.loc[df["buurtcode"].duplicated(), "buurtcode"].tolist()
        raise ValueError(f"{path}: duplicate buurtcode(s), e.g. {dup[:5]}.")
    pct = pd.to_numeric(df[COLUMN], errors="coerce")
    bad = pct.notna() & ((pct < 0) | (pct > 100))
    if bad.any():
        raise ValueError(f"{path}: {COLUMN} must be a percentage in 0-100; "
                         f"got {pct[bad].tolist()[:5]}.")
    share = (pct / 100.0).set_axis(df["buurtcode"])
    if buurtcodes is not None:
        share = share.reindex([str(c) for c in buurtcodes])
    if share.isna().any():
        missing = share.index[share.isna()].tolist()
        if fill is None:
            raise ValueError(f"{path}: no {COLUMN} for {len(missing)} "
                             f"buurt(en), e.g. {missing[:5]}.")
        if not 0.0 <= fill <= 1.0:
            raise ValueError("fill is a fraction in [0, 1].")
        share = share.fillna(fill)
    share.name = "bike_share"
    return share


def availability_frames(origins: Sequence[str], segment_names: Sequence[str],
                        *, bike_share: pd.Series | None = None,
                        car_table: pd.DataFrame | None = None,
                        ) -> dict[str, pd.DataFrame]:
    """Availability of each mode as origins x segments frames in [0, 1], for
    `run_accessibility(availability=...)`. Modes without a source are absent
    (available to everyone).

    bike_share : per origin buurt (`load_bike_ownership`), the same for every
        segment.
    car_table : `car_availability` output (household_type, income_class,
        share), the same for every origin.
    """
    origins = [str(o) for o in origins]
    out: dict[str, pd.DataFrame] = {}
    if bike_share is not None:
        s = bike_share.reindex(origins)
        if s.isna().any():
            raise ValueError(f"No bicycle share for {int(s.isna().sum())} "
                             f"origin(s), e.g. {s.index[s.isna()][:3].tolist()}.")
        out["bike"] = pd.DataFrame(
            {n: s.to_numpy() for n in segment_names}, index=origins)
    if car_table is not None:
        by = {f"{r.household_type}_{r.income_class}": float(r.share)
              for r in car_table.itertuples()}
        missing = [n for n in segment_names if n not in by]
        if missing:
            raise KeyError(f"Car availability lacks segment(s) {missing[:5]}.")
        out["car"] = pd.DataFrame(
            {n: np.full(len(origins), by[n]) for n in segment_names},
            index=origins)
    return out
