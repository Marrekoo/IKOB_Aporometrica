"""
Accessibility per segment, by mode: the glue between the pieces.

For every mode m, every job type w (jobs that do / do not admit working
from home) and every segment s (household type x income decile):

    a[i, s, m] = sum over w, j of  D[j, class(s), w]
                 * S_T(t_ij; m, w) * S_M(c_ij; s)          (M2: independent gates)

with S_T the mode's Weibull time margin for job type w, S_M the uniform
cost margin of the segment's reference budget (with its atom at zero),
and D the LISA jobs matched to the segment's income decile and split by
whether they admit home working. Free modes (bike) have no cost margin.
The measure is linear in the opportunities, so the two job types are
computed separately and added.

`run_accessibility` is pure: it takes matrices and tables, not files.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from ikob2.domain.filter_config import INDEPENDENCE, CopulaSpec
from ikob2.engine.runner import SegmentedRunner
from ikob2.segments.bridge import (
    build_segments,
    envelope_segment_names,
    segment_name,
)
from ikob2.segments.jobs import sector_income_weights, sector_pools
from ikob2.segments.wfh import split_jobs_by_wfh

logger = logging.getLogger(__name__)

WFH_TYPES = ("no_wfh", "wfh_possible")


@dataclass(frozen=True)
class ModeMatrices:
    """Origins x destinations matrices of one mode: travel time in
    minutes and, for priced modes, money cost in EUR per one-way trip.
    NaN or non-finite entries are treated as unreachable."""
    time: np.ndarray
    cost: np.ndarray | None = None
    cost_id: str | None = None

    def __post_init__(self):
        if self.cost is not None:
            if self.cost.shape != self.time.shape:
                raise ValueError("time and cost must have the same shape.")
            if self.cost_id is None:
                raise ValueError("A cost matrix needs a cost_id.")


@dataclass(frozen=True)
class AccessibilityResult:
    table: pd.DataFrame            # long: one row per origin x segment x mode
    meta: dict = field(default_factory=dict)

    def summary(self, by: str = "income_class",
                value: str = "accessibility") -> pd.DataFrame:
        """Population-weighted mean per mode and group, over all origins."""
        t = self.table
        w = t["population"].to_numpy()
        g = t.assign(_num=t[value] * w, _den=w).groupby(["mode", by])
        s = g[["_num", "_den"]].sum()
        s[value] = s["_num"] / s["_den"].where(s["_den"] > 0)
        return s[[value]].join(g["population"].sum())


def run_accessibility(
    *,
    origins: Sequence[str],
    destinations: Sequence[str],
    populations: pd.DataFrame,
    sector_jobs: pd.DataFrame,
    wfh_share: pd.Series,
    sector_wage: pd.Series,
    envelope: pd.DataFrame,
    time_margins: Mapping,
    matrices: Mapping[str, ModeMatrices],
    copula: CopulaSpec = INDEPENDENCE,
    epsilon: float | None = 1e-9,
    unreachable_minutes: float = 1e4,
) -> AccessibilityResult:
    """Accessibility of every origin, segment and mode.

    populations : persons per segment column (household_type_class) per
        origin buurt, indexed or keyed by `buurtcode`.
    sector_jobs : imputed jobs, buurt x LISA sector (covering the
        destinations; buurten missing there have no jobs).
    wfh_share / sector_wage : per LISA sector.
    envelope : validated cost-margin table (only its segments are run).
    time_margins : {(mode, wfh): CurveSpec} (segments.time_margins).
    matrices : mode -> ModeMatrices over (origins, destinations).
    """
    origins = [str(o) for o in origins]
    destinations = [str(d) for d in destinations]
    n_o, n_d = len(origins), len(destinations)
    names = envelope_segment_names(envelope)

    pop = populations
    if "buurtcode" in pop.columns:
        pop = pop.set_index("buurtcode")
    pop = pop.reindex(origins)
    missing_pop = [n for n in names if n not in pop.columns]
    if missing_pop:
        raise KeyError(f"Populations lack segment column(s) "
                       f"{missing_pop[:5]}.")

    # income pools for both job types, one weight table (partition)
    W = sector_income_weights(sector_wage, sector_jobs.sum())
    jobs_by_type = dict(zip(WFH_TYPES,
                            split_jobs_by_wfh(sector_jobs, wfh_share)))
    used = {n.rsplit("_", 1)[1] for n in names}
    pools = {w: {c: v for c, v in
                 sector_pools(jobs_by_type[w], destinations, W).items()
                 if c in used}                 # 'onbekend' has no segment
             for w in WFH_TYPES}

    runner = SegmentedRunner(decay_epsilon=epsilon)
    rows = []
    for mode, mm in matrices.items():
        if mm.time.shape != (n_o, n_d):
            raise ValueError(f"Mode '{mode}' matrix {mm.time.shape} does "
                             f"not match ({n_o}, {n_d}) origins x "
                             f"destinations.")
        cost_matrices = {"time": _finite(mm.time, unreachable_minutes)}
        if mm.cost is not None:
            cost_matrices[mm.cost_id] = _finite(mm.cost, unreachable_minutes)
        total = {n: np.zeros(n_o) for n in names}
        for wfh in WFH_TYPES:
            spec = time_margins.get((mode, wfh))
            if spec is None:
                raise KeyError(f"No time margin for ({mode}, {wfh}); "
                               f"have {sorted(time_margins)}.")
            segs = build_segments(
                spec,
                envelope=envelope if mm.cost is not None else None,
                money_cost_id=mm.cost_id if mm.cost is not None else None,
                copula=copula if mm.cost is not None else INDEPENDENCE,
                pool_by="income_class", only=names)
            per = runner.run_hansen(None, segs, cost_matrices=cost_matrices,
                                    opportunities=pools[wfh])
            for n in names:
                total[n] += per[n].astype(np.float64)
            logger.info("mode %s, %s: %d segments, %d composed filters",
                        mode, wfh, len(segs),
                        len({s.weight_key for s in segs}))
        rows.append(_long(mode, origins, names, total, pop, envelope,
                          priced=mm.cost is not None))
    table = pd.concat(rows, ignore_index=True)
    meta = {"origins": n_o, "destinations": n_d, "segments": len(names),
            "modes": list(matrices), "epsilon": epsilon,
            "copula": copula.family, "theta": copula.theta,
            "jobs_total": float(sector_jobs.sum().sum())}
    return AccessibilityResult(table, meta)


def _finite(a: np.ndarray, fill: float) -> np.ndarray:
    out = np.asarray(a, dtype=np.float32).copy()
    out[~np.isfinite(out)] = fill
    return out


def _long(mode, origins, names, total, pop, envelope, *, priced):
    atom = {segment_name(r.household_type, r.income_class): float(r.atom)
            for r in envelope.itertuples()}
    parts = []
    for n in names:
        ht, ic = n.rsplit("_", 1)[0], n.rsplit("_", 1)[1]
        # 'couple_children_D3' -> household 'couple_children', class 'D3'
        a = atom[n] if priced else 0.0
        raw = total[n]
        parts.append(pd.DataFrame({
            "buurtcode": origins, "mode": mode, "segment": n,
            "household_type": ht, "income_class": ic,
            "population": pop[n].fillna(0.0).to_numpy(),
            "accessibility": raw,
            "atom": a,
            "accessibility_normalised": (raw / (1.0 - a) if a < 1.0
                                         else np.nan),
        }))
    return pd.concat(parts, ignore_index=True)
