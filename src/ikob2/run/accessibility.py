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
from ikob2.segments.specs import (
    SPECS, atom_reported, cost_curve_factory, spec_copula, time_margin_for,
    vot_weighted_cost)
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
    rail_share: np.ndarray | None = None   # share of km by rail (PT, for M1)

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
    envelope: pd.DataFrame | None,
    time_margins: Mapping,
    matrices: Mapping[str, ModeMatrices],
    copula: CopulaSpec = INDEPENDENCE,
    epsilon: float | None = 1e-9,
    unreachable_minutes: float = 1e4,
    segment_names: Sequence[str] | None = None,
    spec: str = "m2",
    theta: float | None = None,
    vot: Mapping[str, float] | None = None,
    availability: Mapping[str, pd.DataFrame] | None = None,
) -> AccessibilityResult:
    """Accessibility of every origin, segment and mode.

    populations : persons per segment column (household_type_class) per
        origin buurt, indexed or keyed by `buurtcode`.
    sector_jobs : imputed jobs, buurt x LISA sector (covering the
        destinations; buurten missing there have no jobs).
    wfh_share / sector_wage : per LISA sector.
    envelope : validated cost-margin table (only its segments are run);
        None switches the cost gate off: time-only accessibility, cost
        matrices are ignored, and `segment_names` says which segments to
        run.
    time_margins : {(mode, wfh): CurveSpec} (segments.time_margins).
    matrices : mode -> ModeMatrices over (origins, destinations).
    spec : 'm1', 'm1p', 'm2' (default) or 'm3' (segments.specs). M3 uses a
        Gumbel-Hougaard copula with `theta` (inf: comonotone); `copula`
        applies to M2 only. M1 needs `vot`: mode -> value of time in
        EUR/hour.
    availability : mode -> origins x segments frame in [0, 1]: the share of
        the segment at that origin that can use the mode (car in the
        household, private bicycle). `accessibility` stays conditional on
        having the mode; `availability` and `accessibility_expected` (their
        product) are added. Modes not listed have availability 1.
    """
    if spec not in SPECS:
        raise ValueError(f"Unknown specification {spec!r}; use {SPECS}.")
    if spec == "m3":
        copula = spec_copula(spec, theta)
    elif spec != "m2":
        copula = INDEPENDENCE
    origins = [str(o) for o in origins]
    destinations = [str(d) for d in destinations]
    n_o, n_d = len(origins), len(destinations)
    if envelope is not None:
        names = envelope_segment_names(envelope)
    elif segment_names is not None:
        names = list(segment_names)
    else:
        raise ValueError("Without an envelope, give segment_names.")

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
        gated = mm.cost is not None and envelope is not None
        cost_matrices = {"time": _finite(mm.time, unreachable_minutes)}
        cost_id = mm.cost_id
        mode_vot = (vot or {}).get(mode)
        if gated:
            cost = mm.cost
            if (spec == "m1" and mm.rail_share is not None
                    and f"{mode}_other" in (vot or {})):
                cost = vot_weighted_cost(cost, mm.rail_share, vot[mode],
                                         vot[f"{mode}_other"])
                cost_id = f"{mm.cost_id}@vot"
            cost_matrices[cost_id] = _finite(cost, unreachable_minutes)
        total = {n: np.zeros(n_o) for n in names}
        for wfh in WFH_TYPES:
            margin = time_margins.get((mode, wfh))
            if margin is None:
                raise KeyError(f"No time margin for ({mode}, {wfh}); "
                               f"have {sorted(time_margins)}.")
            segs = build_segments(
                time_margin_for(spec, margin),
                envelope=envelope if gated else None,
                money_cost_id=cost_id if gated else None,
                copula=copula if gated else INDEPENDENCE,
                pool_by="income_class", only=names,
                cost_curve=(cost_curve_factory(spec, margin,
                                               mode_vot)
                            if gated else None))
            per = runner.run_hansen(None, segs, cost_matrices=cost_matrices,
                                    opportunities=pools[wfh])
            for n in names:
                total[n] += per[n].astype(np.float64)
            logger.info("mode %s, %s: %d segments, %d composed filters",
                        mode, wfh, len(segs),
                        len({s.weight_key for s in segs}))
        avail = None
        if availability and mode in availability:
            avail = availability[mode].reindex(index=origins,
                                               columns=list(names))
            if avail.isna().any().any() or ((avail < 0) | (avail > 1)).any().any():
                raise ValueError(f"Availability of '{mode}' must cover all "
                                 f"origins and segments, within [0, 1].")
        rows.append(_long(mode, origins, names, total, pop, envelope,
                          priced=gated and atom_reported(spec), avail=avail))
    table = pd.concat(rows, ignore_index=True)
    meta = {"origins": n_o, "destinations": n_d, "segments": len(names),
            "modes": list(matrices), "epsilon": epsilon,
            "copula": copula.family, "theta": copula.theta,
            "spec": spec, "vot_eur_per_hour": dict(vot or {}),
            "jobs_total": float(sector_jobs.sum().sum())}
    return AccessibilityResult(table, meta)


def _finite(a: np.ndarray, fill: float) -> np.ndarray:
    out = np.asarray(a, dtype=np.float32).copy()
    out[~np.isfinite(out)] = fill
    return out


def _long(mode, origins, names, total, pop, envelope, *, priced, avail=None):
    atom = ({segment_name(r.household_type, r.income_class): float(r.atom)
             for r in envelope.itertuples()} if envelope is not None else {})
    parts = []
    for n in names:
        ht, ic = n.rsplit("_", 1)[0], n.rsplit("_", 1)[1]
        # 'couple_children_D3' -> household 'couple_children', class 'D3'
        a = atom.get(n, 0.0) if priced else 0.0
        raw = total[n]
        parts.append(pd.DataFrame({
            "buurtcode": origins, "mode": mode, "segment": n,
            "household_type": ht, "income_class": ic,
            "population": pop[n].fillna(0.0).to_numpy(),
            "accessibility": raw,
            "atom": a,
            "accessibility_normalised": (raw / (1.0 - a) if a < 1.0
                                         else np.nan),
            "availability": (1.0 if avail is None else avail[n].to_numpy()),
        }))
        parts[-1]["accessibility_expected"] = (parts[-1]["accessibility"]
                                               * parts[-1]["availability"])
    return pd.concat(parts, ignore_index=True)
