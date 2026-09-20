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

import itertools
import logging
from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from ikob2.domain.filter_config import INDEPENDENCE, CopulaSpec
from ikob2.engine.runner import SegmentedRunner, evaluate_marginal
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
class OptionSet:
    """Alternative journeys between the same origins and destinations that
    one person can choose from (for example plain public transport, and
    public transport with a shared bicycle at the destination). The person
    accepts a pair if any option passes both gates. `margin_mode` names the
    time margin to use (a chain is judged on its total time)."""
    options: tuple
    margin_mode: str = "pt"


@dataclass(frozen=True)
class LegOption:
    """One journey judged leg by leg: `times` has one origins x destinations
    matrix per leg (0 where the leg does not exist) and `cost` the price of
    the whole journey."""
    times: tuple
    cost: np.ndarray | None = None
    cost_id: str | None = None


@dataclass(frozen=True)
class LegOptionSet:
    """Alternatives judged leg-wise (paper eq. 3.2): each leg has its own
    time threshold (margin of `margin_modes[l]`) and the cost margin is on
    the journey total. With independent thresholds the joint survival of an
    option is prod_l S_T,l(t_l) x S_M(c); a person accepts a pair if any
    option passes, computed by inclusion-exclusion over the options."""
    options: tuple
    margin_modes: tuple = ("bike", "bike", "pt")


@dataclass(frozen=True)
class MixedMode:
    """A mode that is a mixture over groups of the population that have
    different options: parts = ((share per origin, OptionSet), ...), the
    shares of an origin summing to 1 (for example residents with and
    without a private bicycle)."""
    parts: tuple


def union_terms(times: np.ndarray, costs: np.ndarray):
    """Terms (time, cost, sign) whose joint-survival sum is the probability
    that at least one option is acceptable to a person with random
    thresholds (tau, mu).

    An option (t, c) is acceptable iff tau >= t and mu >= c. Sorted by time
    per origin-destination pair, with the running cheapest cost cmin_k of
    the options at least as fast, the acceptable region is the staircase
    union of [t_k, inf) x [cmin_k, inf), so

        P = sum_k f(t_k, cmin_k) - sum_{k<K} f(t_{k+1}, cmin_k)

    exactly, for any dependence between the thresholds. times, costs:
    (K, origins, destinations)."""
    order = np.argsort(times, axis=0, kind="stable")
    ts = np.take_along_axis(times, order, axis=0)
    cs = np.take_along_axis(costs, order, axis=0)
    cmin = np.minimum.accumulate(cs, axis=0)
    k = len(ts)
    terms = [(ts[i], cmin[i], 1.0) for i in range(k)]
    terms += [(ts[i + 1], cmin[i], -1.0) for i in range(k - 1)]
    return terms


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

    def hansen_total(margin_mode, time, cost, cost_id, rail_share=None):
        """Per-segment accessibility of one (time, cost) pair, both job
        types added."""
        gated = cost is not None and envelope is not None
        cost_matrices = {"time": _finite(time, unreachable_minutes)}
        mode_vot = (vot or {}).get(margin_mode)
        if gated:
            if (spec == "m1" and rail_share is not None
                    and f"{margin_mode}_other" in (vot or {})):
                cost = vot_weighted_cost(cost, rail_share, vot[margin_mode],
                                         vot[f"{margin_mode}_other"])
                cost_id = f"{cost_id}@vot"
            cost_matrices[cost_id] = _finite(cost, unreachable_minutes)
        total = {n: np.zeros(n_o) for n in names}
        for wfh in WFH_TYPES:
            margin = time_margins.get((margin_mode, wfh))
            if margin is None:
                raise KeyError(f"No time margin for ({margin_mode}, {wfh}); "
                               f"have {sorted(time_margins)}.")
            segs = build_segments(
                time_margin_for(spec, margin),
                envelope=envelope if gated else None,
                money_cost_id=cost_id if gated else None,
                copula=copula if gated else INDEPENDENCE,
                pool_by="income_class", only=names,
                cost_curve=(cost_curve_factory(spec, margin, mode_vot)
                            if gated else None))
            per = runner.run_hansen(None, segs, cost_matrices=cost_matrices,
                                    opportunities=pools[wfh])
            for n in names:
                total[n] += per[n].astype(np.float64)
            logger.info("mode %s, %s: %d segments, %d composed filters",
                        margin_mode, wfh, len(segs),
                        len({s.weight_key for s in segs}))
        return total

    def check(mm, label):
        if mm.time.shape != (n_o, n_d):
            raise ValueError(f"Mode '{label}' matrix {mm.time.shape} does "
                             f"not match ({n_o}, {n_d}) origins x "
                             f"destinations.")

    def option_total(margin_mode, options):
        """Union of alternative journeys for one person (staircase of
        joint survival, see `union_terms`)."""
        if spec == "m1":
            raise ValueError("Option sets are not defined for M1 (its "
                             "value of time is per journey).")
        for o in options:
            check(o, margin_mode)
        priced = envelope is not None and any(o.cost is not None
                                              for o in options)
        ts = np.stack([_finite(o.time, unreachable_minutes) for o in options])
        cs = np.stack([_finite(o.cost if o.cost is not None
                               else np.zeros_like(o.time),
                               unreachable_minutes) for o in options])
        cid = next((o.cost_id for o in options if o.cost is not None), None)
        total = {n: np.zeros(n_o) for n in names}
        for t, c, sign in union_terms(ts, cs):
            part = hansen_total(margin_mode, t, c if priced else None, cid)
            for n in names:
                total[n] += sign * part[n]
        return total, priced

    def legwise_total(oset):
        """Leg-wise gates, union by inclusion-exclusion (independent
        thresholds, M2)."""
        if spec != "m2" or copula.family != "independence":
            raise ValueError("Leg-wise gates are defined for M2 with "
                             "independent thresholds.")
        opts = oset.options
        n_leg = len(oset.margin_modes)
        for o in opts:
            if len(o.times) != n_leg:
                raise ValueError("Every option needs one time per leg.")
            for tt in o.times:
                check(ModeMatrices(np.asarray(tt)), "leg")
        priced = envelope is not None and any(o.cost is not None for o in opts)
        times = np.stack([[_finite(t, unreachable_minutes) for t in o.times]
                          for o in opts])                  # (K, L, o, d)
        costs = np.stack([_finite(o.cost if o.cost is not None
                                  else np.zeros((n_o, n_d)),
                                  unreachable_minutes) for o in opts])
        quiet = logging.getLogger("ikob2.engine.runner")
        level, quiet.level = quiet.level, logging.WARNING
        try:
            segs = build_segments(time_margins[(oset.margin_modes[0],
                                                WFH_TYPES[0])],
                                  envelope=envelope if priced else None,
                                  money_cost_id="c" if priced else None,
                                  pool_by="income_class", only=names)
            total = {n: np.zeros(n_o) for n in names}
            for r in range(1, len(opts) + 1):
                for subset in itertools.combinations(range(len(opts)), r):
                    sign = 1.0 if r % 2 else -1.0
                    idx = list(subset)
                    t = times[idx].max(axis=0)                   # (L, o, d)
                    c = costs[idx].max(axis=0)
                    tw = {}
                    for wfh in WFH_TYPES:
                        w = None
                        for leg, mode_l in enumerate(oset.margin_modes):
                            spec_l = time_margins[(mode_l, wfh)]
                            f = evaluate_marginal(t[leg], spec_l)
                            w = f if w is None else w * f
                        tw[wfh] = w
                    for sg in segs:
                        sm = (evaluate_marginal(c, sg.class_filter.cost)
                              if priced else None)
                        for wfh in WFH_TYPES:
                            wgt = tw[wfh] if sm is None else tw[wfh] * sm
                            total[sg.name] += sign * (
                                wgt @ np.asarray(pools[wfh][sg.pool], dtype=np.float32))
        finally:
            quiet.level = level
        return total, priced

    def set_total(oset):
        if isinstance(oset, LegOptionSet):
            return legwise_total(oset)
        return option_total(oset.margin_mode, oset.options)

    rows = []
    for mode, mm in matrices.items():
        priced = False
        if isinstance(mm, MixedMode):
            total = {n: np.zeros(n_o) for n in names}
            for weight, oset in mm.parts:
                w = np.asarray(weight, dtype=float)
                if w.shape != (n_o,) or (w < 0).any() or (w > 1).any():
                    raise ValueError("Mixture weights need one share in "
                                     "[0, 1] per origin.")
                part, pr = set_total(oset)
                priced = priced or pr
                for n in names:
                    total[n] += w * part[n]
        elif isinstance(mm, (OptionSet, LegOptionSet)):
            total, priced = set_total(mm)
        else:
            check(mm, mode)
            total = hansen_total(mode, mm.time, mm.cost, mm.cost_id,
                                 mm.rail_share)
            priced = mm.cost is not None and envelope is not None
        avail = None
        if availability and mode in availability:
            avail = availability[mode].reindex(index=origins,
                                               columns=list(names))
            if avail.isna().any().any() or ((avail < 0) | (avail > 1)).any().any():
                raise ValueError(f"Availability of '{mode}' must cover all "
                                 f"origins and segments, within [0, 1].")
        rows.append(_long(mode, origins, names, total, pop, envelope,
                          priced=priced and atom_reported(spec), avail=avail))
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
