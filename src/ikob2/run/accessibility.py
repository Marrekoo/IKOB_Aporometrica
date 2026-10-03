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

import contextlib
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from ikob2.domain.filter_config import INDEPENDENCE, CopulaSpec
from ikob2.engine.runner import SegmentedRunner, evaluate_marginal
from ikob2.params import DEFAULTS
from ikob2.segments.bridge import (
    build_segments,
    envelope_segment_names,
    segment_name,
)
from ikob2.segments.jobs import sector_pools
from ikob2.core.compose import apply_copula
from ikob2.segments.specs import (
    SPECS, atom_reported, cost_curve_factory, reported_atoms, spec_copula,
    time_margin_for, vot_factor, vot_weighted_cost)

logger = logging.getLogger(__name__)

n_threads = DEFAULTS.numerics.threads or (os.cpu_count() or 1)

WFH_TYPES = tuple(DEFAULTS.accessibility.wfh_types)


@dataclass(frozen=True)
class ModeMatrices:
    """Origins x destinations matrices of one mode: travel time in
    minutes and, for priced modes, money cost in EUR per one-way trip.
    NaN or non-finite entries are treated as unreachable."""
    time: np.ndarray
    cost: np.ndarray | None = None
    cost_id: str | None = None
    rail_share: np.ndarray | None = None   # share of km by rail (PT, for M1)
    # part of `cost` paid to the shared-bicycle operator whose prices can be
    # scaled per segment (Lime), and the number of such rentals
    lime: np.ndarray | None = None
    lime_rentals: float = 0.0
    fare: np.ndarray | None = None   # public transport fare part of `cost` (scalable per segment)

    def __post_init__(self):
        if self.cost is not None:
            if self.cost.shape != self.time.shape:
                raise ValueError("time and cost must have the same shape.")
            if self.cost_id is None:
                raise ValueError("A cost matrix needs a cost_id.")
        for part in (self.lime, self.fare):
            if part is not None and (self.cost is None or
                                     np.shape(part) != self.time.shape):
                raise ValueError("lime and fare need a cost of the same shape.")


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
    the whole journey; `lime` and `lime_rentals` as in ModeMatrices."""
    times: tuple
    cost: np.ndarray | None = None
    cost_id: str | None = None
    lime: np.ndarray | None = None
    lime_rentals: float = 0.0
    rail_share: np.ndarray | None = None    # share of km by rail (for M1)
    fare: np.ndarray | None = None          # public transport fare part of `cost`


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
    """Result of `run_accessibility`: the long table (one row per origin x
    segment x mode) and run metadata."""
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


def _single_threaded_blas():
    """One BLAS thread per worker thread: without it the workers' matrix
    products spawn n_threads x n_cores threads that only get in each other's
    way (about 2x slower). A no-op when threadpoolctl is not installed."""
    try:
        from threadpoolctl import threadpool_limits
    except ImportError:            # pragma: no cover
        return contextlib.nullcontext()
    return threadpool_limits(limits=1, user_api="blas")


def prepare_inputs(origins, destinations, populations, sector_jobs,
                   job_weights, envelope, segment_names, common_jobs=False):
    """(segment names, populations by origin, job pools by job type and
    income class) of a run; shared by the accessibility run and the
    scenario calibration. `job_weights` maps each job type to its weights
    income class x sector (segments.occupations.JobMatching.by_type)."""
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
    # income pools for both job types: the weights partition the jobs
    lacking = [w for w in WFH_TYPES if w not in job_weights]
    if lacking:
        raise KeyError(f"job_weights lacks job type(s) {lacking}.")
    used = {n.rsplit("_", 1)[1] for n in names}
    all_pools = {w: sector_pools(sector_jobs, destinations, job_weights[w])
                 for w in WFH_TYPES}
    pools = {w: {c: v for c, v in all_pools[w].items()
                 if c in used}                 # 'onbekend' has no segment
             for w in WFH_TYPES}
    if common_jobs:
        # controlled comparison: every segment reaches the same jobs, the
        # income matching of the opportunities is switched off
        for w in WFH_TYPES:
            everything = sum(np.asarray(v, dtype=np.float64)
                             for c, v in all_pools[w].items()
                             if c != "onbekend")
            pools[w] = {c: everything for c in pools[w]}
    return names, pop, pools


def run_accessibility(
    *,
    origins: Sequence[str],
    destinations: Sequence[str],
    populations: pd.DataFrame,
    sector_jobs: pd.DataFrame,
    job_weights: Mapping[str, pd.DataFrame],
    envelope: pd.DataFrame | None,
    time_margins: Mapping,
    matrices: Mapping[str, ModeMatrices],
    copula: CopulaSpec = INDEPENDENCE,
    epsilon: float | None = DEFAULTS.accessibility.epsilon,
    unreachable_minutes: float = DEFAULTS.accessibility.unreachable_minutes,
    segment_names: Sequence[str] | None = None,
    spec: str = DEFAULTS.accessibility.spec,
    theta: float | None = None,
    vot: Mapping[str, float] | None = None,
    availability: Mapping[str, pd.DataFrame] | None = None,
    price_scale: Mapping[str, float] | None = None,
    common_jobs: bool = False,
    cost_mean_eur: float | None = None,
    fare_scale: Mapping[str, float] | None = None,
    cost_cutoff_eur: float | None = None,
    cutoff_share: float = DEFAULTS.accessibility.cutoff_share,
) -> AccessibilityResult:
    """Accessibility of every origin, segment and mode.

    populations : persons per segment column (household_type_class) per
        origin buurt, indexed or keyed by `buurtcode`.
    sector_jobs : imputed jobs, buurt x LISA sector (covering the
        destinations; buurten missing there have no jobs).
    job_weights : job type -> weights income class x LISA sector, which
        place the jobs in the income-matched pools
        (segments.occupations: `occupation_job_weights`,
        `sector_job_weights`).
    envelope : validated cost-margin table (only its segments are run);
        None switches the cost gate off: time-only accessibility, cost
        matrices are ignored, and `segment_names` says which segments to
        run.
    time_margins : {(mode, wfh): CurveSpec} (segments.time_margins).
    matrices : mode -> ModeMatrices over (origins, destinations).
    spec : 'm0', 'm0u', 'm0s', 'm1', 'm1c', 'm1p', 'm2' (default) or 'm3'
        (segments.specs). M3 uses a Gumbel-Hougaard copula with `theta`
        (inf: comonotone); `copula` applies to M2 only. M0 and M1 need
        `vot`: mode -> value of time in EUR/hour.
    availability : mode -> origins x segments frame in [0, 1]: the share of
        the segment at that origin that can use the mode (car in the
        household, private bicycle). `accessibility` stays conditional on
        having the mode; `availability` and `accessibility_expected` (their
        product) are added. Modes not listed have availability 1.
    price_scale : segment name -> multiplier on the shared-bicycle operator
        part (`lime`) of the journey cost, for segments with a concessionary
        price (default 1). Segments with equal scales share cost matrices.
    fare_scale : segment name -> multiplier on the public transport fare part
        (`fare`) of the journey cost, for a fare concession (default 1).
    cost_mean_eur : M1c: the calibrated mean acceptable cost (EUR per trip)
        of the shared exponential cost margin.
    cost_cutoff_eur : M0u: the cost cut-off (EUR per trip) shared by all
        segments.
    cutoff_share : M0, M0u, M0s: the cut-offs are the quantiles of the
        margins that this share of the thresholds reaches (0.5: medians).
    common_jobs : every segment reaches all jobs (no income matching): the
        controlled comparison of the paper, in which R does not vary by
        segment under M1.
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
    names, pop, pools = prepare_inputs(
        origins, destinations, populations, sector_jobs, job_weights,
        envelope, segment_names, common_jobs)

    runner = SegmentedRunner(decay_epsilon=epsilon)

    def hansen_total(margin_mode, time, cost, cost_id, rail_share=None,
                     seg_names=None):
        """Per-segment accessibility of one (time, cost) pair, both job
        types added (for `seg_names`, default all segments)."""
        sn = names if seg_names is None else seg_names
        gated = cost is not None and envelope is not None and spec != "m0"
        if spec == "m0" and cost is not None and envelope is not None:
            # generalised time: the cost in minutes at the value of time
            time = np.asarray(time, dtype=np.float32) + _cost_minutes(
                cost, rail_share, margin_mode)
        cost_matrices = {"time": _finite(time, unreachable_minutes)}
        mode_vot = (vot or {}).get(margin_mode)
        if gated:
            if (spec == "m1" and rail_share is not None
                    and f"{margin_mode}_other" in (vot or {})):
                cost = vot_weighted_cost(cost, rail_share, vot[margin_mode],
                                         vot[f"{margin_mode}_other"])
                cost_id = f"{cost_id}@vot"
            cost_matrices[cost_id] = _finite(cost, unreachable_minutes)
        total = {n: np.zeros(n_o) for n in sn}
        # both job types in ONE engine call: the cost marginals do not
        # depend on the job type, so the engine computes them once
        all_segs, opportunities = [], {}
        for wfh in WFH_TYPES:
            margin = time_margins.get((margin_mode, wfh))
            if margin is None:
                raise KeyError(f"No time margin for ({margin_mode}, {wfh}); "
                               f"have {sorted(time_margins)}.")
            segs = build_segments(
                time_margin_for(spec, margin, cutoff_share),
                envelope=envelope if gated else None,
                money_cost_id=cost_id if gated else None,
                copula=copula if gated else INDEPENDENCE,
                pool_by="income_class", only=sn,
                cost_curve=(cost_curve_factory(
                    spec, margin, mode_vot, cost_mean_eur,
                    cost_cutoff=cost_cutoff_eur, cutoff_share=cutoff_share)
                            if gated else None))
            all_segs += [replace(s, name=f"{wfh}|{s.name}",
                                 pool=f"{wfh}|{s.pool}") for s in segs]
            opportunities.update({f"{wfh}|{c}": v
                                  for c, v in pools[wfh].items()})
        per = runner.run_hansen(None, all_segs, cost_matrices=cost_matrices,
                                opportunities=opportunities)
        for wfh in WFH_TYPES:
            for n in sn:
                total[n] += per[f"{wfh}|{n}"].astype(np.float64)
        logger.info("mode %s: %d segments, %d composed filters",
                    margin_mode, len(all_segs),
                    len({s.weight_key for s in all_segs}))
        return total

    def _cost_minutes(cost, rail_share, margin_mode):
        """A cost in EUR as minutes at the value of time (M0); public
        transport weights the value of time by the rail share of km."""
        v = vot or {}
        vr = v.get(margin_mode)
        if vr is None or vr <= 0:
            raise ValueError("M0 needs a positive value of time for "
                             f"'{margin_mode}' (EUR/hour).")
        c = np.asarray(cost, dtype=np.float32)
        if rail_share is not None and f"{margin_mode}_other" in v:
            c = c * vot_factor(rail_share, vr, v[f"{margin_mode}_other"])
        return c * np.float32(60.0 / vr)

    def _vot_factors(options, margin_mode):
        """(K, o, d) factors putting each option's cost in units of the rail
        value of time (M1): by the option's rail share of kilometres; options
        without a share keep their cost."""
        v = vot or {}
        if f"{margin_mode}_other" not in v or margin_mode not in v:
            return np.ones((len(options), n_o, n_d), dtype=np.float32)
        return np.stack([
            vot_factor(o.rail_share, v[margin_mode], v[f"{margin_mode}_other"])
            if o.rail_share is not None
            else np.ones((n_o, n_d), dtype=np.float32) for o in options])

    def scale_groups(has_lime, has_fare=False):
        """((lime scale, fare scale), segment names): segments with equal
        price scales share cost matrices."""
        use_l = bool(has_lime and price_scale)
        use_f = bool(has_fare and fare_scale)
        if not (use_l or use_f):
            return [((1.0, 1.0), list(names))]
        groups: dict = {}
        for n in names:
            key = (float(price_scale.get(n, 1.0)) if use_l else 1.0,
                   float(fare_scale.get(n, 1.0)) if use_f else 1.0)
            groups.setdefault(key, []).append(n)
        return sorted(groups.items())

    def _joint_function():
        """Joint survival of the time block and the money threshold: the
        product (independent thresholds) or the copula of M3."""
        if copula.family == "independence":
            return lambda u, v: u * v
        return lambda u, v: apply_copula(u, v, copula.family, copula.theta)

    def check(mm, label):
        if mm.time.shape != (n_o, n_d):
            raise ValueError(f"Mode '{label}' matrix {mm.time.shape} does "
                             f"not match ({n_o}, {n_d}) origins x "
                             f"destinations.")

    def option_total(margin_mode, options):
        """Union of alternative journeys for one person (staircase of
        joint survival, see `union_terms`)."""
        for o in options:
            check(o, margin_mode)
        priced = envelope is not None and any(o.cost is not None
                                              for o in options)
        ts = np.stack([_finite(o.time, unreachable_minutes) for o in options])
        cs = np.stack([_finite(o.cost if o.cost is not None
                               else np.zeros_like(o.time),
                               unreachable_minutes) for o in options])
        cid = next((o.cost_id for o in options if o.cost is not None), None)
        ls = np.stack([_finite(o.lime if o.lime is not None
                               else np.zeros_like(o.time), 0.0)
                       for o in options])
        fa = np.stack([_finite(o.fare if o.fare is not None
                               else np.zeros_like(o.time), 0.0)
                       for o in options])
        has_l = any(o.lime is not None for o in options)
        has_f = any(o.fare is not None for o in options)
        if spec == "m0":
            # cumulative opportunities in generalised time: the alternative
            # with the least generalised time decides
            total = {n: np.zeros(n_o) for n in names}
            for (sl, sf), group in scale_groups(has_l, has_f):
                g = np.min(np.stack([
                    ts[k] + (_cost_minutes(cs[k] + (sl - 1.0) * ls[k]
                                           + (sf - 1.0) * fa[k],
                                           options[k].rail_share, margin_mode)
                             if priced else 0.0)
                    for k in range(len(options))]), axis=0)
                part = hansen_total(margin_mode, g, None, None, seg_names=group)
                for n in group:
                    total[n] += part[n]
            return total, priced
        if spec == "m1":                 # cost in units of the rail value of time
            f = _vot_factors(options, margin_mode)
            cs, ls, fa = cs * f, ls * f, fa * f
        total = {n: np.zeros(n_o) for n in names}
        for (sl, sf), group in scale_groups(has_l, has_f):
            for t, c, sign in union_terms(
                    ts, cs + (sl - 1.0) * ls + (sf - 1.0) * fa):
                part = hansen_total(margin_mode, t, c if priced else None,
                                    cid, seg_names=group)
                for n in group:
                    total[n] += sign * part[n]
        return total, priced

    def legwise_total(oset):
        """Leg-wise gates, union by inclusion-exclusion (independent
        thresholds, M2).

        Subsets of the options are visited depth first, so the per-leg
        maximum time and the maximum cost of a subset are one elementwise
        step from its parent. The time weight of a leg only depends on the
        options of the subset that have that leg, so it is cached by that
        key; segments are evaluated in parallel threads (numpy releases the
        GIL on these array operations)."""
        opts = oset.options
        n_leg, n_opt = len(oset.margin_modes), len(opts)
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
        lime = np.stack([_finite(o.lime if o.lime is not None
                                 else np.zeros((n_o, n_d)), 0.0)
                         for o in opts])
        fare = np.stack([_finite(o.fare if o.fare is not None
                                 else np.zeros((n_o, n_d)), 0.0)
                         for o in opts])
        pt_leg = oset.margin_modes[-1]
        if spec == "m1":                 # cost in units of the rail value of time
            f = _vot_factors(opts, pt_leg)
            costs, lime, fare = costs * f, lime * f, fare * f
        groups = [(group, costs + (sl - 1.0) * lime + (sf - 1.0) * fare)
                  for (sl, sf), group in scale_groups(
                      any(o.lime is not None for o in opts),
                      any(o.fare is not None for o in opts))]
        # options that have a time on each leg (a leg that no option of a
        # subset uses has S_T(0) = 1 and is skipped)
        active = [tuple(k for k in range(n_opt) if (times[k, leg] > 0).any())
                  for leg in range(n_leg)]
        leg_cache: dict = {}
        pool32 = {w: {p: np.asarray(v, dtype=np.float32)
                      for p, v in pools[w].items()} for w in WFH_TYPES}
        quiet = logging.getLogger("ikob2.engine.runner")
        level, quiet.level = quiet.level, logging.WARNING
        total = {n: np.zeros(n_o) for n in names}
        try:
            # cost margins per job type (the exponential specifications tie
            # the mean acceptable cost to the mean acceptable time of the PT
            # margin, which differs by job type)
            by_name = {}
            for wfh in WFH_TYPES:
                m_pt = time_margins[(pt_leg, wfh)]
                by_name[wfh] = {sg.name: sg for sg in build_segments(
                    time_margin_for(spec, m_pt, cutoff_share),
                    envelope=envelope if priced else None,
                    money_cost_id="c" if priced else None,
                    pool_by="income_class", only=names,
                    cost_curve=(cost_curve_factory(
                        spec, m_pt, (vot or {}).get(pt_leg), cost_mean_eur,
                        cost_cutoff=cost_cutoff_eur, cutoff_share=cutoff_share)
                        if priced and spec != "m0" else None))}

            def leg_weight(leg, wfh, key, t_leg):
                cache_it = len(active[leg]) < n_opt
                ck = (leg, wfh, key)
                if cache_it and ck in leg_cache:
                    return leg_cache[ck]
                f = evaluate_marginal(t_leg, time_margin_for(
                    spec, time_margins[(oset.margin_modes[leg], wfh)],
                    cutoff_share))
                if cache_it:
                    leg_cache[ck] = f
                return f

            def time_weights(chosen, tmax):
                tw = {}
                for wfh in WFH_TYPES:
                    w = None
                    for leg in range(n_leg):
                        if tmax[leg] is None:
                            continue
                        key = tuple(k for k in chosen if k in active[leg])
                        f = leg_weight(leg, wfh, key, tmax[leg])
                        w = f if w is None else w * f
                    tw[wfh] = w
                return tw

            joint = _joint_function()

            def segment_term(name, c, tw, sign):
                acc = 0.0
                sm_cache = {}
                for wfh in WFH_TYPES:
                    sg = by_name[wfh][name]
                    curve = sg.class_filter.cost
                    if priced and curve not in sm_cache:
                        sm_cache[curve] = evaluate_marginal(c, curve)
                    wgt = tw[wfh] if not priced else joint(tw[wfh],
                                                           sm_cache[curve])
                    acc = acc + wgt @ pool32[wfh][sg.pool]
                total[name] += sign * np.asarray(acc, dtype=np.float64)

            def visit(start, chosen, tmax, cmax, ex):
                for k in range(start, n_opt):
                    ch = chosen + (k,)
                    tm = [tmax[leg] if k not in active[leg] else
                          (times[k, leg] if tmax[leg] is None
                           else np.maximum(tmax[leg], times[k, leg]))
                          for leg in range(n_leg)]
                    cm = [g_costs[k] if c_ is None else np.maximum(c_, g_costs[k])
                          for (_, g_costs), c_ in zip(groups, cmax)]
                    sign = 1.0 if len(ch) % 2 else -1.0
                    tw = time_weights(ch, tm)
                    futures = [ex.submit(segment_term, n, c, tw, sign)
                               for (group, _), c in zip(groups, cm)
                               for n in group]
                    for fut in futures:
                        fut.result()
                    visit(k + 1, ch, tm, cm, ex)

            with _single_threaded_blas(), \
                    ThreadPoolExecutor(max_workers=n_threads) as ex:
                visit(0, (), [None] * n_leg, [None] * len(groups), ex)
        finally:
            quiet.level = level
        return total, priced

    def set_total(oset):
        if isinstance(oset, LegOptionSet) and spec == "m0":
            # generalised time is one total: legs add up
            return option_total(oset.margin_modes[-1], tuple(
                ModeMatrices(sum(np.asarray(t, dtype=np.float32)
                                 for t in o.times), o.cost, o.cost_id,
                             lime=o.lime, lime_rentals=o.lime_rentals,
                             rail_share=o.rail_share, fare=o.fare)
                for o in oset.options))
        if isinstance(oset, LegOptionSet):
            return legwise_total(oset)
        return option_total(oset.margin_mode, oset.options)

    reported = reported_atoms(spec, envelope, cutoff_share)
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
            if (price_scale or fare_scale) and (mm.lime is not None
                                                or mm.fare is not None):
                # a single option with scalable parts: the option-set path
                total, priced = option_total(mode, (mm,))
            else:
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
        rows.append(_long(mode, origins, names, total, pop, reported,
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
