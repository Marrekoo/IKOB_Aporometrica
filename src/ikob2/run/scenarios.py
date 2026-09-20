"""Revenue and rentals of the shared-bicycle operator, and the flat price of
scenario S4.

S4 replaces the Lime tiers by one flat price per rental, revenue neutral. The
model has acceptable opportunities, not trips, so revenue needs a stand-in
for volume and a rule for which option a person uses:

  * the person uses the FASTEST option they find acceptable;
  * volume is the number of acceptable pairs (jobs of the income class,
    population of the segment) for which that option contains a rental.

With independent thresholds (M2) the probability that option k, in time
order, is the fastest acceptable one has a closed form. Options sorted by
time per pair, with the running cheapest cost cmin_(k-1) of the faster ones:

    P_k = S_T(t_k) [ S_M(c_k) - S_M(max(c_k, cmin_(k-1))) ]

(a faster option is rejected on cost iff the money threshold is below its
cost, and thresholds above t_k accept every faster option on time).

Two calibrations of the flat price p, both revenue neutral against the tiers:

  weighted_mean  p = revenue / rentals under the baseline tiers: the average
                 tier price weighted by baseline volume;
  fixed_point    p such that p x rentals(p) = baseline revenue, where the
                 rentals are recomputed at the flat price (people change
                 what they accept). The default.

Rentals are counted per rental: a journey with a dockless access and a Lime
hub egress has two. `rentals_scaled` weights them with the segment's price
scale (concessions), so that the flat price is comparable across segments.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, replace
from typing import Callable, Mapping

import numpy as np
from scipy import optimize

from ikob2.engine.runner import evaluate_marginal
from ikob2.run.accessibility import (WFH_TYPES, MixedMode, OptionSet,
                                     _finite, prepare_inputs)
from ikob2.segments.bridge import build_segments

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Usage:
    """Expected Lime revenue and rentals, in acceptable-pair units (summed
    over origins, weighted by population)."""
    revenue: float
    rentals: float
    rentals_scaled: float
    by_segment: dict = field(default_factory=dict, compare=False)


def choice_terms(times: np.ndarray, costs: np.ndarray):
    """Options (K, o, d) sorted by time per pair: (order, t, c, hi) where
    hi is the running cheapest cost of the faster options, the cost above
    which a faster option is rejected (`inf` for the fastest)."""
    order = np.argsort(times, axis=0, kind="stable")
    t = np.take_along_axis(times, order, axis=0)
    c = np.take_along_axis(costs, order, axis=0)
    cmin = np.minimum.accumulate(c, axis=0)
    hi = np.concatenate([np.full_like(c[:1], np.inf), cmin[:-1]], axis=0)
    return order, t, c, np.maximum(c, hi)


def lime_usage(*, origins, destinations, populations, sector_jobs, wfh_share,
               sector_wage, envelope, time_margins, mode: MixedMode | OptionSet,
               price_scale: Mapping[str, float] | None = None,
               margin_mode: str = "pt", unreachable_minutes: float = 1e4
               ) -> Usage:
    """Lime revenue and rentals of a shared-bicycle mode (M2, independent
    thresholds, options judged on their total time)."""
    origins = [str(o) for o in origins]
    destinations = [str(d) for d in destinations]
    names, pop, pools = prepare_inputs(
        origins, destinations, populations, sector_jobs, wfh_share,
        sector_wage, envelope, None)
    n_o = len(origins)
    segs = {s.name: s for s in build_segments(
        time_margins[(margin_mode, WFH_TYPES[0])], envelope=envelope,
        money_cost_id="c", pool_by="income_class", only=names)}
    parts = mode.parts if isinstance(mode, MixedMode) else ((np.ones(n_o), mode),)
    scale = {n: float((price_scale or {}).get(n, 1.0)) for n in names}
    groups: dict = {}
    for n, s in scale.items():
        groups.setdefault(s, []).append(n)

    rev = {n: 0.0 for n in names}
    ren = {n: 0.0 for n in names}
    for weight, oset in parts:
        w = np.asarray(weight, dtype=float)
        opts = oset.options
        ts = np.stack([_finite(o.time, unreachable_minutes) for o in opts])
        cs = np.stack([_finite(o.cost, unreachable_minutes) for o in opts])
        ls = np.stack([_finite(o.lime if o.lime is not None
                               else np.zeros_like(o.time), 0.0) for o in opts])
        ns = np.array([o.lime_rentals for o in opts], dtype=float)
        if not (ns > 0).any():
            continue
        order = np.argsort(ts, axis=0, kind="stable")
        t_sorted = np.take_along_axis(ts, order, axis=0)
        l_sorted = np.take_along_axis(ls, order, axis=0)
        n_sorted = ns[order]                                   # (K, o, d)
        st = {wfh: [evaluate_marginal(t_sorted[k], time_margins[
            (margin_mode, wfh)]) for k in range(len(opts))]
            for wfh in WFH_TYPES}
        for sc, group in sorted(groups.items()):
            _, _, c_k, hi_k = choice_terms(ts, cs + (sc - 1.0) * ls)
            hi_k = np.where(np.isinf(hi_k), unreachable_minutes, hi_k)
            for k in range(len(opts)):
                if not (n_sorted[k] > 0).any():
                    continue
                price = (l_sorted[k] * sc).astype(np.float32)
                count = n_sorted[k].astype(np.float32)
                for name in group:
                    cost_curve = segs[name].class_filter.cost
                    prob_c = (evaluate_marginal(c_k[k], cost_curve)
                              - evaluate_marginal(hi_k[k], cost_curve))
                    for wfh in WFH_TYPES:
                        m = st[wfh][k] * prob_c
                        pool = np.asarray(pools[wfh][segs[name].pool],
                                          dtype=np.float32)
                        weight_i = w * pop[name].fillna(0.0).to_numpy()
                        rev[name] += float(weight_i @ ((m * price) @ pool))
                        ren[name] += float(weight_i @ ((m * count) @ pool))
    scaled = sum(scale[n] * ren[n] for n in names)
    return Usage(sum(rev.values()), sum(ren.values()), scaled,
                 {n: (rev[n], ren[n]) for n in names})


def calibrate_flat(usage_for: Callable[[float | None], Usage], *,
                   method: str = "fixed_point", tol_eur: float = 0.005,
                   max_evals: int = 30) -> tuple[float, dict]:
    """The revenue-neutral flat price per rental.

    usage_for(None) is the baseline (the tiers); usage_for(p) the usage at a
    flat price p. Returns (price, info)."""
    base = usage_for(None)
    if base.rentals_scaled <= 0:
        raise ValueError("The baseline has no Lime rentals: nothing to "
                         "calibrate.")
    p_mean = base.revenue / base.rentals_scaled
    info = {"method": method, "baseline_revenue": base.revenue,
            "baseline_rentals": base.rentals, "weighted_mean_eur": p_mean,
            "evaluations": 1}
    if method == "weighted_mean":
        return p_mean, info
    if method != "fixed_point":
        raise ValueError(f"method must be weighted_mean or fixed_point, "
                         f"got {method!r}.")
    cache: dict = {}

    def gap(p: float) -> float:
        if p not in cache:
            u = usage_for(p)
            info["evaluations"] += 1
            cache[p] = p * u.rentals_scaled - base.revenue
            logger.info("flat price %.4f: revenue gap %.6g", p, cache[p])
        return cache[p]

    lo = hi = p_mean
    g = gap(p_mean)
    step = 1.15 if g < 0 else 1 / 1.15
    for _ in range(max_evals):
        nxt = hi * step if g < 0 else lo * step
        gn = gap(nxt)
        if (gn >= 0) != (g >= 0):
            lo, hi = sorted((nxt, hi if g < 0 else lo))
            break
        if g < 0:
            hi = nxt
        else:
            lo = nxt
    else:
        raise RuntimeError("No flat price gives the baseline revenue (the "
                           "revenue curve does not reach it).")
    p = optimize.brentq(gap, lo, hi, xtol=tol_eur, maxiter=max_evals)
    info["flat_eur"] = p
    return p, info


def effectiveness(base: "pd.DataFrame", scen: "pd.DataFrame",
                  usage_base: Mapping, usage_scen: Mapping, mode: str = "pt_v2",
                  by: str = "income_class") -> "pd.DataFrame":
    """Who gains from a tariff change, against what it costs the operator or
    the subsidising party.

    base, scen : accessibility tables of two runs (`AccessibilityResult.table`);
    usage_*    : the `scenario.lime_usage.by_segment` of their run.json
                 ({segment: {revenue, rentals}});
    gain       : population-weighted change in acceptable jobs of `mode`;
    cost       : Lime revenue foregone (base minus scenario), in EUR times
                 acceptable pairs, the units of the volume proxy.

    Returns per group of `by`: gain and cost, their shares of the total, and
    the ratio of the shares (above 1: the group gets more of the benefit than
    it costs, a progressive measure; a blanket cut is below 1 for high
    incomes because they use Lime as much but do not need the discount)."""
    import pandas as pd

    key = ["buurtcode", "segment"]
    a = base[base["mode"] == mode].set_index(key)
    b = scen[scen["mode"] == mode].set_index(key)
    w = a["population"]
    gain = ((b["accessibility"] - a["accessibility"]) * w).groupby(
        a[by]).sum()
    seg_cost = pd.Series({n: usage_base[n]["revenue"] - usage_scen[n]["revenue"]
                          for n in usage_base})
    seg_group = a.reset_index().drop_duplicates("segment").set_index(
        "segment")[by]
    cost = seg_cost.groupby(seg_group).sum().reindex(gain.index).fillna(0.0)
    out = pd.DataFrame({"gain": gain, "cost": cost})
    out["gain_share"] = out["gain"] / out["gain"].sum()
    out["cost_share"] = out["cost"] / out["cost"].sum()
    out["share_ratio"] = out["gain_share"] / out["cost_share"].where(
        out["cost_share"].abs() > 1e-12)
    return out
