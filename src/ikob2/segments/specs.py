"""
The impedance specifications of the paper (Table 4) and the ladder of
benchmarks around them.

    M0   1{t + c/VoT <= T*}          cumulative opportunities in generalised
                                     time: the practitioners' isochrone; T* is
                                     the median acceptable time (Santana
                                     Palacios and El-Geneidy, 2022)
    M0u  1{t <= T*} 1{c <= C*}       dual cut-off, one cost cut-off for
                                     everyone: C* the median of the
                                     population's cost thresholds (Herszenhut
                                     et al., 2022; Conway and Stewart, 2019)
    M0s  1{t <= T*} 1{c <= C*_s}     dual cut-off per segment: C*_s the median
                                     of the segment's cost margin (zero for a
                                     segment with atom 1): the degenerate
                                     corner of M2
    M1   exp{-(t + c/VoT)/beta}      generalised cost, one VoT per mode,
                                     beta = mean acceptable time
    M1c  the same with ONE value of time calibrated so that the implied mean
         acceptable cost equals the (population-weighted) median of the
         segment means of the envelope: a well-calibrated fixed VoT
    M1'  the same with a VoT per segment such that the implied mean
         acceptable cost beta*VoT_s equals the segment's mean envelope
    M2   S_T(t) S_M(c)               Weibull time margin, uniform cost
                                     margin with an atom, independent gates
    M3   Gumbel-Hougaard copula of the M2 margins, theta imposed

The cut-offs of M0, M0u and M0s are quantiles of the margins at
`accessibility.cutoff_share` (0.5: the median). T* is the same for everyone;
only the cost cut-off of M0s varies by segment.

M1 and M1' are products of two exponential survival functions, exp(-t/beta)
exp(-c/(beta VoT)): a generalised-cost decay written as gates. M1' and M2
share the first moment of both margins, so the M1' -> M2 contrast is the
shape of the thresholds and M2 -> M3 is dependence. M1 and M1' have no atom
(f(0,0) = 1), so their accessibility needs no normalisation.
"""

from __future__ import annotations

import numpy as np

from ikob2.core.families import mean_threshold
from ikob2.domain.filter_config import INDEPENDENCE, CopulaSpec, CurveSpec
from ikob2.params import DEFAULTS

SPECS = ("m0", "m0u", "m0s", "m1", "m1c", "m1p", "m2", "m3")
EXPONENTIAL_SPECS = ("m1", "m1c", "m1p")
CUTOFF_SPECS = ("m0", "m0u", "m0s")
DUAL_CUTOFF_SPECS = ("m0u", "m0s")

# Values of time (EUR per hour) of the M1 benchmark, from the Dutch national
# value-of-time study (LMS/NRM): car driver, train, bus/tram/metro. Public
# transport is priced at `pt` (rail) and `pt_other` per journey in proportion
# to the rail share of its kilometres. Bicycle (10.50-11.00) and walking
# (12.50-13.00) have no cost and need no VoT.
DEFAULT_VOT = DEFAULTS.vot.to_dict()

# Rate of an exponential with (effectively) zero mean: acceptable only at c = 0.
_ZERO_MEAN_RATE = 1e6


def spec_copula(spec: str, theta: float | None) -> CopulaSpec:
    """The copula of a specification: dependence only in M3 (theta = inf is
    the comonotone limit, theta = 1 is independence)."""
    if spec != "m3":
        return INDEPENDENCE
    if theta is None:
        raise ValueError("M3 needs theta.")
    if np.isinf(theta):
        return CopulaSpec("comonotone")
    return CopulaSpec("gumbel", float(theta))


def mean_time(time_margin: CurveSpec) -> float:
    """Mean acceptable travel time of a time margin (minutes)."""
    return mean_threshold(time_margin.curve, time_margin.params)


def exponential_time(time_margin: CurveSpec) -> CurveSpec:
    """Exponential margin with the same mean acceptable time."""
    return CurveSpec("exponential", (1.0 / mean_time(time_margin),))


def mean_cost(row) -> float:
    """Mean acceptable cost of a segment's envelope: uniform on [low, high]
    with an atom at zero (mass `atom` contributes zero)."""
    atom = float(row.atom)
    if atom >= 1.0:
        return 0.0
    return (1.0 - atom) * 0.5 * (float(row.low) + float(row.high))


def exponential_cost(mean: float) -> CurveSpec:
    """Exponential cost margin with the given mean acceptable cost (EUR); a
    mean of 0 or less gives a margin that accepts only free trips."""
    rate = _ZERO_MEAN_RATE if mean <= 0 else 1.0 / mean
    return CurveSpec("exponential", (rate,))


def median_time(time_margin: CurveSpec) -> float:
    """Median acceptable time of a time margin (minutes): S_T(t) = 1/2."""
    return cutoff_time(time_margin, 0.5)


def cutoff_time(time_margin: CurveSpec, share: float) -> float:
    """Acceptable time that `share` of the thresholds reach: S_T(t) = share
    (minutes)."""
    from scipy import optimize

    from ikob2.core.families import get_family

    if not 0.0 < share < 1.0:
        raise ValueError(f"The cut-off share must lie in (0, 1), got {share}.")
    if time_margin.curve == "step":
        return float(time_margin.params[0])
    fam = get_family(time_margin.curve)
    params = [float(p) for p in time_margin.params]
    target = np.log(share)
    f = lambda t: float(fam.log_survival(np.array(t), *params)) - target  # noqa: E731
    hi = 1.0
    while f(hi) > 0 and hi < 1e7:
        hi *= 2.0
    return float(optimize.brentq(f, 1e-9, hi, xtol=1e-9))


def _cost_survival(row, c: np.ndarray) -> np.ndarray:
    """S_M(c) of a segment's envelope row for c > 0: uniform on [low, high]
    times 1 - atom (a segment with atom 1 accepts no positive cost)."""
    from ikob2.core.decay_curves import uniform

    atom = float(row.atom)
    if atom >= 1.0:
        return np.zeros_like(c)
    return (1.0 - atom) * uniform(c, float(row.low), float(row.high)
                                  ).astype(np.float64)


def cutoff_cost(rows, share: float, weights=None) -> float:
    """Cost cut-off (EUR per trip) of the dual cut-off specifications: the
    largest c with S(c) >= share, S the (`weights`-weighted) mixture of the
    rows' cost margins. One row gives the segment's own cut-off (M0s); all
    segments weighted by persons give the population cut-off (M0u). Zero
    when fewer than `share` accept any positive cost (a segment with atom
    1): only free trips pass.

    The mixture is piecewise linear between the rows' low and high values
    (left-continuous, with jumps where low = high), so the cut-off is found
    exactly, interval by interval."""
    if not 0.0 < share < 1.0:
        raise ValueError(f"The cut-off share must lie in (0, 1), got {share}.")
    rows = list(rows)
    w = np.ones(len(rows)) if weights is None else np.asarray(weights, float)
    if len(w) != len(rows) or (w < 0).any() or w.sum() <= 0:
        raise ValueError("Need one non-negative weight per row, not all zero.")
    w = w / w.sum()

    def surv(c: float) -> float:
        z = np.array([c])
        return float(sum(wi * _cost_survival(r, z)[0]
                         for wi, r in zip(w, rows) if wi > 0))

    knots = sorted({0.0} | {float(v) for r in rows if float(r.atom) < 1.0
                            for v in (r.low, r.high)})
    for a, b in zip(knots[:-1], knots[1:]):
        s_b = surv(b)
        if s_b >= share:
            continue
        s_a = 2.0 * surv(0.5 * (a + b)) - s_b       # the limit at a+ (linear)
        if s_a < share:
            return a
        return a + (s_a - share) / (s_a - s_b) * (b - a)
    return knots[-1]


def median_segment_cost(envelope, weights=None) -> float:
    """Population-weighted median of the mean acceptable cost of the
    segments (EUR per trip): the calibration target of M1c. `weights` maps
    the segment name '<type>_<class>' to persons (default: equal)."""
    means, w = [], []
    for row in envelope.itertuples():
        means.append(mean_cost(row))
        w.append(1.0 if weights is None else float(
            weights.get(f"{row.household_type}_{row.income_class}", 0.0)))
    means, w = np.array(means), np.array(w)
    order = np.argsort(means, kind="stable")
    cum = np.cumsum(w[order])
    if cum[-1] <= 0:
        raise ValueError("No population to weight the median cost.")
    return float(means[order][np.searchsorted(cum, 0.5 * cum[-1])])


def implied_vot(cost_mean_eur: float, time_margin: CurveSpec) -> float:
    """Value of time (EUR per hour) implied by a mean acceptable cost and
    the mean acceptable time of a margin."""
    return cost_mean_eur * 60.0 / mean_time(time_margin)


def cost_curve_factory(spec: str, time_margin: CurveSpec,
                       vot_per_hour: float | None,
                       cost_mean: float | None = None, *,
                       cost_cutoff: float | None = None,
                       cutoff_share: float = DEFAULTS.accessibility.cutoff_share):
    """Function envelope row -> cost CurveSpec, or None to keep the uniform
    margin of the envelope (M2, M3) or when there is no cost margin (M0).
    M0u needs the population cost cut-off `cost_cutoff` (EUR per trip)."""
    if spec in ("m0", "m2", "m3"):
        return None
    if spec == "m0u":
        if cost_cutoff is None or cost_cutoff < 0:
            raise ValueError("M0u needs the population cost cut-off.")
        shared_step = CurveSpec("step", (float(cost_cutoff),))
        return lambda row: shared_step
    if spec == "m0s":
        return lambda row: CurveSpec("step", (cutoff_cost([row], cutoff_share),))
    if spec == "m1c":
        if cost_mean is None or cost_mean <= 0:
            raise ValueError("M1c needs the calibrated mean acceptable cost.")
        shared_c = exponential_cost(cost_mean)
        return lambda row: shared_c
    if spec == "m1p":
        return lambda row: exponential_cost(mean_cost(row))
    if spec == "m1":
        if vot_per_hour is None or vot_per_hour <= 0:
            raise ValueError("M1 needs a positive value of time (EUR/hour).")
        mean = mean_time(time_margin) * vot_per_hour / 60.0
        shared = exponential_cost(mean)
        return lambda row: shared
    raise ValueError(f"Unknown specification {spec!r}; use {SPECS}.")


def time_margin_for(spec: str, time_margin: CurveSpec,
                    cutoff_share: float = DEFAULTS.accessibility.cutoff_share
                    ) -> CurveSpec:
    """The exponential specifications use an exponential time margin with the
    mean of the margin; M0, M0u and M0s a step at its `cutoff_share`
    quantile (the median); M2 and M3 the margin."""
    if spec in CUTOFF_SPECS:
        return CurveSpec("step", (cutoff_time(time_margin, cutoff_share),))
    return exponential_time(time_margin) if spec in EXPONENTIAL_SPECS \
        else time_margin


def vot_factor(rail_share: np.ndarray, vot_rail: float,
               vot_other: float) -> np.ndarray:
    """vot_rail / VoT_ij with VoT_ij = share * vot_rail + (1 - share) *
    vot_other (unknown shares count as rail): the factor that puts a cost
    in units of the rail value of time."""
    share = np.where(np.isfinite(rail_share), rail_share, 1.0)
    return (vot_rail / (share * vot_rail + (1.0 - share) * vot_other)
            ).astype(np.float32)


def vot_weighted_cost(cost: np.ndarray, rail_share: np.ndarray,
                      vot_rail: float, vot_other: float) -> np.ndarray:
    """Cost matrix in units of the rail value of time: c * vot_rail / VoT_ij
    with VoT_ij = share * vot_rail + (1 - share) * vot_other. Under M1 the
    cost gate exp(-c/(beta VoT_ij)) then has the single rate 1/(beta
    vot_rail) on this matrix (equal to the generalised-cost form
    exp(-(t + c/VoT_ij)/beta)). Unknown shares use the rail value."""
    share = np.where(np.isfinite(rail_share), rail_share, 1.0)
    vot = share * vot_rail + (1.0 - share) * vot_other
    return (np.asarray(cost, dtype=np.float64) * (vot_rail / vot)
            ).astype(np.float32)


def atom_reported(spec: str) -> bool:
    """M0s, M2 and M3 report the atom; M0u, M1 and M1' have f(0,0) = 1."""
    return spec in ("m0s", "m2", "m3")


def reported_atoms(spec: str, envelope, cutoff_share: float):
    """The envelope with the atom each specification reports: under M0s a
    segment has atom 1 when its cost cut-off is zero (no positive cost
    passes) and 0 otherwise; the other specifications keep the envelope's
    atom."""
    if spec != "m0s" or envelope is None:
        return envelope
    env = envelope.copy()
    env["atom"] = [1.0 if cutoff_cost([r], cutoff_share) == 0.0 else 0.0
                   for r in envelope.itertuples()]
    return env
