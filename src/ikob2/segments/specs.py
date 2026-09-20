"""
The four impedance specifications of the paper (Table 4).

    M1   exp{-(t + c/VoT)/beta}      generalised cost, one VoT per mode,
                                     beta = mean acceptable time
    M1'  the same with a VoT per segment such that the implied mean
         acceptable cost beta*VoT_s equals the segment's mean envelope
    M2   S_T(t) S_M(c)               Weibull time margin, uniform cost
                                     margin with an atom, independent gates
    M3   Gumbel-Hougaard copula of the M2 margins, theta imposed

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

SPECS = ("m1", "m1p", "m2", "m3")

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
    rate = _ZERO_MEAN_RATE if mean <= 0 else 1.0 / mean
    return CurveSpec("exponential", (rate,))


def cost_curve_factory(spec: str, time_margin: CurveSpec,
                       vot_per_hour: float | None):
    """Function envelope row -> cost CurveSpec, or None to keep the uniform
    margin of the envelope (M2, M3)."""
    if spec in ("m2", "m3"):
        return None
    if spec == "m1p":
        return lambda row: exponential_cost(mean_cost(row))
    if spec == "m1":
        if vot_per_hour is None or vot_per_hour <= 0:
            raise ValueError("M1 needs a positive value of time (EUR/hour).")
        mean = mean_time(time_margin) * vot_per_hour / 60.0
        shared = exponential_cost(mean)
        return lambda row: shared
    raise ValueError(f"Unknown specification {spec!r}; use {SPECS}.")


def time_margin_for(spec: str, time_margin: CurveSpec) -> CurveSpec:
    """M1 and M1' use an exponential time margin with the Weibull's mean."""
    return exponential_time(time_margin) if spec in ("m1", "m1p") \
        else time_margin


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
    """M2 and M3 report the atom; M1 and M1' have f(0,0) = 1."""
    return spec in ("m2", "m3")
