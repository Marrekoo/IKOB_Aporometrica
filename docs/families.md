# Threshold shapes (`core.families`)

A decay function normalised to 1 at zero impedance is the survival
function `f(z) = Pr(X >= z)` of a maximum acceptable impedance `X`. The
shape of `f` is a claim about the hazard `h(z) = -d log f / dz`: how
fast acceptance is lost per extra unit. All families below are proper
survival functions (f(0) = 1, non-increasing, tending to 0), so any of
them can be a time or cost marginal in the gate `f(t, c) = S_T(t) S_M(c)`.

| curve (`"curve"` in the filter config) | parameters | f(z) | hazard | notes |
|---|---|---|---|---|
| `exponential` | `beta` (rate) | exp(-beta z) | constant | the gravity/GC corner |
| `weibull` | `shape`, `scale` | exp(-(z/scale)^shape) | increasing (shape > 1), constant (= 1), decreasing (< 1) | soft threshold |
| `lomax` | `alpha`, `scale` | (1 + z/scale)^-alpha | decreasing | heterogeneity-mixed exponential; power-law tail |
| `pareto` | `alpha`, `z0` | 1 up to z0, then (z/z0)^-alpha | alpha/z beyond z0 | the survival form of the power law: constant elasticity alpha |
| `tanner` | `rho`, `chi`, `scale` | (1 + z/scale)^-rho exp(-chi z) | rho/(scale+z) + chi | gamma friction factor a z^-rho e^-chi z, shifted so f(0) = 1; rho = 0 is exponential, chi = 0 is Lomax |
| `gamma` | `shape`, `scale` | gamma-distributed threshold | rises to 1/scale (shape > 1) | |
| `lognormal` | `mu`, `sigma` | ln X ~ N(mu, sigma^2) | unimodal | |
| `loglogistic` | `shape`, `scale` | 1 / (1 + (z/scale)^shape) | unimodal (shape > 1) | |
| `step` | `threshold` | 1 up to threshold | - | the isochrone (degenerate threshold) |
| `uniform` | `low`, `high` | piecewise linear | - | cost margin from an envelope |
| `piecewise_linear` | `knots` `[[z, f], ...]` | linear interpolation between knots; 0 beyond the last | piecewise constant | e.g. stated-tolerance survival at survey bin edges; `uniform` is the knots (0,1), (low,1), (high,0) |
| `piecewise_quadratic` | `knots` `[[z, f], ...]` | shape-preserving C1 quadratic spline through the knots; 0 beyond the last | continuous, piecewise linear | the smooth counterpart: same knots, continuous hazard, always monotone |
| `triangular` | `low`, `mode`, `high` | survival of a triangular density on [low, high] with its peak at `mode` (F = (z-low)^2 / ((high-low)(mode-low)) up to the mode, 1 - (high-z)^2 / ((high-low)(high-mode)) after) | continuous | mean (low+mode+high)/3; mode = low is a right-angled density falling to `high`, mode = high one rising to it |
| `quadratic_ramp` | `low`, `high` | 1 up to low, then a smooth two-piece quadratic ramp (f = 1 - 2u^2, then 2(1-u)^2, u = (z-low)/(high-low)) to 0 at high | continuous | the classic quadratic kernel cut-off; mean = (low+high)/2 |
| `logistic` | `alpha`, `omega` | legacy IKOB sigmoid | - | not derived from a threshold distribution |

**Knot curves.** `piecewise_linear` and `piecewise_quadratic` are given
by a survival function at knots, for example the stated-tolerance
survey read at its 15-minute bin edges:
`{"curve": "piecewise_quadratic", "knots": [[0, 1], [15, 0.92], [30, 0.5],
[45, 0.15], [60, 0.02], [90, 0]]}`. The first knot must be (0, 1),
impedances strictly increasing, survival values in [0, 1] and
non-increasing; beyond the last knot the survival is 0 (a last value
above 0 is a jump: a share of the population whose threshold sits exactly
there). The quadratic spline interpolates the knots exactly, is
continuously differentiable, and never overshoots (slopes are limited and
each knot interval gets one extra breakpoint), so it can be used where a
linear interpolation's kinked hazard would matter; on smooth data it is
closer to the truth than linear interpolation.

Every curve also takes an optional `"atom"` (share for whom no positive
impedance is acceptable).

**Which shape is used where.** The cost envelopes use `uniform`: the
survival function of a threshold with a uniform density on
[low, high]. The time margin of the non-exponential specifications is
the Weibull (paper, Section 3.2); the exponential is the benchmark.
`triangular` is the survival function of a triangular density on
[low, high] with a free mode; `quadratic_ramp` is its symmetric case (mode
at the midpoint): f = 1 - 2u^2 up to the midpoint and 2(1-u)^2 after it.

**Two things that differ from the usual friction-function names.** The
plain power law `z^-beta` and the raw gamma friction factor `z^-rho
e^-chi z` diverge at zero impedance, so they are not survival functions;
`pareto` and `tanner` are their normalised forms. The old `power` curve
(`c^-beta`) stays for the legacy path only; it exceeds 1 below 1 and
`compose_filters` rejects it as a marginal. The `gamma` curve is the
survival function of a gamma-DISTRIBUTED threshold, a different object
from the gamma friction factor (`tanner`).

## Diagnostics from "The Fixed-VOT Trap in Generalised Cost Models"

`core.families` also provides the hazard tools used there:

* `survival`, `cumulative_hazard` (Lambda = -log f), `hazard`,
  `elasticity` (eta = -z h(z)) for every family;
* `hazard_shape` (constant / increasing / decreasing / unimodal);
* `implied_vot(time_family, cost_family, t, c)` = h_T(t) / h_M(c), the
  local value of time of a separable surface. It is constant only for
  exponential margins (h_T / h_M) and equals (delta/gamma)(c/t) for
  power-law margins; it varies for every other shape. That is the paper's
  point: a fixed-VOT generalised cost is consistent with separable
  margins only under exponential decay;
* `ttt_transform`: the scaled total-time-on-test transform. Exponential
  is the diagonal, increasing hazard (Weibull shape > 1) concave,
  decreasing hazard (Lomax) convex. It needs a finite mean;
* `mean_threshold` (E[X], infinite for heavy tails) and
  `moment_matched_scale`, which rescales a family to a common mean so
  shapes can be compared on dispersion alone (the paper's Figure 2:
  common mean 30 minutes, Weibull shape 2.5, Lomax alpha 2.2).

Example:

    from ikob2.core import families as f
    p = f.moment_matched_scale("weibull", (2.5, 1.0), 30.0)
    f.survival("weibull", p, [10, 30, 60])
    f.implied_vot(("weibull", p), ("exponential", (0.5,)), 30.0, 6.0)

## Time margins from the survey fits (`segments.time_margins`)

The Weibull time margins of the non-exponential specifications are in
`data/margins/S_T_work.csv` (from the R fit of the professionals'
stated maximum acceptable commuting times): scale `eta` (minutes) and
shape `k` per mode (bike, public transport, car) and per job type
(no home working / home working possible). All shapes are between 2.7
and 3.2 (increasing hazard, class IFR); home-working jobs accept longer
trips in every mode (for cars a median of 48.6 against 39.9 minutes).
`load_time_margins(path)` returns `{(mode, wfh): CurveSpec("weibull",
(k, eta))}` and checks the stored median and class against the
parameters. Modes use the skim names `bike`, `pt`, `car`.

The home-working split belongs to the JOB, not the traveller, so which
curve applies depends on the destination. Until jobs are divided into
"admits home working" and "does not", a run uses one of the two curves for
all jobs.
