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
| `logistic` | `alpha`, `omega` | legacy IKOB sigmoid | - | not derived from a threshold distribution |

Every curve also takes an optional `"atom"` (share for whom no positive
impedance is acceptable).

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
