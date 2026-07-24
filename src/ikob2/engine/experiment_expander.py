import numpy as np
from dataclasses import dataclass, field

from ikob2.domain.run_config import RunConfig
from ikob2.variants.base import MultiplyGeneralizedCost, SetDecayBeta, Variant


@dataclass(frozen=True)
class RunSpec:
    run_id: str
    variants: tuple[Variant, ...]
    tags: dict = field(default_factory=dict)
    # {"kind": "baseline" | "sensitivity" | "monte_carlo", ...}


VARIANT_FACTORIES = {
    "decay_beta": SetDecayBeta,
    "cost_factor": MultiplyGeneralizedCost,
}


def expand(config: RunConfig) -> list[RunSpec]:
    runs = [RunSpec("baseline", (), {"kind": "baseline"})]

    for spec in config.sensitivity:
        factory = VARIANT_FACTORIES[spec.parameter]
        for value in spec.values:
            runs.append(RunSpec(
                run_id=f"sens_{spec.parameter}_{value}",
                variants=(factory(value),),
                tags={"kind": "sensitivity",
                      "parameter": spec.parameter, "value": value},
            ))

    for spec in config.monte_carlo:
        rng = np.random.default_rng(spec.seed)
        draws = _sample(rng, spec)
        factory = VARIANT_FACTORIES[spec.parameter]
        for i, value in enumerate(draws):
            runs.append(RunSpec(
                run_id=f"mc_{spec.parameter}_{i:04d}",
                variants=(factory(float(value)),),
                tags={"kind": "monte_carlo",
                      "parameter": spec.parameter, "draw": i},
            ))

    return runs


def _sample(rng, spec):
    draws = _raw_sample(rng, spec, spec.n_draws)
    if spec.bounds is None:
        return draws
    lo, hi = spec.bounds
    # Truncated sampling: redraw out-of-bounds values rather than
    # clipping (clipping piles probability mass onto the bounds).
    for _ in range(100):
        bad = (draws < lo) | (draws > hi)
        if not bad.any():
            return draws
        draws[bad] = _raw_sample(rng, spec, int(bad.sum()))
    raise ValueError(
        f"Could not draw {spec.n_draws} values inside bounds {spec.bounds} "
        f"for {spec.parameter} ~ {spec.distribution}{spec.params}; "
        "the distribution barely overlaps the bounds."
    )


def _raw_sample(rng, spec, size):
    if spec.distribution == "normal":
        return rng.normal(*spec.params, size=size)
    if spec.distribution == "uniform":
        return rng.uniform(*spec.params, size=size)
    if spec.distribution == "lognormal":
        return rng.lognormal(*spec.params, size=size)
    raise ValueError(spec.distribution)