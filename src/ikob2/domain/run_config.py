from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class SensitivitySpec:
    parameter: str                     # e.g. "decay_beta"
    values: tuple[float, ...]          # one-at-a-time sweep


@dataclass(frozen=True)
class MonteCarloSpec:
    parameter: str
    distribution: Literal["normal", "uniform", "lognormal"]
    params: tuple[float, ...]          # (mean, std) or (low, high)
    n_draws: int
    seed: int = 42
    # Draws outside [lower, upper] are redrawn (truncated sampling).
    # Essential for e.g. decay_beta, where a negative normal draw would
    # silently produce exploding weights instead of an error.
    bounds: tuple[float, float] | None = None


@dataclass(frozen=True)
class RunConfig:
    name: str
    decay_type: str
    decay_beta: float
    sensitivity: tuple[SensitivitySpec, ...] = ()
    monte_carlo: tuple[MonteCarloSpec, ...] = ()