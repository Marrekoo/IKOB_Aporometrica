"""
domain/filter_config.py — declarative (class, mode) tolerance filters.

Schema (JSON):

{
  "copula": {"family": "frank", "theta": 2.0},        // optional, global
  "modes": {
    "Auto": {
      "copula": {"family": "independence"},           // optional, shadows global
      "classes": {
        "laag": {
          "time": {"curve": "logistic", "alpha": 0.125, "omega": 50},
          "cost": {"curve": "logistic", "alpha": 0.9, "omega": 6.0},
          "scaling": 1.0,                              // optional, default 1.0
          "copula": {"family": "frank", "theta": 0.5}  // optional, shadows mode/global
        },
        "middellaag": { "time": {...} },               // no cost => pure time filter
        ...
      }
    }
  }
}

Resolution: class copula > mode copula > global copula > independence.

Fail-loud policy:
  * every mode must define EXACTLY the canonical income classes;
  * every class must define "time"; "cost" is optional;
  * "scaling" is forbidden inside curve blocks (it is not part of a
    probability marginal — it lives at the class-filter level and is
    applied after composition, see core/compose.py);
  * unknown keys anywhere are errors, not warnings;
  * "frank" and "gumbel" require theta, all other families forbid it;
    "gumbel" (Gumbel-Hougaard) additionally requires theta >= 1
    (theta = 1 is independence; use "comonotone" for the theta -> inf
    limit, JSON has no infinity).

epsilon is deliberately ABSENT from this schema: sparsification
strength keeps its single owner (the CLI) and is applied once, to the
composed matrix.
"""

import json
import pathlib
from dataclasses import dataclass

INCOME_CLASSES = ("laag", "middellaag", "middelhoog", "hoog")

_PARAMETRIC_FAMILIES = {"frank", "gumbel"}
_PARAMETERLESS_FAMILIES = {"independence", "comonotone", "countermonotone"}
_FAMILIES = _PARAMETRIC_FAMILIES | _PARAMETERLESS_FAMILIES

# curve name -> exact required parameter names (order = call order)
_CURVE_ARITY = {
    "logistic": ("alpha", "omega"),
    "exponential": ("beta",),
    "power": ("beta",),
    "weibull": ("shape", "scale"),
    "uniform": ("low", "high"),
    "lomax": ("alpha", "scale"),
    "pareto": ("alpha", "z0"),
    "tanner": ("rho", "chi", "scale"),
    "gamma": ("shape", "scale"),
    "lognormal": ("mu", "sigma"),
    "loglogistic": ("shape", "scale"),
    "step": ("threshold",),
}


class FilterConfigError(ValueError):
    pass


def _reject_unknown_keys(block: dict, allowed: set[str], where: str):
    unknown = set(block) - allowed
    if unknown:
        raise FilterConfigError(
            f"Unknown key(s) {sorted(unknown)} in {where}. "
            f"Allowed: {sorted(allowed)}."
        )


@dataclass(frozen=True)
class CopulaSpec:
    family: str = "independence"
    theta: float | None = None

    def __post_init__(self):
        if self.family not in _FAMILIES:
            raise FilterConfigError(
                f"Unknown copula family '{self.family}'. "
                f"Known: {sorted(_FAMILIES)}."
            )
        if self.family in _PARAMETRIC_FAMILIES and self.theta is None:
            raise FilterConfigError(f"Copula '{self.family}' requires theta.")
        if self.family == "gumbel" and self.theta < 1.0:
            raise FilterConfigError(
                f"Copula 'gumbel' requires theta >= 1, got {self.theta}."
            )
        if self.family in _PARAMETERLESS_FAMILIES and self.theta is not None:
            raise FilterConfigError(
                f"Copula '{self.family}' takes no theta (got {self.theta}). "
                f"Remove it, or use a parametric family (frank, gumbel)."
            )

    @classmethod
    def from_dict(cls, block: dict, where: str) -> "CopulaSpec":
        _reject_unknown_keys(block, {"family", "theta"}, where)
        if "family" not in block:
            raise FilterConfigError(f"Copula block in {where} lacks 'family'.")
        theta = block.get("theta")
        return cls(block["family"], None if theta is None else float(theta))


INDEPENDENCE = CopulaSpec()  # module-level default, single instance


@dataclass(frozen=True)
class CurveSpec:
    """A marginal survival filter: curve name + ordered params, plus an
    optional atom at zero (share for whom no positive value is
    acceptable; see core.decay_curves.with_atom).

    Frozen and fully value-based, so (matrix_id, CurveSpec) is a valid
    dict key — this IS the marginal-cache deduplication key.
    """
    curve: str
    params: tuple[float, ...]
    atom: float = 0.0

    def __post_init__(self):
        if not (0.0 <= self.atom <= 1.0):
            raise FilterConfigError(
                f"Curve '{self.curve}' atom must be in [0, 1], "
                f"got {self.atom}."
            )

    @classmethod
    def from_dict(cls, block: dict, where: str) -> "CurveSpec":
        if "curve" not in block:
            raise FilterConfigError(f"Curve block in {where} lacks 'curve'.")
        curve = block["curve"]
        if curve not in _CURVE_ARITY:
            raise FilterConfigError(
                f"Unknown curve '{curve}' in {where}. "
                f"Known: {sorted(_CURVE_ARITY)}."
            )
        if "scaling" in block:
            raise FilterConfigError(
                f"'scaling' inside curve block in {where}: scaling is not "
                f"part of a probability marginal. Move it to the class "
                f"filter level."
            )
        required = _CURVE_ARITY[curve]
        _reject_unknown_keys(block, {"curve", "atom", *required}, where)
        missing = [p for p in required if p not in block]
        if missing:
            raise FilterConfigError(
                f"Curve '{curve}' in {where} missing parameter(s) {missing}."
            )
        params = tuple(float(block[p]) for p in required)
        from ikob2.core import families
        if curve in families.FAMILIES:
            try:
                families.validate_params(curve, params)
            except ValueError as exc:
                raise FilterConfigError(f"{where}: {exc}") from None
        return cls(curve, params, atom=float(block.get("atom", 0.0)))


@dataclass(frozen=True)
class ClassFilter:
    """Fully resolved filter recipe for one (mode, income class)."""
    time: CurveSpec
    cost: CurveSpec | None
    copula: CopulaSpec
    scaling: float = 1.0

    def __post_init__(self):
        if not (0.0 <= self.scaling <= 1.0):
            raise FilterConfigError(
                f"scaling must be in [0,1], got {self.scaling}."
            )


@dataclass(frozen=True)
class FilterConfig:
    """filters[mode][income_class] -> ClassFilter, everything resolved."""
    filters: dict  # mode -> {class -> ClassFilter}

    def additivity_armed(self, mode: str) -> bool:
        """Hansen additivity is exact iff all classes of a mode share
        one composed matrix, i.e. identical ClassFilters."""
        per_class = self.filters[mode]
        return len(set(per_class.values())) == 1

    def unique_marginals(self) -> set[tuple[str, CurveSpec]]:
        """Deduplicated marginal workload: {(matrix_id, CurveSpec)}.
        matrix_id is 'time:<mode>' or 'cost:<mode>'."""
        out = set()
        for mode, per_class in self.filters.items():
            for cf in per_class.values():
                out.add((f"time:{mode}", cf.time))
                if cf.cost is not None:
                    out.add((f"cost:{mode}", cf.cost))
        return out


def load_filter_config(path) -> FilterConfig:
    path = pathlib.Path(path)
    with open(path) as fh:
        raw = json.load(fh)

    _reject_unknown_keys(raw, {"copula", "modes"}, f"top level of {path.name}")

    global_copula = INDEPENDENCE
    if "copula" in raw:
        global_copula = CopulaSpec.from_dict(raw["copula"], "top-level copula")

    if "modes" not in raw or not raw["modes"]:
        raise FilterConfigError(f"{path.name} defines no modes.")

    filters = {}
    for mode, mode_block in raw["modes"].items():
        where_mode = f"mode '{mode}'"
        _reject_unknown_keys(mode_block, {"copula", "classes"}, where_mode)

        mode_copula = global_copula
        if "copula" in mode_block:
            mode_copula = CopulaSpec.from_dict(
                mode_block["copula"], f"{where_mode} copula"
            )

        classes = mode_block.get("classes")
        if not classes:
            raise FilterConfigError(f"{where_mode} defines no classes.")
        got, want = set(classes), set(INCOME_CLASSES)
        if got != want:
            raise FilterConfigError(
                f"{where_mode}: classes must be exactly "
                f"{list(INCOME_CLASSES)}; missing {sorted(want - got)}, "
                f"unexpected {sorted(got - want)}."
            )

        per_class = {}
        for cls_name in INCOME_CLASSES:  # canonical order, always
            block = classes[cls_name]
            where = f"{where_mode}, class '{cls_name}'"
            _reject_unknown_keys(
                block, {"time", "cost", "scaling", "copula"}, where
            )
            if "time" not in block:
                raise FilterConfigError(f"{where} lacks required 'time'.")

            copula = mode_copula
            if "copula" in block:
                copula = CopulaSpec.from_dict(block["copula"], f"{where} copula")

            cost = None
            if "cost" in block:
                cost = CurveSpec.from_dict(block["cost"], f"{where} cost")
            elif copula is not INDEPENDENCE and "copula" in block:
                # A class-level copula without a cost block is dead
                # config: C(u, 1) = u for every family. Loud, not silent.
                raise FilterConfigError(
                    f"{where} sets a copula but has no 'cost' filter; "
                    f"the copula would have no effect."
                )

            per_class[cls_name] = ClassFilter(
                time=CurveSpec.from_dict(block["time"], f"{where} time"),
                cost=cost,
                copula=copula,
                scaling=float(block.get("scaling", 1.0)),
            )

        filters[mode] = per_class

    return FilterConfig(filters)