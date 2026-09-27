"""
Filter specifications: survival curves, copulas and their composition.

  CurveSpec    a marginal survival filter: curve name, ordered parameters,
               optional atom at zero; `from_dict` reads a JSON-style block
               such as {"curve": "weibull", "shape": 2.5, "scale": 40};
  CopulaSpec   a survival copula family and its theta;
  ClassFilter  time marginal, optional cost marginal, copula and scaling:
               the full recipe of one segment's weight matrix.

Fail-loud policy for curve and copula blocks:
  * unknown keys are errors;
  * "scaling" is forbidden inside curve blocks (it is not part of a
    probability marginal; it lives on ClassFilter and is applied after
    composition, see core/compose.py);
  * "frank" and "gumbel" require theta, all other families forbid it;
    "gumbel" (Gumbel-Hougaard) requires theta >= 1 (theta = 1 is
    independence; "comonotone" is the theta -> inf limit).

epsilon is not part of a filter: sparsification has a single owner (the
runner) and is applied once, to the composed matrix.
"""

from dataclasses import dataclass

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
    "quadratic_ramp": ("low", "high"),
    "triangular": ("low", "mode", "high"),
}
# Curves given by knots: {"curve": ..., "knots": [[z, f], ...]}
_KNOT_CURVES = {"piecewise_linear", "piecewise_quadratic"}


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
    def _from_knots(cls, block: dict, curve: str, where: str) -> "CurveSpec":
        from ikob2.core import families
        _reject_unknown_keys(block, {"curve", "atom", "knots"}, where)
        if "knots" not in block:
            raise FilterConfigError(
                f"Curve '{curve}' in {where} needs 'knots': [[z, f], ...].")
        try:
            knots = [(float(z), float(f)) for z, f in block["knots"]]
        except (TypeError, ValueError):
            raise FilterConfigError(
                f"'knots' in {where} must be a list of [z, f] pairs.") \
                from None
        params = tuple(v for pair in knots for v in pair)
        try:
            families.validate_params(curve, params)
        except ValueError as exc:
            raise FilterConfigError(f"{where}: {exc}") from None
        return cls(curve, params, atom=float(block.get("atom", 0.0)))

    @classmethod
    def from_dict(cls, block: dict, where: str) -> "CurveSpec":
        if "curve" not in block:
            raise FilterConfigError(f"Curve block in {where} lacks 'curve'.")
        curve = block["curve"]
        if curve not in _CURVE_ARITY and curve not in _KNOT_CURVES:
            raise FilterConfigError(
                f"Unknown curve '{curve}' in {where}. "
                f"Known: {sorted(set(_CURVE_ARITY) | _KNOT_CURVES)}."
            )
        if "scaling" in block:
            raise FilterConfigError(
                f"'scaling' inside curve block in {where}: scaling is not "
                f"part of a probability marginal. Move it to the class "
                f"filter level."
            )
        if curve in _KNOT_CURVES:
            return cls._from_knots(block, curve, where)
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
    """Fully resolved filter recipe of one segment and mode."""
    time: CurveSpec
    cost: CurveSpec | None
    copula: CopulaSpec
    scaling: float = 1.0

    def __post_init__(self):
        if not (0.0 <= self.scaling <= 1.0):
            raise FilterConfigError(
                f"scaling must be in [0,1], got {self.scaling}."
            )
