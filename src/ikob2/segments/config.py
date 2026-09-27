"""Configuration of the segment pipeline (mirrors `cfg` of the R script)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from ikob2.params import DEFAULTS

HOUSEHOLD_TYPES = ("single", "couple", "single_parent", "couple_children")
INCOME_CLASSES = tuple(f"D{i}" for i in range(1, 11)) + ("onbekend",)
LOW_INCOME_CLASSES = tuple(DEFAULTS.segments.low_income_classes)
_SEG = DEFAULTS.segments


def _default_kwb_vars() -> dict[str, str]:
    return {
        "zonecode": "buurtcode",
        "zonename": "buurtnaam",
        "citycode": "gemeentecode",
        "cityname": "gemeentenaam",
        "inhabitants": "aantal_inwoners",
        "households": "aantal_huishoudens",
        "p_hh_single": "percentage_eenpersoonshuishoudens",
        "p_hh_no_child": "percentage_huishoudens_zonder_kinderen",
        "p_hh_with_child": "percentage_huishoudens_met_kinderen",
        "stedelijkheid": "stedelijkheid_adressen_per_km2",
        "avg_house_value": "gemiddelde_woningwaarde",
        # 2022 vintage. The 2024 file also has
        # 'percentage_huishoudens_met_laagste_inkomen'; verify its
        # definition before switching (see docs).
        "p_hh_low_income": "percentage_huishoudens_met_laag_inkomen",
    }


@dataclass(frozen=True)
class SegmentConfig:
    # ── study area ───────────────────────────────────────────────────
    gemeente_codes: tuple[str, ...] | None = None
    reference_gemeente_codes: tuple[str, ...] | None = None

    # ── KWB ──────────────────────────────────────────────────────────
    kwb_layer: str | None = "buurten"
    kwb_vars: dict = field(default_factory=_default_kwb_vars)

    # ── StatLine ─────────────────────────────────────────────────────
    income_period: str = _SEG.income_period
    children_period: str = _SEG.children_period
    children_age_total: str = "1017000"
    income_population_key: str = "1050010"
    income_total_col: str = "ParticuliereHuishoudens_1"   # x 1000
    household_key_map: dict = field(default_factory=lambda: {
        "single": "1050015",
        "couple": "1017780",
        "single_parent": "1050190",
        "couple_children": "1016090",
    })
    income_decile_cols: dict = field(default_factory=lambda: {
        f"D{i}": f"GestandaardiseerdInkomen{i}e10Groep_{6 + i}"
        for i in range(1, 11)
    })

    # ── model ────────────────────────────────────────────────────────
    household_types: tuple[str, ...] = HOUSEHOLD_TYPES
    income_classes: tuple[str, ...] = INCOME_CLASSES
    hh_size: dict = field(default_factory=lambda: _SEG.hh_size.to_dict())
    fallback_single_parent_share_of_with_children: float = \
        _SEG.single_parent_fallback_share
    # KWB "zonder kinderen" excludes one-person households (FALSE gave
    # a plausible distribution in the R run).
    kwb_no_child_includes_single: bool = _SEG.kwb_no_child_includes_single
    fallback_hh_composition: dict = field(
        default_factory=lambda: _SEG.fallback_hh_composition.to_dict())
    local_income_calibration: Literal["laagste40", "none"] = \
        _SEG.local_income_calibration

    ipf_tol: float = _SEG.ipf_tol
    ipf_max_iter: int = _SEG.ipf_max_iter

    @property
    def segment_columns(self) -> list[str]:
        """Column-major, as R's as.vector(outer(types, classes, paste)):
        all household types for D1, then for D2, ..."""
        return [f"{h}_{c}" for c in self.income_classes
                for h in self.household_types]
