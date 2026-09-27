"""
Lazy matrix registry.

Stores *recipes* (key -> callable) instead of materialised N×N
matrices. get() evaluates a recipe on demand and by default does NOT
cache the result, keeping peak memory flat (one derived matrix live at
a time). pin() trades memory for speed on hot keys.

Recipes receive the registry itself, so they can compose:

    registry.register(("marginal", key), lambda r: evaluate(...))
    registry.register(("decay", key),
                      lambda r: compose(r.get(("marginal", a)), r.get(("marginal", b))))
"""

import logging
from typing import Callable, Hashable

from ikob2.core.numerics import nbytes_of

logger = logging.getLogger(__name__)


class MatrixRegistry:
    def __init__(self):
        self._recipes: dict[Hashable, Callable[["MatrixRegistry"], object]] = {}
        self._pinned: dict[Hashable, object] = {}

    # ── Registration ─────────────────────────────────────────────────

    def register(self, key: Hashable, recipe: Callable[["MatrixRegistry"], object]) -> None:
        """Register a recipe: a callable (registry) -> matrix."""
        self._recipes[key] = recipe

    def register_value(self, key: Hashable, value) -> None:
        """Pin a precomputed matrix directly."""
        self._pinned[key] = value

    # ── Access ───────────────────────────────────────────────────────

    def __contains__(self, key: Hashable) -> bool:
        return key in self._pinned or key in self._recipes

    def get(self, key: Hashable):
        """Evaluate *key*. Pinned values are returned directly; recipe
        results are intentionally NOT cached (memory stays flat)."""
        if key in self._pinned:
            return self._pinned[key]
        if key in self._recipes:
            return self._recipes[key](self)
        raise KeyError(f"No recipe or value registered for key: {key!r}")

    # ── Memory control ───────────────────────────────────────────────

    def pin(self, key: Hashable):
        """Materialise and cache *key* (subsequent get()s are free)."""
        if key not in self._pinned:
            self._pinned[key] = self.get(key)
        return self._pinned[key]

    def unpin(self, key: Hashable) -> None:
        self._pinned.pop(key, None)

    def clear(self) -> None:
        n = len(self._pinned)
        self._pinned.clear()
        logger.debug("Cleared %d pinned matrices.", n)

    def pinned_size_mb(self) -> float:
        return sum(nbytes_of(v) for v in self._pinned.values()) / (1024 * 1024)

    def recipe_count(self) -> int:
        return len(self._recipes)