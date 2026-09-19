"""
Runners.

SimulationRunner — single-population baseline (variants + Shen model).

SegmentedRunner — multi-segment run with explicit competition POOLS
and composed tolerance FILTERS.

  Each segment belongs to a pool (Segment.pool). Segments compete only
  within their pool, against that pool's opportunity supply. The
  caller passes `opportunities` as a pool -> (n,) vector mapping;
  omitting it puts everything in one "default" pool fed by
  state.opportunities (backwards compatible un-pooled behaviour).

  Each segment also names a FILTER IDENTITY:

      Segment.weight_key == (time_cost_id, money_cost_id, ClassFilter)

  The weight matrix for a segment is a joint survival filter: a time
  marginal and an optional money marginal composed through a survival
  copula (core.compose). A segment without a cost filter collapses
  exactly to the pure time filter (copula boundary C(u, 1) = u), so
  time-only segments take the same code path with no special casing.

Performance patterns:
  1. Marginal dedup: raw curve evaluations are registered under
     ("marginal", matrix_id, CurveSpec) and shared by every filter
     that references them. The exp-heavy work happens once per unique
     marginal — the expected production shape (one shared time curve,
     per-class cost curves) evaluates 5 marginals, not 8.
  2. Composed dedup: composed matrices are keyed on weight_key ONLY.
     Segments in different pools with equal ClassFilters share one
     matrix; pools multiply matvecs (cheap), never matrices
     (expensive). Because ClassFilter is frozen and value-based, this
     is the same identity notion FilterConfig.additivity_armed() uses
     — engine and invariants cannot disagree about matrix sharing.
  3. Matmul batching: populations are summed per (pool, weight_key)
     before the matvec.
  4. pin_marginals (default True): marginals stay materialised for
     the whole run, so re-deriving an unpinned composed matrix costs
     one elementwise copula pass instead of curve evaluation. This is
     the memory/speed knob that matters at NRM zone counts; the
     pinned footprint is logged.

epsilon policy: sparsification strength has a SINGLE OWNER (the CLI).
It arrives through the constructor — a required keyword, deliberately
without default — and is applied exactly once, inside compose_filters,
to each composed matrix. Marginals are NEVER sparsified: the copula
needs genuine probabilities, and C(u,v) <= min(u,v) guarantees that
thresholding after composition is equivalent anyway.

The Frechet–Hoeffding sandwich (certified invariant #3) runs inside
compose_filters, i.e. inside the composed recipes: it re-verifies on
every rebuild. check_frechet=False disables it for production-scale
batches, mirroring the opt-out philosophy of the nesting test.

The weighted total uses safe_divide: zones with zero total population
come out as 0. That is correct INTERNAL semantics (zero weight in the
average), but 0 is misleading as an exported accessibility value —
SegmentedResult therefore also carries total_population so the output
layer can decide how to present empty zones (e.g. NaN).

NOTE: a SegmentedRunner instance is reusable across sequential run()
calls but is NOT thread-safe (the shared registry's recipes are
overwritten per run).
"""

import logging
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from ikob2.core.accessibility import compute_accessibility
from ikob2.core.compose import compose_filters
from ikob2.core.decay_curves import get_decay_function, with_atom
from ikob2.core.numerics import (
    DTYPE,
    as_dtype,
    ensure_dense,
    matvec,
    matvec_T,
    safe_divide,
    zeros,
)
from ikob2.data.validation import validate_pools
from ikob2.domain.segments import Segment
from ikob2.domain.state import ModelState
from ikob2.engine.cache import MatrixRegistry
from ikob2.variants.base import Variant

logger = logging.getLogger(__name__)


# ── Simple single-population runner ──────────────────────────────────

class SimulationRunner:
    def run(self, state: ModelState, variants: Sequence[Variant] = ()) -> np.ndarray:
        for variant in variants:
            state = variant(state)
        return compute_accessibility(state)


# ── Marginal evaluation (module-level: shared by recipes) ────────────

def evaluate_marginal(cost, spec) -> np.ndarray:
    fn = get_decay_function(spec.curve)
    dense = ensure_dense(cost).astype(DTYPE, copy=False)
    result = with_atom(fn(dense, *spec.params), dense, spec.atom)
    logger.info(
        "Marginal %s%s on matrix[min=%.4g p50=%.4g max=%.4g] "
        "-> filter[mean=%.4g, frac>0.05=%.3f]",
        spec.curve, spec.params,
        float(dense.min()), float(np.median(dense)), float(dense.max()),
        float(result.mean()), float((result > 0.05).mean()),
    )
    return result


# ── Segmented runner ─────────────────────────────────────────────────

@dataclass(frozen=True)
class SegmentedResult:
    """Accessibility per segment plus shared intermediate quantities."""
    per_segment: dict[str, np.ndarray]      # segment name -> A_i
    competition: dict[str, np.ndarray]      # pool name -> V_j
    total: np.ndarray                       # population-weighted total A_i
    total_population: np.ndarray            # summed weights behind `total`
    # total is 0 wherever total_population is 0 (safe_divide fill);
    # callers presenting `total` should mask those zones explicitly.


class SegmentedRunner:
    def __init__(
        self,
        registry: MatrixRegistry | None = None,
        *,
        decay_epsilon: float | None,
        pin_decay: bool = False,
        pin_marginals: bool = True,
        check_frechet: bool = True,
    ):
        # decay_epsilon is a REQUIRED keyword with no default: the
        # sparsification choice belongs to the CLI, and forgetting to
        # thread it through must be a TypeError, not a silent drift.
        self.registry = registry or MatrixRegistry()
        self.decay_epsilon = decay_epsilon
        self.pin_decay = pin_decay
        self.pin_marginals = pin_marginals
        self.check_frechet = check_frechet
        self._marginal_keys: set = set()

    # ── Recipe wiring ────────────────────────────────────────────────

    def _register_filter_recipes(
        self,
        cost_matrices: Mapping[str, object],
        segments: Sequence[Segment],
    ) -> None:
        """Two-level recipes: marginals compose into filter matrices.

        ALWAYS overwrite existing recipes: the same cost id can map to
        a different matrix on the next run() call (variants!). A stale
        recipe would silently serve weights from the wrong matrix.
        """
        self._marginal_keys = set()

        for segment in segments:
            cf = segment.class_filter
            time_id = segment.time_cost_id
            money_id = segment.money_cost_id

            # Filter identity and cost wiring must agree, loudly.
            if (cf.cost is None) != (money_id is None):
                raise ValueError(
                    f"Segment '{segment.name}': cost filter and money "
                    f"cost id must be present together (cost="
                    f"{cf.cost!r}, money_cost_id={money_id!r})."
                )
            if time_id not in cost_matrices:
                raise KeyError(
                    f"Segment '{segment.name}' references time cost "
                    f"'{time_id}' but cost_matrices has "
                    f"{sorted(cost_matrices)}."
                )

            tkey = ("marginal", time_id, cf.time)
            tcost = cost_matrices[time_id]
            self.registry.register(
                tkey,
                lambda r, c=tcost, s=cf.time: evaluate_marginal(c, s),
            )
            self._marginal_keys.add(tkey)

            ckey = None
            if cf.cost is not None:
                if money_id not in cost_matrices:
                    raise KeyError(
                        f"Segment '{segment.name}' references money "
                        f"cost '{money_id}' but cost_matrices has "
                        f"{sorted(cost_matrices)}."
                    )
                ckey = ("marginal", money_id, cf.cost)
                mcost = cost_matrices[money_id]
                self.registry.register(
                    ckey,
                    lambda r, c=mcost, s=cf.cost: evaluate_marginal(c, s),
                )
                self._marginal_keys.add(ckey)

            # Composed matrix: pulls its marginals through the
            # registry, so pinned marginals make this an elementwise
            # pass. epsilon and the Frechet check are applied HERE and
            # nowhere else.
            self.registry.register(
                ("decay", segment.weight_key),
                lambda r, tk=tkey, ck=ckey, cf=cf: compose_filters(
                    r.get(tk),
                    r.get(ck) if ck is not None else None,
                    family=cf.copula.family,
                    theta=cf.copula.theta,
                    scaling=cf.scaling,
                    epsilon=self.decay_epsilon,
                    check_frechet=self.check_frechet,
                ),
            )

    # ── Main entry point ─────────────────────────────────────────────

    def run(
        self,
        state: ModelState,
        segments: Sequence[Segment],
        populations: Mapping[str, np.ndarray],
        variants: Sequence[Variant] = (),
        cost_matrices: Mapping[str, object] | None = None,
        opportunities: Mapping[str, np.ndarray] | None = None,
    ) -> SegmentedResult:
        """
        Parameters
        ----------
        state : base ModelState (population, default cost matrix, ...)
        segments : population segments (each names its pool and its
            filter identity; see module docstring)
        populations : segment name -> (n_zones,) population vector
        variants : applied to *state* before anything else
        cost_matrices : cost id -> matrix; defaults to
            {"time": state.generalized_cost}. Segments with cost
            filters additionally require their money_cost_id here.
        opportunities : pool name -> (n_zones,) opportunity vector;
            defaults to {"default": state.opportunities} (single pool).
        """
        for variant in variants:
            state = variant(state)
        state.validate()

        n = state.n_zones
        if cost_matrices is None:
            cost_matrices = {"time": state.generalized_cost}
        if opportunities is None:
            opportunities = {"default": state.opportunities}

        missing_pop = [s.name for s in segments if s.name not in populations]
        if missing_pop:
            raise KeyError(f"No population vector for segments: {missing_pop}")

        # Pool validation: hard errors raise, warnings are logged.
        validate_pools(
            segments, opportunities,
            populations=populations,
            total_opportunities=state.opportunities,
        ).raise_if_failed()

        opportunities = {
            pool: as_dtype(vec) for pool, vec in opportunities.items()
        }
        for pool, vec in opportunities.items():
            if vec.shape != (n,):
                raise ValueError(
                    f"Opportunities for pool '{pool}' have shape "
                    f"{vec.shape}, expected ({n},)"
                )

        self._register_filter_recipes(cost_matrices, segments)

        # ── Batch populations by (pool, weight_key) ──────────────────
        batches: dict[tuple, np.ndarray] = {}
        members: dict[tuple, list[Segment]] = {}
        for segment in segments:
            bk = (segment.pool, segment.weight_key)
            if bk not in batches:
                batches[bk] = zeros(n)
                members[bk] = []
            pop = as_dtype(populations[segment.name])
            if pop.shape != (n,):
                raise ValueError(
                    f"Population for {segment.name} has shape {pop.shape}, "
                    f"expected ({n},)"
                )
            batches[bk] += pop
            members[bk].append(segment)

        unique_wks = {wk for (_, wk) in batches}
        logger.info(
            "Segmented run: %d segments -> %d pools, %d unique composed "
            "filters from %d unique marginals, %d (pool x filter) "
            "matvec batches.",
            len(segments), len(opportunities), len(unique_wks),
            len(self._marginal_keys), len(batches),
        )

        run_keys = [("decay", wk) for wk in unique_wks]
        # Drop pins left over from a previous (possibly crashed) run so
        # they cannot shadow the freshly registered recipes.
        for key in run_keys:
            self.registry.unpin(key)
        for mkey in self._marginal_keys:
            self.registry.unpin(mkey)

        try:
            # Pin marginals for the whole run: recomposition then
            # costs one elementwise pass. This is the knob to watch at
            # NRM scale, hence the footprint log line.
            if self.pin_marginals:
                for mkey in self._marginal_keys:
                    self.registry.pin(mkey)
                logger.info(
                    "Pinned %d marginal matrices (%.1f MB).",
                    len(self._marginal_keys),
                    self.registry.pinned_size_mb(),
                )

            # ── Pass 1: per-pool competition V_j ─────────────────────
            # float64 accumulators: with many segments/keys, float32
            # accumulation is the one place batching loses precision.
            competition64: dict[str, np.ndarray] = {
                pool: np.zeros(n, dtype=np.float64) for pool in opportunities
            }
            for (pool, wk), pop_batch in batches.items():
                key = ("decay", wk)
                D = (self.registry.pin(key) if self.pin_decay
                     else self.registry.get(key))
                competition64[pool] += matvec_T(D, pop_batch)

            competition = {
                pool: v.astype(DTYPE) for pool, v in competition64.items()
            }
            # Zero-competition zones contribute ZERO accessibility
            # (safe_divide fill), never O / floor.
            adjusted = {
                pool: safe_divide(opportunities[pool], competition[pool])
                for pool in opportunities
            }

            # ── Pass 2: A_i per (pool, weight_key) batch ─────────────
            # Grouped by weight_key so each composed matrix is
            # materialised once and reused across pools before being
            # dropped.
            pools_by_wk: dict[tuple, list[str]] = {}
            for (pool, wk) in batches:
                pools_by_wk.setdefault(wk, []).append(pool)

            per_segment: dict[str, np.ndarray] = {}
            total = zeros(n)
            total_population = zeros(n)

            for wk, wk_pools in pools_by_wk.items():
                key = ("decay", wk)
                D = self.registry.get(key)  # pinned -> free; else
                                            # recomposed (cheap when
                                            # marginals are pinned)
                for pool in wk_pools:
                    accessibility = matvec(D, adjusted[pool])
                    if state.zone_weights is not None:
                        accessibility = (
                            accessibility * state.zone_weights
                        ).astype(DTYPE)
                    for segment in members[(pool, wk)]:
                        # copy: segments sharing a batch must not alias
                        # one buffer, or a caller mutating one result
                        # mutates all
                        per_segment[segment.name] = accessibility.copy()
                        pop = as_dtype(populations[segment.name])
                        total += accessibility * pop
                        total_population += pop
                if self.pin_decay:
                    self.registry.unpin(key)
        finally:
            # Never leak pinned matrices past this run, even on error.
            for key in run_keys:
                self.registry.unpin(key)
            for mkey in self._marginal_keys:
                self.registry.unpin(mkey)

        return SegmentedResult(
            per_segment=per_segment,
            competition=competition,
            total=safe_divide(total, total_population),
            total_population=total_population,
        )

    # ── Hansen (no competition) ──────────────────────────────────────

    def run_hansen(
        self,
        state: ModelState | None,
        segments: Sequence[Segment],
        variants: Sequence[Variant] = (),
        cost_matrices: Mapping[str, object] | None = None,
        opportunities: Mapping[str, np.ndarray] | None = None,
    ) -> dict[str, np.ndarray]:
        """
        Potential accessibility per segment, without competition:

            a_i^s = sum_j D_j^pool(s) * f_s(t_ij, c_ij)

        the measure of the threshold-gate model (the expected number of
        acceptable opportunities). Each segment reads its own pool's
        opportunity vector, so income-matched jobs D_{j,s} are pools
        with one segment group each. No populations are needed.

        Two modes:

        * With a `state` (square): origins and destinations are the same
          n zones; variants, the state's zone_weights and its default
          cost matrix / opportunities apply, as in run().
        * With `state=None` (rectangular): origins i and destinations j
          are different zone sets. `cost_matrices` (n_origins x
          n_destinations, dense or scipy sparse) and `opportunities`
          (pool -> (n_destinations,)) are then required, variants are
          not supported (they transform a ModelState), and the result
          has n_origins entries. This is the study-area case: a few
          hundred origin zones against every destination that matters,
          a matrix of megabytes instead of a national n x n one. Only
          the destination side needs jobs; only the origin side needs
          rows in the skims.

        Composed filter matrices are deduplicated on weight_key exactly
        as in run(): segments with equal filters share one matrix.
        Returns segment name -> (n_origins,) accessibility.
        """
        if state is not None:
            for variant in variants:
                state = variant(state)
            state.validate()
            n_origins = n_dest = state.n_zones
            if cost_matrices is None:
                cost_matrices = {"time": state.generalized_cost}
            if opportunities is None:
                opportunities = {"default": state.opportunities}
            total_opportunities = state.opportunities
        else:
            if variants:
                raise ValueError(
                    "variants transform a ModelState; pass a state, or "
                    "apply the variant to the cost matrices yourself.")
            if cost_matrices is None or opportunities is None:
                raise ValueError(
                    "Rectangular run_hansen (state=None) needs both "
                    "cost_matrices and opportunities.")
            n_origins = n_dest = None
            total_opportunities = None

        validate_pools(
            segments, opportunities,
            total_opportunities=total_opportunities,
        ).raise_if_failed()
        opportunities = {
            pool: as_dtype(vec) for pool, vec in opportunities.items()
        }
        lengths = {pool: vec.shape for pool, vec in opportunities.items()}
        if any(len(shape) != 1 for shape in lengths.values()):
            raise ValueError(f"Opportunity vectors must be 1-D: {lengths}")
        if n_dest is None:
            dests = {shape[0] for shape in lengths.values()}
            if len(dests) != 1:
                raise ValueError(
                    f"Opportunity pools disagree on the number of "
                    f"destinations: {lengths}")
            n_dest = dests.pop()
        for pool, (length,) in lengths.items():
            if length != n_dest:
                raise ValueError(
                    f"Opportunities for pool '{pool}' have shape "
                    f"({length},), expected ({n_dest},)")

        self._register_filter_recipes(cost_matrices, segments)

        if n_origins is None:
            first = segments[0].time_cost_id
            n_origins = cost_matrices[first].shape[0]
        expected = (n_origins, n_dest)
        for segment in segments:
            for cost_id in (segment.time_cost_id, segment.money_cost_id):
                if cost_id is None:
                    continue
                shape = tuple(cost_matrices[cost_id].shape)
                if shape != expected:
                    raise ValueError(
                        f"Cost matrix '{cost_id}' (segment "
                        f"'{segment.name}') has shape {shape}, expected "
                        f"(origins, destinations) = {expected}.")

        by_wk: dict[tuple, list[Segment]] = {}
        for segment in segments:
            by_wk.setdefault(segment.weight_key, []).append(segment)

        run_keys = [("decay", wk) for wk in by_wk]
        for key in run_keys:
            self.registry.unpin(key)
        for mkey in self._marginal_keys:
            self.registry.unpin(mkey)

        per_segment: dict[str, np.ndarray] = {}
        try:
            if self.pin_marginals:
                for mkey in self._marginal_keys:
                    self.registry.pin(mkey)
            for wk, members in by_wk.items():
                D = self.registry.get(("decay", wk))
                for segment in members:
                    a = matvec(D, opportunities[segment.pool])
                    if state is not None and state.zone_weights is not None:
                        a = (a * state.zone_weights).astype(DTYPE)
                    per_segment[segment.name] = a
        finally:
            for key in run_keys:
                self.registry.unpin(key)
            for mkey in self._marginal_keys:
                self.registry.unpin(mkey)
        return per_segment
