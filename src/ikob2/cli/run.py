"""
CLI entry point for ikob2 (single IKOB version).

This script computes:

1. Hansen (naive gravity), on the EFFECTIVE filter
2. Segmented Hansen — per income class: each class's Hansen against
   its own class's jobs, through that class's tolerance filter. No
   competition, hence no pools; population plays no role.
3. Shen (competition-adjusted gravity), on the EFFECTIVE filter
4. Segmented Shen — income-pooled competition: each income class
   competes only for jobs of its own class, through its own filter.

Exactly one IKOB per run.

Filters
-------
A trip weight is a joint survival probability: the fraction of a
class whose TIME tolerance exceeds the trip time and whose MONEY
tolerance exceeds the trip cost, composed through a survival copula
(core.compose). Per-class filters come from an optional JSON file
(--filter-config, schema in domain/filter_config.py) with copula
resolution class > mode > global > independence.

Without --filter-config every class uses the REFERENCE filter: the
CLI logistic (--decay-*) as time marginal, no cost filter,
independence. Since C(u, 1) = u this is exactly the pre-filter
behaviour — the zero-config path IS the regression test against the
certified run.

The un-segmented models (1) and (3) run on the EFFECTIVE filter:
the shared per-class filter when the config is uniform (so plain
Hansen/Shen see cost filters and fares), the CLI reference when
per-class filters genuinely differ (no canonical single filter
exists; the fallback is logged). Without --filter-config the
effective filter IS the reference, so the zero-config path is
unchanged. All matrices — effective included — are built through
the same composer path, so the nesting test compares identical
arithmetic.

Money
-----
The money skim is an affine fare model over the container's freeflow
distance skim: fee + rate * detour * km (domain/fare.py — this IS
the Dutch tariff structure, not an approximation of it). Fares are
scenario assumptions owned by the CLI, never baked into the cache.

--fare-fee and --fare-rate-km have NO defaults, deliberately: every
other check in this run is decay-invariant or self-referential, so
a silently defaulted fare level would pass all of them. When cost
filters are configured, fares must be stated; 0 0 is valid and means
free travel. check_money_sanity is the ONLY level-sensitive check in
the pipeline — it exists to catch unit errors (metres/km) that
nothing else can see.

The money matrix's identity is FareModel.matrix_id, and that string
is its key in cost_matrices — which makes it the marginal-cache and
registry key too. Different fare assumptions therefore can never
share a composed matrix, with zero special-casing downstream.

Model notes
-----------
* Sparsification epsilon (--decay-epsilon) is owned HERE and applied
  exactly once per composed matrix, in compose_filters. It feeds
  ModelState (validation), the local FilterMatrixCache (models 1-3)
  and the SegmentedRunner constructor (model 4); the paths cannot
  drift apart because none of them has a default.
* Nesting test: a single-pool segmented run ON THE EFFECTIVE FILTER
  must reproduce single Shen exactly. This stays armed regardless of
  per-class filter config — it certifies pool mechanics, and (when
  the effective filter carries a cost curve) that the runner's
  registry and the local cache compose the money marginal
  identically.
* Hansen additivity: Hansen is LINEAR in opportunities, so when all
  classes share ONE filter, per-class columns must sum to the Hansen
  of that shared filter — which, under a uniform config, IS plain
  Hansen (model 1), since both use the same effective matrix. The
  check is self-referential (it compares against D_shared @
  O_total), stays armed for any uniform config, and disarms itself
  only for genuinely per-class filters.
* Frechet–Hoeffding sandwich (invariant #3) runs inside every
  composition: max(u+v-1,0) <= C(u,v) <= min(u,v) holds for EVERY
  valid copula, so a violation means the copula evaluation or theta
  handling is broken. --no-frechet-check disables it at scale.
* Hansen has NO balance identity; balance lines are logged for Shen
  models only. Per class, shen_k / hansen_k is the competition
  discount: what fraction of nominally reachable class-k jobs
  survives competition from class-k workers.
* Segmented Shen output: one column per income class plus the
  population-weighted total. Zones with zero residents get NaN in
  the total column (0 would silently poison downstream averages).
* Balance identities are decay-invariant and CANNOT catch filter,
  epsilon or FARE mismatches — that is what the nesting test and the
  money sanity check are for, respectively.

Paths
-----
Cache and output defaults are anchored to the REPO CHECKOUT
(examples/test/{cache,output}), not the CWD — a CWD-relative
default is how output directories previously appeared inside
src/ikob2/cli/. Both are plain defaults; --cache-root and --out
override them. Resolved locations are logged at startup.
"""

import argparse
import logging
from pathlib import Path

import numpy as np

from ikob2.core.accessibility import (
    compute_accessibility,              # Shen
    compute_naive_accessibility,        # Hansen
)
from ikob2.core.compose import compose_filters
from ikob2.core.numerics import as_dtype, matvec
from ikob2.data.container import get_container
from ikob2.data.readers.csv_reader import CsvIndex
from ikob2.domain.fare import FareModel
from ikob2.domain.filter_config import (
    INCOME_CLASSES,
    INDEPENDENCE,
    ClassFilter,
    CurveSpec,
    FilterConfigError,
    load_filter_config,
)
from ikob2.domain.segments import CarAccess, Income, Preference, Segment
from ikob2.domain.state import ModelState
from ikob2.engine.cache import MatrixRegistry
from ikob2.engine.runner import SegmentedRunner, evaluate_marginal
from ikob2.outputs.async_writer import AsyncCsvWriter

# Default data locations, anchored to the repo checkout rather than
# the CWD. run.py lives at src/ikob2/cli/run.py, so parents[3] is
# the repository root; examples/test is where test_cost_filter.json
# and the other run fixtures live. Both are ONLY defaults —
# --cache-root and --out override them — so an installed
# (site-packages) copy of the package, where parents[3] is
# meaningless, must simply pass both flags explicitly. Keep both
# directories in .gitignore.
_REPO_ROOT = Path(__file__).resolve().parents[3]
_EXAMPLES_TEST = _REPO_ROOT / "examples" / "test"

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

def parse_args(argv=None):
    parser = argparse.ArgumentParser("ikob2 run script (single IKOB)")

    parser.add_argument("--ikob", required=True,
                        help="Single IKOB identifier (e.g., IKOB01)")
    parser.add_argument("--scenario", type=str, default="2018")
    parser.add_argument("--segs-base", type=Path, required=True)

    # Skim locations
    parser.add_argument("--skim-root", type=Path, required=True,
                        help="Root directory of LMS NRMdata")
    parser.add_argument("--omnummer-root", type=Path, required=True,
                        help="Directory containing IKOB Omnummer files")

    # Container cache
    parser.add_argument(
        "--cache-root", type=Path,
        default=_EXAMPLES_TEST / "cache",
        help="Directory for binary run containers "
             "(default: examples/test/cache in the repo checkout).")
    parser.add_argument("--force-rebuild", action="store_true",
                        help="Ignore existing container and re-parse CSVs")

    # Generalized cost parameters (applied at run time, cache-free)
    parser.add_argument("--beta-time", type=float, default=1.0,
                        help="Generalized cost weight for travel time")
    parser.add_argument("--beta-distance", type=float, default=0.02,
                        help="Generalized cost weight for distance (per km)")

    # REFERENCE filter: the nesting baseline and the fallback for the
    # un-segmented models under heterogeneous per-class filters.
    # Per-class filters come from --filter-config; without it, every
    # class uses this reference (certified-run path).
    parser.add_argument("--decay-alpha", type=float, default=0.125)
    parser.add_argument("--decay-omega", type=float, default=45.0)
    parser.add_argument("--decay-scaling", type=float, default=1.0)

    # Per-class tolerance filters (time x money via copula).
    parser.add_argument(
        "--filter-config", type=Path, default=None,
        help="JSON filter config (domain/filter_config.py schema). "
             "Omit for uniform reference behaviour.")
    parser.add_argument(
        "--filter-mode", type=str, default=None,
        help="Mode to select from the filter config. Only required "
             "when the config defines more than one mode.")

    # Fare model (consulted only when the filter config declares cost
    # filters). No defaults on fee/rate ON PURPOSE: a fabricated fare
    # level would pass every decay-invariant check in this run
    # unnoticed. Refuse to guess — same policy as income_enum.
    parser.add_argument("--fare-fee", type=float, default=None,
                        help="Starting fee in EUR (e.g. PT boarding "
                             "fee; 0 for car). Required with cost "
                             "filters.")
    parser.add_argument("--fare-rate-km", type=float, default=None,
                        help="Cost per km in EUR (fuel or tariff). "
                             "Required with cost filters.")
    parser.add_argument("--fare-detour", type=float, default=1.0,
                        help="Routed-km / car-network-km ratio "
                             "(car: 1.0, OV proxy: ~1.2-1.4). The "
                             "money skim derives from the CAR "
                             "freeflow distance network.")

    # Sparsification epsilon: SINGLE OWNER for the whole run, applied
    # once per COMPOSED matrix (marginals are never sparsified; see
    # core/compose.py for why that is exact). Not just a numerics
    # knob: for the default logistic (alpha=0.125, omega=45) it sets
    # the effective reach horizon — 1e-6 keeps weights out to ~155
    # cost units, 1e-3 amputates beyond ~100.
    parser.add_argument(
        "--decay-epsilon", type=float, default=1e-6,
        help="Decay-weight sparsification threshold, applied "
             "identically to all composed matrices in the run.")

    parser.add_argument(
        "--nesting-test", action="store_true",
        help="Also run the segmented model with a single pool on the "
             "EFFECTIVE filter and verify it reproduces single Shen "
             "(hard-fails on mismatch)")

    parser.add_argument(
        "--no-frechet-check", dest="check_frechet",
        action="store_false", default=True,
        help="Disable the Frechet-Hoeffding sandwich invariant on "
             "composed filters (only sensible at large zone counts).")

    # Composed-matrix pinning: with per-class filters each composed
    # matrix serves one pool, so pinning trades 4x matrix memory for
    # skipping one recomposition per matrix. With pinned marginals
    # that recomposition is a single elementwise pass — hence both
    # knobs, both defaulting to the cheap-and-fast configuration.
    parser.add_argument("--no-pin-decay", dest="pin_decay",
                        action="store_false", default=True)
    parser.add_argument("--no-pin-marginals", dest="pin_marginals",
                        action="store_false", default=True)

    parser.add_argument(
        "--out", type=Path,
        default=_EXAMPLES_TEST / "output",
        help="Output directory; per-IKOB results go to OUT/<ikob>/ "
             "(default: examples/test/output in the repo checkout).")
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])

    return parser.parse_args(argv)


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

# SEGS income labels -> Income enum. The old expression
# (LOW if "laag" else HIGH) silently misclassified every middle class
# as HIGH; unknown labels now map explicitly or fail loudly.
_INCOME_NAME_MAP = {
    "laag": "LOW",
    "middellaag": "MID_LOW",
    "middelhoog": "MID_HIGH",
    "hoog": "HIGH",
}


def income_enum(label: str) -> Income:
    member = _INCOME_NAME_MAP.get(label.lower())
    if member is not None and hasattr(Income, member):
        return getattr(Income, member)
    # Enum lacks the middle categories (or the label is unknown).
    # Refuse to guess: anything keying off Segment.income downstream
    # would be silently wrong, which is exactly the old bug.
    raise ValueError(
        f"Cannot map income label '{label}' onto Income enum "
        f"({[m.name for m in Income]}). Extend Income or "
        f"_INCOME_NAME_MAP in run.py."
    )


class FilterMatrixCache:
    """Composed-filter matrices for the NON-runner models (Hansen,
    segmented Hansen, single Shen), deduplicated on ClassFilter.

    Marginals are cached separately on (matrix_id, CurveSpec) — the
    same identity the runner's registry uses — so the expected
    production shape (shared time curve, per-class cost curves) costs
    5 marginal evaluations, not 8. Marginals are UNSPARSIFIED;
    epsilon is applied once per composed matrix, in compose_filters.

    money_id is the fare-derived matrix key (FareModel.matrix_id) or
    None on the pure-time path; it is threaded in rather than
    hardcoded so fare identity participates in the cache keys.
    """

    def __init__(self, cost_matrices, *, money_id, epsilon,
                 check_frechet):
        self.cost_matrices = cost_matrices
        self.money_id = money_id
        self.epsilon = epsilon
        self.check_frechet = check_frechet
        self._marginals: dict = {}
        self._composed: dict = {}

    def _marginal(self, matrix_id: str, spec: CurveSpec):
        key = (matrix_id, spec)
        if key not in self._marginals:
            if matrix_id not in self.cost_matrices:
                raise KeyError(
                    f"Filter needs cost matrix '{matrix_id}' but only "
                    f"{sorted(self.cost_matrices)} are available."
                )
            self._marginals[key] = evaluate_marginal(
                self.cost_matrices[matrix_id], spec
            )
        return self._marginals[key]

    def composed(self, cf: ClassFilter):
        if cf not in self._composed:
            u = self._marginal("time", cf.time)
            if cf.cost is not None:
                if self.money_id is None:
                    # resolve_fare guarantees this cannot happen; a
                    # loud error beats a KeyError on None.
                    raise RuntimeError(
                        "Cost filter present but no money matrix was "
                        "resolved — resolve_fare/resolve_cost_matrices "
                        "wiring is broken.")
                v = self._marginal(self.money_id, cf.cost)
            else:
                v = None
            self._composed[cf] = compose_filters(
                u, v,
                family=cf.copula.family,
                theta=cf.copula.theta,
                scaling=cf.scaling,
                epsilon=self.epsilon,
                check_frechet=self.check_frechet,
            )
        return self._composed[cf]


def resolve_class_filters(args, income_labels, reference_filter):
    """label -> ClassFilter, from config or uniform reference."""
    if args.filter_config is None:
        logger.info(
            "No filter config: all classes use the reference filter "
            "(time-only logistic alpha=%g omega=%g scaling=%g, "
            "independence). This is the certified-run path.",
            args.decay_alpha, args.decay_omega, args.decay_scaling,
        )
        return {label: reference_filter for label in income_labels}

    fc = load_filter_config(args.filter_config)
    modes = sorted(fc.filters)
    if args.filter_mode is not None:
        if args.filter_mode not in fc.filters:
            raise FilterConfigError(
                f"--filter-mode '{args.filter_mode}' not in config "
                f"modes {modes}."
            )
        mode = args.filter_mode
    elif len(modes) == 1:
        mode = modes[0]
    else:
        raise FilterConfigError(
            f"Config defines modes {modes}; pass --filter-mode."
        )

    per_class = fc.filters[mode]
    if set(per_class) != set(income_labels):
        raise FilterConfigError(
            f"Filter config classes {sorted(per_class)} do not match "
            f"container income classes {sorted(income_labels)}."
        )

    logger.info("Filter config '%s', mode '%s':",
                args.filter_config.name, mode)
    for label in income_labels:
        cf = per_class[label]
        cost_desc = ("none (pure time)" if cf.cost is None
                     else f"{cf.cost.curve}{cf.cost.params}")
        cop_desc = (cf.copula.family if cf.copula.theta is None
                    else f"{cf.copula.family}(theta={cf.copula.theta})")
        logger.info("  %-11s time=%s%s cost=%s copula=%s scaling=%g",
                    label, cf.time.curve, cf.time.params,
                    cost_desc, cop_desc, cf.scaling)
    return {label: per_class[label] for label in income_labels}


def resolve_effective_filter(class_filters, reference_filter):
    """The filter for the un-segmented models (1) and (3).

    Uniform config -> the shared class filter, so single Hansen and
    Shen see cost filters and fares (previously they silently ran
    time-only regardless of config — the source of the 'Hansen is
    fare-invariant' confusion). Heterogeneous config -> there is no
    canonical single filter; fall back to the CLI reference and say
    so. Zero-config path: class filters ARE the reference, so this
    returns the reference and the certified run is unchanged.
    """
    unique = set(class_filters.values())
    if len(unique) == 1:
        eff = next(iter(unique))
        if eff != reference_filter:
            logger.info(
                "Un-segmented Hansen/Shen use the uniform config "
                "filter (cost=%s), not the CLI reference.",
                "none" if eff.cost is None
                else f"{eff.cost.curve}{eff.cost.params}")
        return eff
    logger.info(
        "Un-segmented Hansen/Shen fall back to the REFERENCE filter: "
        "%d distinct per-class filters, no canonical single filter.",
        len(unique))
    return reference_filter


def resolve_fare(args, class_filters):
    """FareModel iff any class has a cost filter; loud otherwise.

    Free travel is --fare-fee 0 --fare-rate-km 0 (a valid, explicit
    statement); an ABSENT fare with cost filters is an error. The
    reverse mismatch — fare given, no cost filters — is legal but
    inert (C(u, 1) = u), and says so in the log.
    """
    needs_money = any(cf.cost is not None
                      for cf in class_filters.values())
    fare_given = (args.fare_fee is not None
                  or args.fare_rate_km is not None)

    if not needs_money:
        if fare_given:
            logger.info(
                "Fare parameters given but no class has a cost "
                "filter: fare is inert this run (C(u, 1) = u).")
        return None

    if args.fare_fee is None or args.fare_rate_km is None:
        raise ValueError(
            "Filter config uses cost filters; pass BOTH --fare-fee "
            "and --fare-rate-km explicitly (0 0 is valid and means "
            "free travel). There is no default fare on purpose: "
            "nothing downstream can detect a wrong cost level."
        )

    fare = FareModel(fee=args.fare_fee, rate_per_km=args.fare_rate_km,
                     detour=args.fare_detour)
    logger.info("Fare model: %s", fare.matrix_id)
    return fare


def check_money_sanity(money, n):
    """The ONLY level-sensitive check in the run, and therefore the
    only defense against unit errors (metres vs km would inflate
    costs by 1000x and sail through every other invariant, all of
    which are decay-invariant or self-referential)."""
    off = money[~np.eye(n, dtype=bool)]
    if not np.isfinite(off).all():
        raise ValueError("Money cost matrix has non-finite "
                         "off-diagonal entries.")
    if float(off.min()) < 0:
        raise ValueError("Money cost matrix has negative "
                         "off-diagonal entries.")
    p10, p50, p90 = np.percentile(off, [10, 50, 90])
    logger.info("Money cost off-diagonal: p10=%.2f p50=%.2f p90=%.2f "
                "EUR", p10, p50, p90)
    if p50 > 100:
        logger.warning(
            "Median off-diagonal money cost %.0f EUR is implausible "
            "for a regional model — check fare parameters and the "
            "distance skim units (container stores METERS; the m->km "
            "conversion lives in IkobContainer.money_cost).", p50)


def resolve_cost_matrices(container, generalized_cost, class_filters,
                          fare, n):
    """{"time": ...} plus {fare.matrix_id: ...} iff cost filters are
    in use. Returns (matrices, money_id); money_id is None on the
    pure-time path.

    Using fare.matrix_id as the dict key is the load-bearing choice:
    it flows into the (matrix_id, CurveSpec) marginal keys of BOTH
    caches and from there into composed-matrix identity, so different
    fare assumptions can never alias — without any downstream code
    knowing fares exist."""
    matrices = {"time": generalized_cost}
    if fare is None:
        return matrices, None
    money = as_dtype(container.money_cost(fare))
    check_money_sanity(money, n)
    matrices[fare.matrix_id] = money
    return matrices, fare.matrix_id


def build_segments(income_labels, class_filters, money_id,
                   pooled: bool):
    """One segment per income class. pooled=True gives each class its
    own competition pool; pooled=False fuses everything into a single
    'default' pool (the nesting/regression configuration)."""
    return [
        Segment(
            name=label,
            income=income_enum(label),
            car_access=CarAccess.WITH_CAR,
            preference=Preference.CAR,
            class_filter=class_filters[label],
            time_cost_id="time",
            money_cost_id=(
                money_id if class_filters[label].cost is not None
                else None
            ),
            pool=label if pooled else "default",
        )
        for label in income_labels
    ]


def compute_segmented_hansen(state, income_labels, class_filters,
                             jobs_by_income, cache: FilterMatrixCache):
    """Per-class Hansen: A_i^k = D_k @ O_k.

    No competition, no pools, no population — the opportunity vector
    and the tolerance filter vary per class. Matrices are
    deduplicated on the frozen ClassFilter inside the cache, so a
    uniform config builds (or reuses) exactly one matrix.
    """
    per_class = {}
    for label in income_labels:
        per_class[label] = compute_naive_accessibility(
            state.with_updates(
                opportunities=as_dtype(jobs_by_income[label])),
            decay_matrix=cache.composed(class_filters[label]),
        )
    return per_class


def check_hansen_additivity(per_class, class_filters, cache,
                            opportunities_total):
    """Hansen linearity: with ONE shared filter the per-class columns
    must sum to D_shared @ O_total (up to float32 summation-order
    noise). Under a uniform config D_shared is the effective matrix,
    so this asserts sum(per-class) == plain Hansen. Self-referential
    on purpose: it stays armed for ANY uniform config, not just the
    CLI reference. Under genuinely per-class filters additivity
    legitimately fails, so the check disarms itself — and says so,
    since arming is config-driven."""
    unique = set(class_filters.values())
    if len(unique) != 1:
        logger.info(
            "Hansen additivity check DISARMED: %d distinct per-class "
            "filters in use (additivity does not apply).", len(unique))
        return
    D = cache.composed(next(iter(unique)))
    expected = matvec(D, as_dtype(opportunities_total))
    total = sum(per_class.values())
    worst = float(np.max(
        np.abs(total - expected) / np.maximum(np.abs(expected), 1e-12)
    ))
    logger.info("Hansen additivity: max rel diff %.3e", worst)
    if worst > 1e-4:
        raise RuntimeError(
            f"Hansen additivity FAILED (max rel diff {worst:.3e}): "
            "per-class columns do not sum to shared-filter Hansen "
            "despite a uniform filter — opportunity vectors and cost "
            "matrix are misaligned.")


def log_balance(name, accessibility, population, opportunities):
    """Population-weighted mean accessibility vs job/worker ratio.
    For Shen-type models these must agree; drift beyond float32 noise
    means the vectors and the cost matrix are misaligned.
    (Hansen models have no such identity — do not add them here.)"""
    pop_sum = float(population.sum())
    if pop_sum <= 0:
        logger.warning("Balance check %s skipped: zero population.", name)
        return
    weighted = float((accessibility * population).sum() / pop_sum)
    expected = float(opportunities.sum() / pop_sum)
    logger.info("Balance %-14s %.4f (job/worker ratio: %.4f)",
                name + ":", weighted, expected)
    if not np.isclose(weighted, expected, rtol=1e-3):
        logger.warning(
            "Balance identity violated for %s (%.4f vs %.4f) — check "
            "SEGS/cost alignment.", name, weighted, expected)


def run_nesting_test(runner, state, income_labels, effective_filter,
                     populations, cost_matrices, money_id,
                     opportunities_total, shen):
    """Single-pool segmented run ON THE EFFECTIVE FILTER must
    reproduce single Shen exactly (same composed matrix, fused pool
    => identical math path). Certifies pool mechanics — and, when
    the effective filter carries a cost curve, that the runner's
    registry and the local FilterMatrixCache compose the money
    marginal identically. Stays armed regardless of per-class
    filter config."""
    logger.info("Nesting test: segmented (single pool, effective "
                "filter) vs single Shen...")
    nesting_filters = {label: effective_filter
                       for label in income_labels}
    result = runner.run(
        state,
        build_segments(income_labels, nesting_filters, money_id,
                       pooled=False),
        populations,
        cost_matrices=cost_matrices,
        opportunities={"default": opportunities_total},
    )
    worst = 0.0
    for label in income_labels:
        diff = float(np.max(np.abs(result.per_segment[label] - shen)))
        worst = max(worst, diff)
    if worst > 1e-4:
        raise RuntimeError(
            f"Nesting test FAILED: max |segmented - shen| = {worst:.3e}. "
            "Pool mechanics do not reproduce single Shen.")
    logger.info("Nesting test PASSED (max abs diff %.3e).", worst)


def resolve_income_labels(container):
    """Canonical class order everywhere: filter config, SEGS column
    semantics and output columns all assume laag-first. Order is
    meaning here, not convention — fail loud on set mismatch."""
    population_by_income = container.population_by_income()
    jobs_by_income = container.jobs_by_income()

    # A jobs/population classification mismatch would previously have
    # been absorbed by the sum() and never noticed.
    if set(population_by_income) != set(jobs_by_income):
        raise ValueError(
            f"Income classes differ between population "
            f"{sorted(population_by_income)} and jobs "
            f"{sorted(jobs_by_income)}."
        )
    if set(population_by_income) != set(INCOME_CLASSES):
        raise ValueError(
            f"Container income classes {sorted(population_by_income)} "
            f"do not match canonical {list(INCOME_CLASSES)}."
        )
    return list(INCOME_CLASSES), population_by_income, jobs_by_income


def write_outputs(out_dir, n, income_labels, hansen, hansen_by_class,
                  shen, seg_result, seg_total_out):
    out_dir.mkdir(parents=True, exist_ok=True)
    zone_index = CsvIndex.zone_index(n)
    writer = AsyncCsvWriter()
    try:
        writer.submit(
            np.column_stack([hansen]),
            out_dir / "accessibility_hansen.csv",
            header=["hansen"],
            index=zone_index,
        )

        # Per-class Hansen + total. total is the SUM of the class
        # columns (jobs are a partition), which equals single Hansen
        # under ANY uniform filter (both then use the effective
        # matrix) but stays meaningful when per-class filters break
        # that equality.
        writer.submit(
            np.column_stack(
                [hansen_by_class[label] for label in income_labels]
                + [sum(hansen_by_class.values())]
            ),
            out_dir / "accessibility_segmented_hansen.csv",
            header=income_labels + ["total"],
            index=zone_index,
        )

        writer.submit(
            np.column_stack([shen]),
            out_dir / "accessibility_shen.csv",
            header=["shen"],
            index=zone_index,
        )

        # One column per income class + weighted total. With real
        # pools the per-class spread IS the result; the total alone
        # would hide it.
        writer.submit(
            np.column_stack(
                [seg_result.per_segment[label] for label in income_labels]
                + [seg_total_out]
            ),
            out_dir / "accessibility_segmented_shen.csv",
            header=income_labels + ["total"],
            index=zone_index,
        )
    finally:
        writer.shutdown()


# ─────────────────────────────────────────────
# Main IKOB Execution
# ─────────────────────────────────────────────

def main(argv=None):

    args = parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    )

    # Resolved data locations first: a CWD-relative surprise here is
    # exactly the class of problem this line exists to make visible.
    logger.info("Cache root: %s", args.cache_root.resolve())
    logger.info("Output root: %s", args.out.resolve())

    # ─────────────────────────────────────────────
    # Load container (SEGS + raw skim matrices)
    # ─────────────────────────────────────────────

    logger.info("Loading IKOB %s (%s)", args.ikob, args.scenario)

    container = get_container(
        ikob_id=args.ikob,
        scenario=args.scenario,
        skim_root=args.skim_root,
        omnummertabellen_root=args.omnummer_root,
        segs_root=args.segs_base,
        cache_root=args.cache_root,
        force_rebuild=args.force_rebuild,
    )

    n = container.n_zones
    logger.info("Zones in %s: %d", args.ikob, n)

    income_labels, population_by_income, jobs_by_income = \
        resolve_income_labels(container)

    # ─────────────────────────────────────────────
    # Generalized cost (betas applied here, ~10 ms)
    # ─────────────────────────────────────────────

    generalized_cost = container.generalized_cost(
        beta_time=args.beta_time,
        beta_distance=args.beta_distance,
    )

    logger.debug(
        "Generalized cost: NaN=%d min=%.2f max=%.2f",
        np.isnan(generalized_cost).sum(),
        generalized_cost.min(),
        generalized_cost.max(),
    )

    population_total = sum(population_by_income.values())
    opportunities_total = sum(jobs_by_income.values())

    # ─────────────────────────────────────────────
    # Filters: reference + per-class resolution + fare
    # ─────────────────────────────────────────────

    # The reference filter: CLI logistic as time marginal, no cost
    # filter, independence. Note the CurveSpec carries (alpha, omega)
    # ONLY — scaling is not part of a probability marginal; it lives
    # at the filter level and is applied after composition.
    reference_filter = ClassFilter(
        time=CurveSpec("logistic", (args.decay_alpha, args.decay_omega)),
        cost=None,
        copula=INDEPENDENCE,
        scaling=args.decay_scaling,
    )

    class_filters = resolve_class_filters(args, income_labels,
                                          reference_filter)
    fare = resolve_fare(args, class_filters)
    cost_matrices, money_id = resolve_cost_matrices(
        container, generalized_cost, class_filters, fare, n)

    # Shared matrix cache for models 1-3 (the segmented Shen runner
    # builds its own copies inside its registry, as before).
    cache = FilterMatrixCache(
        cost_matrices,
        money_id=money_id,
        epsilon=args.decay_epsilon,
        check_frechet=args.check_frechet,
    )

    # ModelState carries the REFERENCE decay fields, which may differ
    # from the effective filter under a uniform cost config. Today
    # that is harmless — both un-segmented models receive their
    # decay_matrix injected and state's decay fields only feed
    # validate() — but any future code path that rebuilds a matrix
    # FROM state would agree with the reference, not the effective
    # filter. If validate() ever recomputes matrices, revisit this.
    state = ModelState.create(
        generalized_cost=generalized_cost,
        population=population_total,
        opportunities=opportunities_total,
        decay_type="logistic",
        decay_beta=(args.decay_alpha, args.decay_omega,
                    args.decay_scaling),
        decay_epsilon=args.decay_epsilon,
    )

    # Effective filter for the un-segmented models: the uniform
    # config filter when one exists (plain Hansen/Shen then see cost
    # filters and fares), the CLI reference otherwise. Composed ONCE
    # through the composer path and reused by Hansen, Shen and any
    # class whose filter equals it.
    effective_filter = resolve_effective_filter(class_filters,
                                                reference_filter)
    D_eff = cache.composed(effective_filter)

    # ─────────────────────────────────────────────
    # 1. Hansen (Naive Gravity, effective filter)
    # ─────────────────────────────────────────────

    logger.info("Computing Hansen accessibility...")
    hansen = compute_naive_accessibility(state, decay_matrix=D_eff)

    # ─────────────────────────────────────────────
    # 2. Segmented Hansen (per-class jobs and filters, no competition)
    # ─────────────────────────────────────────────

    logger.info("Computing segmented Hansen accessibility "
                "(%d income classes, %d distinct filters)...",
                len(income_labels), len(set(class_filters.values())))
    hansen_by_class = compute_segmented_hansen(
        state, income_labels, class_filters, jobs_by_income, cache,
    )
    # Linearity: per-class columns must sum to shared-filter Hansen
    # (uniform filter only). The Hansen analogue of the nesting test.
    check_hansen_additivity(hansen_by_class, class_filters, cache,
                            opportunities_total)

    # ─────────────────────────────────────────────
    # 3. Shen (Single Population, effective filter)
    # ─────────────────────────────────────────────

    logger.info("Computing Shen accessibility...")
    shen = compute_accessibility(state, decay_matrix=D_eff)

    log_balance("shen", shen, population_total, opportunities_total)

    # ─────────────────────────────────────────────
    # 4. Segmented Shen (one competition pool per income class)
    # ─────────────────────────────────────────────

    seg_runner = SegmentedRunner(
        registry=MatrixRegistry(),
        decay_epsilon=args.decay_epsilon,
        pin_decay=args.pin_decay,
        pin_marginals=args.pin_marginals,
        check_frechet=args.check_frechet,
    )

    if args.nesting_test:
        run_nesting_test(seg_runner, state, income_labels,
                         effective_filter, population_by_income,
                         cost_matrices, money_id,
                         opportunities_total, shen)

    logger.info("Computing segmented Shen accessibility "
                "(%d income pools)...", len(income_labels))

    seg_result = seg_runner.run(
        state,
        build_segments(income_labels, class_filters, money_id,
                       pooled=True),
        population_by_income,
        cost_matrices=cost_matrices,
        opportunities=dict(jobs_by_income),
    )

    # Per-pool balance identities: each income class's weighted mean
    # accessibility equals its own job/worker ratio. The spread across
    # pools is the substantive result of segmenting at all. These
    # identities hold for ANY filter (decay-invariant), which is why
    # they survive per-class filters unchanged — and why they cannot
    # replace the nesting test.
    for label in income_labels:
        log_balance(label, seg_result.per_segment[label],
                    population_by_income[label], jobs_by_income[label])
    log_balance("seg total", np.nan_to_num(seg_result.total),
                population_total, opportunities_total)

    # Zero-resident zones: runner's weighted total is 0/0 -> 0 there.
    # Export NaN instead — 0 reads as "no access" and silently drags
    # down any downstream mean; NaN forces explicit handling. The
    # per-class columns stay un-masked: a segment's accessibility at
    # an empty zone is still a well-defined property of the location.
    # (No Hansen analogue: Hansen never divides by population.)
    empty = seg_result.total_population <= 0
    seg_total_out = seg_result.total.copy()
    seg_total_out[empty] = np.nan
    n_empty = int(empty.sum())
    if n_empty:
        logger.info("%d zero-resident zones exported as NaN in "
                    "segmented total.", n_empty)

    # ─────────────────────────────────────────────
    # Output
    # ─────────────────────────────────────────────

    write_outputs(args.out / args.ikob, n, income_labels, hansen,
                  hansen_by_class, shen, seg_result, seg_total_out)

    logger.info("Finished IKOB %s", args.ikob)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())