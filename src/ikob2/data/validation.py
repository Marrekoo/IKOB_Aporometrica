"""
Warn-and-collect validation: collect EVERY problem, log it, and let the
caller decide whether to abort.
"""

import logging
from dataclasses import dataclass, field

import numpy as np


logger = logging.getLogger(__name__)


@dataclass
class ValidationReport:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def error(self, msg: str, *args) -> None:
        text = msg % args if args else msg
        self.errors.append(text)
        logger.error(text)

    def warning(self, msg: str, *args) -> None:
        text = msg % args if args else msg
        self.warnings.append(text)
        logger.warning(text)

    @property
    def ok(self) -> bool:
        return not self.errors

    def merge(self, other: "ValidationReport") -> "ValidationReport":
        self.errors.extend(other.errors)
        self.warnings.extend(other.warnings)
        return self

    def raise_if_failed(self) -> None:
        if not self.ok:
            raise ValueError(
                f"Input validation failed with {len(self.errors)} error(s); "
                "see log for details."
            )


def validate_pools(
    segments,
    opportunities: dict,
    *,
    populations: dict | None = None,
    total_opportunities=None,
    rtol: float = 1e-3,
) -> ValidationReport:
    """Consistency checks for explicit competition pools.

    Errors (fatal): a segment references a pool with no opportunities.
    Warnings: unused pools; pool supplies not summing to the total
    (double-counted or dropped jobs); pools with zero population.
    """
    report = ValidationReport()

    segment_pools = {s.pool for s in segments}
    supplied_pools = set(opportunities)

    for pool in sorted(segment_pools - supplied_pools):
        report.error(
            "Segments reference pool '%s' but no opportunities vector "
            "was supplied for it.", pool,
        )
    for pool in sorted(supplied_pools - segment_pools):
        report.warning(
            "Opportunities pool '%s' has no segments — this supply is "
            "unreachable by construction. Config mistake?", pool,
        )

    if total_opportunities is not None and supplied_pools:
        pool_total = float(sum(np.asarray(v).sum()
                               for v in opportunities.values()))
        grand_total = float(np.asarray(total_opportunities).sum())
        if grand_total > 0 and not np.isclose(
            pool_total, grand_total, rtol=rtol
        ):
            report.warning(
                "Pool opportunity totals sum to %.6g but the state total "
                "is %.6g — supply is double-counted or dropped across "
                "pools (deliberate overlap is allowed, but check this "
                "is intended).", pool_total, grand_total,
            )

    if populations is not None:
        pool_pop: dict[str, float] = {p: 0.0 for p in segment_pools}
        for s in segments:
            if s.name in populations:
                pool_pop[s.pool] += float(np.asarray(populations[s.name]).sum())
        for pool, total in sorted(pool_pop.items()):
            if pool in supplied_pools and total == 0.0:
                report.warning(
                    "Pool '%s' has zero total population; its "
                    "accessibility results will be meaningless.", pool,
                )

    return report