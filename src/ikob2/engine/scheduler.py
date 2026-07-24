"""
Process-parallel execution of RunSpecs.

The base ModelState is shipped to each worker ONCE via the pool
initializer instead of being pickled into every task (the state
contains the dense N×N cost matrix; per-task pickling is O(runs × N²)
bytes). Failures are collected per run_id, not re-raised eagerly, so
one bad Monte Carlo draw cannot discard an otherwise complete batch.
"""

import logging
from concurrent.futures import ProcessPoolExecutor

from ikob2.engine.runner import SimulationRunner

logger = logging.getLogger(__name__)

_WORKER_STATE = None


def _init_worker(state):
    global _WORKER_STATE
    _WORKER_STATE = state


def _run_one(variants):
    return SimulationRunner().run(_WORKER_STATE, list(variants))


class Scheduler:
    def __init__(self, max_workers: int | None = None):
        self.max_workers = max_workers

    def run_all(self, state, run_specs):
        results, failures = {}, {}
        with ProcessPoolExecutor(
            self.max_workers, initializer=_init_worker, initargs=(state,)
        ) as pool:
            futures = {
                spec.run_id: pool.submit(_run_one, spec.variants)
                for spec in run_specs
            }
            for run_id, future in futures.items():
                try:
                    results[run_id] = future.result()
                except Exception:
                    logger.exception("Run %s failed.", run_id)
                    failures[run_id] = None
        if failures:
            logger.error("%d of %d runs failed: %s",
                         len(failures), len(futures), sorted(failures))
        return results