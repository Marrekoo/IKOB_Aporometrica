"""
Run container: one binary file per (IKOB, scenario) bundling the
dense raw skim matrices and SEGS arrays.

Design principle: the container stores only parameter-free data.
Betas, decay parameters, FARES etc. are applied at load time, so
sensitivity analysis and Monte Carlo never invalidate the cache.
money_cost() is the second instance of this principle after
generalized_cost(): the distance skim is data, the fare model is a
scenario assumption, and only the former lives in the file. This is
also why CONTAINER_VERSION stays at 1 — the money skim needs no new
stored array, dist_matrix has been in every container since v1.

Staleness: a fingerprint (mtime + size) of every source file is
stored inside the container; if any source changed, the container
is rebuilt automatically.
"""

from dataclasses import dataclass
import json
import logging
from pathlib import Path

import numpy as np

from ikob2.core.numerics import DTYPE
from ikob2.data.loaders.skims_loader import (
    IKOB_TO_REGION,
    load_ikob_raw_matrices,
    combine_generalized_cost,
)
from ikob2.data.loaders.segs_loader import INCOME_CLASSES, load_segs
from ikob2.domain.fare import FareModel

logger = logging.getLogger(__name__)

CONTAINER_VERSION = 1


@dataclass
class IkobContainer:
    ikob_id: str
    scenario: str
    lms_ids: np.ndarray          # (n,)   int64
    time_matrix: np.ndarray      # (n, n) float32, minutes
    dist_matrix: np.ndarray      # (n, n) float32, METERS
    population: np.ndarray       # (n, 4) float32, INCOME_CLASSES order
    jobs: np.ndarray             # (n, 4) float32, INCOME_CLASSES order

    @property
    def n_zones(self) -> int:
        return len(self.lms_ids)

    def generalized_cost(
        self,
        beta_time: float = 1.0,
        beta_distance: float = 0.0,
        unreachable_value: float = 9999.0,
    ) -> np.ndarray:
        """Recombine raw matrices with betas — milliseconds."""
        return combine_generalized_cost(
            self.time_matrix, self.dist_matrix,
            beta_time=beta_time,
            beta_distance=beta_distance,
            unreachable_value=unreachable_value,
        )

    def money_cost(self, fare: FareModel) -> np.ndarray:
        """Out-of-pocket cost in EUR: fee + rate * detour * km.

        Computed at call time from the stored freeflow distance skim
        (an affine transform, microseconds — no caching question).
        Freeflow is the RIGHT distance for cost even in a spits run:
        congestion changes travel time, not route length to first
        order, so peak time x freeflow distance is coherent.

        dist_matrix is in METERS; fare rates are per km — the 1e-3
        here is the single place that conversion happens. run.py's
        money-sanity check is the guard on this line: nothing else
        in the pipeline is sensitive to cost LEVELS.

        Intrazonal distance 0 gives cost == fee exactly: right for
        PT (boarding fee applies), acceptably wrong for car.
        """
        if fare.skim_id != "afstand_auto_freeflow":
            raise ValueError(
                f"Container only carries 'afstand_auto_freeflow'; "
                f"fare requests '{fare.skim_id}'."
            )
        per_meter = fare.rate_per_km * fare.detour * 1e-3
        return (fare.fee + per_meter * self.dist_matrix).astype(
            DTYPE, copy=False)

    def population_by_income(self) -> dict[str, np.ndarray]:
        return {inc: self.population[:, i]
                for i, inc in enumerate(INCOME_CLASSES)}

    def jobs_by_income(self) -> dict[str, np.ndarray]:
        return {inc: self.jobs[:, i]
                for i, inc in enumerate(INCOME_CLASSES)}


# ─────────────────────────────────────────────
# Source fingerprinting
# ─────────────────────────────────────────────

def _source_paths(
    ikob_id: str,
    scenario: str,
    skim_root: Path,
    omnummertabellen_root: Path,
    segs_root: Path,
) -> list[Path]:
    region = IKOB_TO_REGION[ikob_id]
    skim_dir = skim_root / region / scenario
    segs_dir = segs_root / ikob_id / "SEGS" / scenario
    return [
        omnummertabellen_root / f"{ikob_id}_Omnummer.csv",
        skim_dir / "reistijd_auto_ochtendspits.csv",
        skim_dir / "afstand_auto_freeflow.csv",
        segs_dir / "Beroepsbevolking_inkomensklasse.csv",
        segs_dir / "Arbeidsplaatsen_inkomensklasse.csv",
    ]


def _fingerprint(paths: list[Path]) -> str:
    fp = {}
    for p in paths:
        st = p.stat()
        fp[p.name] = [st.st_mtime_ns, st.st_size]
    return json.dumps(fp, sort_keys=True)


# ─────────────────────────────────────────────
# Build / save / load
# ─────────────────────────────────────────────

def container_path(cache_root: Path, ikob_id: str, scenario: str) -> Path:
    return cache_root / f"{ikob_id}_{scenario}.npz"


def build_container(
    ikob_id: str,
    scenario: str,
    skim_root: Path,
    omnummertabellen_root: Path,
    segs_root: Path,
    cache_root: Path,
) -> IkobContainer:
    """Parse all CSVs once and write the binary container."""

    logger.info("Building container %s %s ...", ikob_id, scenario)

    lms_ids, time_matrix, dist_matrix = load_ikob_raw_matrices(
        ikob_id=ikob_id,
        scenario=scenario,
        skim_root=skim_root,
        omnummertabellen_root=omnummertabellen_root,
    )

    segs = load_segs(segs_root, ikob_id, scenario)

    if segs.n_zones != len(lms_ids):
        raise ValueError(
            f"{ikob_id} {scenario}: SEGS has {segs.n_zones} zones, "
            f"omnummer has {len(lms_ids)}"
        )

    population = np.column_stack(
        [segs.population_by_income[inc] for inc in INCOME_CLASSES]
    ).astype(DTYPE, copy=False)
    jobs = np.column_stack(
        [segs.jobs_by_income[inc] for inc in INCOME_CLASSES]
    ).astype(DTYPE, copy=False)

    sources = _source_paths(
        ikob_id, scenario, skim_root, omnummertabellen_root, segs_root
    )
    meta = {
        "version": CONTAINER_VERSION,
        "ikob_id": ikob_id,
        "scenario": scenario,
        "income_classes": INCOME_CLASSES,
        "fingerprint": _fingerprint(sources),
    }

    cache_root.mkdir(parents=True, exist_ok=True)
    out_path = container_path(cache_root, ikob_id, scenario)

    # Uncompressed on purpose: ~15 MB, loads in ~0.1 s.
    np.savez(
        out_path,
        meta=np.frombuffer(
            json.dumps(meta).encode("utf-8"), dtype=np.uint8
        ),
        lms_ids=lms_ids,
        time_matrix=time_matrix,
        dist_matrix=dist_matrix,
        population=population,
        jobs=jobs,
    )
    logger.info("Container written: %s", out_path)

    return IkobContainer(
        ikob_id=ikob_id,
        scenario=scenario,
        lms_ids=lms_ids,
        time_matrix=time_matrix,
        dist_matrix=dist_matrix,
        population=population,
        jobs=jobs,
    )


def _load_cached(path: Path, sources: list[Path],
                 ikob_id: str, scenario: str) -> IkobContainer | None:
    """Load the cached container iff it is current; None means
    'rebuild'. Split out of get_container so the freshness logic
    reads as one linear checklist."""
    with np.load(path) as npz:
        meta = json.loads(bytes(npz["meta"]).decode("utf-8"))

        if meta.get("version") != CONTAINER_VERSION:
            logger.info("%s: container version changed, rebuilding",
                        path.name)
            return None
        if meta.get("fingerprint") != _fingerprint(sources):
            logger.info("%s: source files changed, rebuilding",
                        path.name)
            return None

        logger.info("Loaded container %s", path.name)
        return IkobContainer(
            ikob_id=ikob_id,
            scenario=scenario,
            lms_ids=npz["lms_ids"],
            time_matrix=npz["time_matrix"],
            dist_matrix=npz["dist_matrix"],
            population=npz["population"],
            jobs=npz["jobs"],
        )


def get_container(
    ikob_id: str,
    scenario: str,
    skim_root: Path,
    omnummertabellen_root: Path,
    segs_root: Path,
    cache_root: Path,
    force_rebuild: bool = False,
) -> IkobContainer:
    """
    Main entry point: load the container if present and fresh,
    otherwise (re)build it from the CSVs.
    """
    path = container_path(cache_root, ikob_id, scenario)

    if not force_rebuild and path.exists():
        sources = _source_paths(
            ikob_id, scenario, skim_root, omnummertabellen_root,
            segs_root,
        )
        try:
            container = _load_cached(path, sources, ikob_id, scenario)
            if container is not None:
                return container
        except Exception as exc:  # corrupt cache → rebuild, never crash
            logger.warning(
                "%s: failed to load container (%s), rebuilding",
                path.name, exc,
            )

    return build_container(
        ikob_id=ikob_id,
        scenario=scenario,
        skim_root=skim_root,
        omnummertabellen_root=omnummertabellen_root,
        segs_root=segs_root,
        cache_root=cache_root,
    )