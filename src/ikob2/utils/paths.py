"""
The data folder layout (default: /home/marco/IKOB data).

    <root>/
      README.md                  what lives where (written by `create`)
      inputs/                    source data, read only
        kwb/                     CBS wijken en buurten GeoPackages (per year)
        lisa/                    LISA jobs per municipality and sector
        osm/                     OpenStreetMap extracts (.pbf)
        gtfs/                    GTFS feeds
        legacy_ikob/             legacy IKOB SEGS files (jobs by income, education)
        survey/                  fitted time margins (S_T_*.csv)
        odin/                    ODiN tables and codebook
      cache/
        statline/                CBS StatLine snapshots
      intermediate/              derived, reproducible from inputs
        segments/                44 household-type x income segments per buurt
        jobs/                    imputed sector jobs per buurt
        skims/<study>/           skim stores
        calibration/             detour model and other calibrations
      outputs/
        runs/<run>/              one folder per accessibility run

Nothing under `inputs/` is ever written by the code. `intermediate/` and
`outputs/` can be deleted and rebuilt.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

DEFAULT_ROOT = Path("/home/marco/IKOB data")

INPUT_DIRS = ("kwb", "lisa", "osm", "gtfs", "legacy_ikob", "survey", "odin")
CACHE_DIRS = ("statline",)
INTERMEDIATE_DIRS = ("segments", "jobs", "skims", "calibration")
OUTPUT_DIRS = ("runs",)

README = """# IKOB data

Data folder of the IKOB Aporometrica project (code: IKOB_Aporometrica).

    inputs/          source data, read only (kwb, lisa, osm, gtfs,
                     legacy_ikob, survey, odin)
    cache/statline/  CBS StatLine snapshots (python -m ikob2.cli.segments fetch)
    intermediate/    derived, rebuildable
        segments/    44 household-type x income segments per buurt
        jobs/        imputed LISA sector jobs per buurt
        skims/<study>/   skim stores (python -m ikob2.cli.skims build)
        calibration/     detour model for car distances
    outputs/runs/<run>/  accessibility results of one run
                         (python -m ikob2.cli.accessibility)

Files that were already in the root of this folder (the GeoPackages and
S_T_work.csv) were left where they are; `inputs/` links to them.
Rules: the code never writes under inputs/; intermediate/ and outputs/
can be deleted and rebuilt from inputs/ and the repository.
"""


@dataclass(frozen=True)
class DataLayout:
    root: Path = DEFAULT_ROOT

    def __post_init__(self):
        object.__setattr__(self, "root", Path(self.root))

    # ── folders ──────────────────────────────────────────────────────
    @property
    def inputs(self) -> Path:
        return self.root / "inputs"

    @property
    def cache(self) -> Path:
        return self.root / "cache"

    @property
    def intermediate(self) -> Path:
        return self.root / "intermediate"

    @property
    def outputs(self) -> Path:
        return self.root / "outputs"

    def all_dirs(self) -> list[Path]:
        return ([self.inputs / d for d in INPUT_DIRS]
                + [self.cache / d for d in CACHE_DIRS]
                + [self.intermediate / d for d in INTERMEDIATE_DIRS]
                + [self.outputs / d for d in OUTPUT_DIRS])

    def ensure(self) -> list[Path]:
        """Create every folder (idempotent) and the README if missing.
        Returns the folders that were created."""
        created = []
        for d in self.all_dirs():
            if not d.exists():
                d.mkdir(parents=True)
                created.append(d)
        readme = self.root / "README.md"
        if not readme.exists():
            readme.write_text(README)
        return created

    # ── conventional paths ───────────────────────────────────────────
    def kwb(self, year: int = 2022, version: str = "v3") -> Path:
        return self.inputs / "kwb" / f"wijkenbuurten_{year}_{version}.gpkg"

    def skim_dir(self, study: str) -> Path:
        return self.intermediate / "skims" / study

    def sector_jobs(self, year: int = 2022) -> Path:
        return self.intermediate / "jobs" / f"sector_jobs_{year}.csv"

    def detour_model(self) -> Path:
        return self.intermediate / "calibration" / "car_detour.json"

    def run_dir(self, run: str) -> Path:
        return self.outputs / "runs" / run

    def statline(self) -> Path:
        return self.cache / "statline"
