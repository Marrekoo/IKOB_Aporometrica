"""
The data folder layout. The root is never built in: it comes from --data-root,
$IKOB_DATA_ROOT or `paths.data_root` (ikob2.params.data_root).

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
        veh_owners/              vehicle ownership per buurt (bicycle share)
      cache/
        statline/                CBS StatLine snapshots
      intermediate/              derived, reproducible from inputs
        segments/                44 household-type x income segments per buurt
        jobs/                    imputed sector jobs per buurt
        ownership/               car availability per segment (from ODiN)
        skims/<study>/           skim stores
        calibration/             detour model and other calibrations
        valhalla/                local routing server: config, tiles, logs
        otp/                     OpenTripPlanner: jar, graph inputs, logs
        osm_peak/                OSM extracts with peak-load speeds
      outputs/
        runs/<run>/              one folder per accessibility run
        comparisons/             run-versus-run comparison tables

Nothing under `inputs/` is ever written by the code. `intermediate/` and
`outputs/` can be deleted and rebuilt.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ikob2.params import DEFAULTS

INPUT_DIRS = ("kwb", "lisa", "osm", "gtfs", "legacy_ikob", "survey", "odin")
CACHE_DIRS = ("statline",)
INTERMEDIATE_DIRS = ("segments", "jobs", "skims", "calibration", "valhalla", "otp", "osm_peak")
OUTPUT_DIRS = ("runs", "comparisons")

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
        valhalla/        local Valhalla routing server (config, tiles, logs)
        otp/             local OpenTripPlanner (jar, graph, logs)
        osm_peak/        OSM extracts with peak-load speeds (skims.peak)
    outputs/runs/<run>/  accessibility results of one run
                         (python -m ikob2.cli.accessibility)
    outputs/comparisons/ run-versus-run comparisons (cli.compare)

Files that were already in the root of this folder (the GeoPackages and
S_T_work.csv) were left where they are; `inputs/` links to them.
Rules: the code never writes under inputs/; intermediate/ and outputs/
can be deleted and rebuilt from inputs/ and the repository.
"""


@dataclass(frozen=True)
class DataLayout:
    root: Path

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
    def kwb(self, year: int = DEFAULTS.accessibility.kwb_year,
            version: str = DEFAULTS.paths.kwb_version) -> Path:
        return self.inputs / "kwb" / f"wijkenbuurten_{year}_{version}.gpkg"

    def skim_dir(self, study: str) -> Path:
        return self.intermediate / "skims" / study

    def bike_ownership(self) -> Path:
        return self.inputs / "veh_owners" / DEFAULTS.paths.bike_ownership

    def pt_spend(self, study: str = DEFAULTS.paths.car_availability_study) -> Path:
        return self.intermediate / "ownership" / f"pt_spend_{study}.csv"

    def car_availability(
            self, study: str = DEFAULTS.paths.car_availability_study) -> Path:
        return self.intermediate / "ownership" / f"car_availability_{study}.csv"

    def sector_jobs(self, year: int = DEFAULTS.accessibility.jobs_year) -> Path:
        return self.intermediate / "jobs" / f"sector_jobs_{year}.csv"

    def valhalla_dir(self) -> Path:
        return self.intermediate / "valhalla"

    def osm_peak_dir(self) -> Path:
        return self.intermediate / "osm_peak"

    def otp_dir(self) -> Path:
        return self.intermediate / "otp"

    def detour_model(self) -> Path:
        return self.intermediate / "calibration" / "car_detour.json"

    def run_dir(self, run: str) -> Path:
        return self.outputs / "runs" / run

    def comparison_dir(self) -> Path:
        return self.outputs / "comparisons"

    def statline(self) -> Path:
        return self.cache / "statline"
