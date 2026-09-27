"""
The data folder layout. The root is never built in: it comes from --data-root,
$IKOB_DATA_ROOT or `paths.data_root` (ikob2.params.data_root).

    <root>/
      README.md                  what lives where (written by `create`)
      inputs/                    source data, read only
        kwb/                     CBS wijken en buurten GeoPackages (per year)
        lisa/                    LISA jobs per municipality and sector
        legacy_ikob/             IKOB job table per buurt; LISA 2016 jobs by education
        osm/                     OpenStreetMap extracts (.pbf)
        gtfs/                    GTFS feeds
        survey/                  fitted time margins (S_T_*.csv)
        envelope/                reference budgets per segment
        tariffs/                 price and fare multipliers per segment
        odin/                    ODiN tables and codebook
        veh_owners/              vehicle ownership per buurt (bicycle share)
        hubs/                    shared-bicycle hubs (municipal hubs)
        ovfiets/                 OV-fiets locations
      cache/
        statline/                CBS StatLine snapshots
      intermediate/              derived, reproducible from inputs
        segments/                44 household-type x income segments per buurt
        jobs/                    imputed sector jobs per buurt
        ownership/               car availability and PT fare spending (ODiN)
        hubs/                    hubs made by the model (scenario S2)
        skims/<study>/           skim stores
        calibration/             detour model and other calibrations
        valhalla/                local routing server: config, tiles, logs
        otp/                     OpenTripPlanner: jar, graph inputs, logs
        osm_peak/                OSM extracts with peak-load speeds
        osm_walk/                OSM extracts reduced to the walking network
      outputs/
        runs/<run>/              one folder per accessibility run
        comparisons/             run-versus-run comparison tables

Nothing under `inputs/` is ever written by the code, except that
`seed` (`cli.layout create`) adds missing reference files from the
repository's `data/` folder; it never overwrites a file. `intermediate/`
and `outputs/` can be deleted and rebuilt.

A reference file named in the parameters (`paths.budgets`, ...) is used as
given when that path exists, and otherwise looked up in its folder of the
layout (`resolve_input`).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ikob2.params import DEFAULTS

INPUT_DIRS = ("kwb", "lisa", "legacy_ikob", "osm", "gtfs", "survey",
              "envelope", "tariffs", "odin", "veh_owners", "hubs", "ovfiets")
CACHE_DIRS = ("statline",)
INTERMEDIATE_DIRS = ("segments", "jobs", "ownership", "hubs", "skims",
                     "calibration", "valhalla", "otp", "osm_peak", "osm_walk")
OUTPUT_DIRS = ("runs", "comparisons")

README = """# IKOB data

Data folder of the IKOB Aporometrica project (code: IKOB_Aporometrica).

    inputs/          source data, read only (kwb, lisa, legacy_ikob, osm,
                     gtfs, survey, envelope, tariffs, odin, veh_owners,
                     hubs, ovfiets)
    cache/statline/  CBS StatLine snapshots (python -m ikob2.cli.segments fetch)
    intermediate/    derived, rebuildable
        segments/    44 household-type x income segments per buurt
        jobs/        imputed LISA sector jobs per buurt (cli.segments jobs)
        ownership/   car availability, PT fare spending (cli.segments)
        hubs/        extra hubs of scenario S2 (cli.hubs propose)
        skims/<study>/   skim stores (python -m ikob2.cli.skims build)
        calibration/     detour model for car distances
        valhalla/        local Valhalla routing server (config, tiles, logs)
        otp/             local OpenTripPlanner (jar, graph, logs)
        osm_peak/        OSM extracts with peak-load speeds (skims.peak)
        osm_walk/        OSM extracts reduced to the walking network
    outputs/runs/<run>/  accessibility results of one run
                         (python -m ikob2.cli.accessibility)
    outputs/comparisons/ run-versus-run comparisons (cli.compare,
                         cli.interchange, cli.paper_tables)

`cli.layout create` copies the reference files shipped with the
repository (budgets, time margins, tariff tables, StatLine snapshots, detour
calibration) into this folder when they are missing, and never overwrites
one. From then on the files here are the ones the runs use.

Rules: apart from that seeding the code never writes under inputs/;
intermediate/ and outputs/ can be deleted and rebuilt from inputs/.
"""

# repository data/ subfolder -> (layout area, subfolder, file pattern): the
# reference files `DataLayout.seed` copies into a data folder
SEED = (
    ("envelope", "inputs", "envelope", "*.csv"),
    ("margins", "inputs", "survey", "*.csv"),
    ("tariffs", "inputs", "tariffs", "*.csv"),
    ("statline", "cache", "statline", "*.csv"),
    ("calibration", "intermediate", "calibration", "*.json"),
)
REPO_DATA = Path(__file__).resolve().parents[3] / "data"


def resolve_input(value: str | Path, folder: Path | None) -> Path:
    """A reference file of the parameters: `value` itself if that path
    exists (absolute or relative to the current folder), else `folder /
    value` (the file's folder in the data layout)."""
    p = Path(value).expanduser()
    if p.exists() or p.is_absolute() or folder is None:
        return p
    return Path(folder) / p


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

    def seed(self, source: str | Path = REPO_DATA) -> list[Path]:
        """Copy the reference files of `source` (the repository's data/
        folder) that are missing here; existing files are never touched.
        Returns the files copied."""
        import shutil

        source = Path(source)
        if not source.is_dir():
            raise FileNotFoundError(f"No reference data folder {source}.")
        copied = []
        for sub, area, dest, pattern in SEED:
            for f in sorted((source / sub).glob(pattern)):
                target = self.root / area / dest / f.name
                if target.exists() or target.is_symlink():
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(f, target)
                copied.append(target)
        return copied

    # ── conventional paths ───────────────────────────────────────────
    def kwb(self, year: int = DEFAULTS.accessibility.kwb_year,
            version: str = DEFAULTS.paths.kwb_version) -> Path:
        return self.inputs / "kwb" / f"wijkenbuurten_{year}_{version}.gpkg"

    def budgets(self, name: str = DEFAULTS.paths.budgets) -> Path:
        return resolve_input(name, self.inputs / "envelope")

    def survey_margins(self, name: str = DEFAULTS.paths.survey_margins) -> Path:
        return resolve_input(name, self.inputs / "survey")

    def tariff(self, name: str) -> Path:
        return resolve_input(name, self.inputs / "tariffs")

    def lisa(self) -> Path:
        return self.inputs / "lisa" / DEFAULTS.paths.lisa

    def ikob_jobs(self) -> Path:
        return self.inputs / "legacy_ikob" / DEFAULTS.paths.ikob_jobs

    def education_jobs(self) -> Path:
        return self.inputs / "legacy_ikob" / DEFAULTS.paths.education_jobs

    def odin(self) -> Path:
        return self.inputs / "odin" / DEFAULTS.paths.odin

    def segments_gpkg(self) -> Path:
        return self.intermediate / "segments" / "nl_segments.gpkg"

    def s2_hubs(self) -> Path:
        return self.intermediate / "hubs" / "utrecht_hubs_s2.csv"

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

    def osm_walk_dir(self) -> Path:
        return self.intermediate / "osm_walk"

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
