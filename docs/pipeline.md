# From data to accessibility: the pipeline

The commands in the order they are run, with what each step reads and
writes. `<root>` is the data folder (`--data-root` or `$IKOB_DATA_ROOT`),
laid out by `ikob2.utils.paths.DataLayout` (`data_specification.md`).

| Step | Command | Output |
|---|---|---|
| 1 | `cli.layout create` | folder structure |
| 2 | `cli.segments fetch` | StatLine snapshots |
| 3 | `cli.segments jobs` | jobs per LISA sector per buurt |
| 4 | `cli.segments car-availability`, `pt-spend` | car availability, PT fare spending by decile |
| 5 | `cli.skims build`, `calibrate-detour`, `build-distance` | car, bike, walk times; car distances |
| 6 | `cli.skims make-peak` + `build` | peak-load car times (optional) |
| 7 | `cli.skims build-pt` | PT times, kilometres and boardings; bicycle-leg modes |
| 8 | `cli.accessibility` | `outputs/runs/<run>/` |
| 9 | `cli.compare`, `cli.interchange`, `cli.paper_tables` | `outputs/comparisons/` |

The segment populations are computed on the fly by `cli.accessibility`;
`cli.segments run` writes them to
`intermediate/segments/nl_segments.gpkg` for inspection only.

## 1. Data folder

    python -m ikob2.cli.layout --root "<root>" create
    python -m ikob2.cli.layout --root "<root>" link <file elsewhere> <inputs-subfolder>

`create` makes the folders and a README and copies the reference files of
the repository's `data/` folder where they are missing: the reference
budgets (`inputs/envelope/`), the time margins (`inputs/survey/`), the tariff
tables (`inputs/tariffs/`), the StatLine snapshots (`cache/statline/`) and
the detour calibration (`intermediate/calibration/`). It never overwrites,
moves or deletes a file, so it can be rerun; `--seed-from DIR` copies from
another folder and `--no-seed` only makes the folders. From then on the
runs read these copies, and the data folder is what to archive with a set
of results. `link` adds a symlink under `inputs/` so sources can stay where
they are.

## 2. StatLine snapshots

    python -m ikob2.cli.segments --data-root <root> fetch

The only network step for the segments. It stores CSV snapshots of CBS
86161NED, 71487ned, 81431ned, 85318NED, 85718NED and 82072NED in
`<root>/cache/statline` (or `--out`). `layout create` already copies the
snapshots of the repository's `data/statline/` there, so this step is only
needed for other periods (`--income-period`, `--wage-period`,
`--wfh-period`, `--kwb-table`, or the matching parameters).

`cli.segments` takes its values from the parameters (`[segments]`,
`[jobs]`, `[ownership]`, `accessibility.*_year` and `*_period`; `--params`,
`--set` or the flags) and its input and output paths from the data folder;
every path can also be given explicitly.

## 3. Jobs per sector per buurt

    python -m ikob2.cli.segments --data-root <root> jobs

reads `inputs/kwb/wijkenbuurten_<kwb year>_v3.gpkg`, `inputs/lisa/` (the
LISA file, `paths.lisa`), `inputs/legacy_ikob/` (the IKOB job table
`paths.ikob_jobs` and the education file `paths.education_jobs`) and the
KWB establishment snapshot, and writes
`intermediate/jobs/sector_jobs_<year>.csv`. It imputes LISA municipal jobs
per sector onto buurten (`data_lineage.md`); `jobs.establishment_weight`
(0.25, `--establishment-weight`) sets the weight of the establishment shares
in the buurt totals.

## 4. Availability and fare spending (ODiN)

    python -m ikob2.cli.segments --data-root <root> car-availability
    python -m ikob2.cli.segments --data-root <root> pt-spend

Both read `inputs/odin/ODIN_22_23_clean.csv` (`paths.odin`) and write to
`intermediate/ownership/` (`car_availability_utrecht.csv`,
`pt_spend_utrecht.csv`). The study municipality and the shrinkage priors are
`[ownership]` parameters (344 = Utrecht; `--municipality`, `--prior`). The
car table feeds `--ownership`; the spending table feeds the public cost of PT
fare concessions (`scenarios.md`). Bicycle ownership is an input file
(`inputs/veh_owners/bike_ownership_buurten.csv`).

## Reference budgets (optional)

    python -m ikob2.cli.envelope --data-root <root> aggregates
    python -m ikob2.cli.envelope --data-root <root> build

re-derive `reference_budgets.csv` from its source tables and ODiN 2023
(linked as `inputs/odin/ODIN_23.csv`); the result equals the shipped table
(`envelope/README.md`). The runs read the table from
`inputs/envelope/reference_budgets.csv`.

## 5. Car, bicycle and walking skims

    python -m ikob2.cli.skims build --kwb <gpkg> --study GM0344 \
        --osm <root>/inputs/osm/netherlands-260822.osm.pbf \
        --modes car bike walk --out <root>/intermediate/skims/utrecht_nl \
        --max-memory 11G
    python -m ikob2.cli.skims calibrate-detour --kwb <gpkg> --study GM0344 \
        --out <root>/intermediate/calibration/car_detour.json
    python -m ikob2.cli.servers valhalla build
    python -m ikob2.cli.servers valhalla start
    python -m ikob2.cli.skims build-distance <root>/intermediate/skims/utrecht_nl \
        --kwb <gpkg> --detour <root>/intermediate/calibration/car_detour.json
    python -m ikob2.cli.servers valhalla stop

R5 (via r5py, Java 21) routes car and bicycle; walking comes from zone
geometry. Car distances come from Valhalla within 30 km and from the
calibrated detour model beyond (`skims.md`, `servers.md`). A calibrated
detour model (`data/calibration/car_detour.json`) is copied to
`intermediate/calibration/` by `layout create`.

## 6. Peak load (optional)

    python -m ikob2.cli.skims make-peak --osm <free-flow.pbf> \
        --out <root>/intermediate/osm_peak/netherlands.peak.osm.pbf
    python -m ikob2.cli.skims build --kwb <gpkg> --study GM0344 \
        --osm <peak.pbf> --modes car --out <root>/intermediate/skims/utrecht_nl_peak

Run with `--study utrecht_nl_peak --distance-study utrecht_nl` to use the
free-flow store's distances with peak times.

## 7. Public transport

    python -m ikob2.cli.skims build-pt <root>/intermediate/skims/utrecht_nl \
        --kwb <gpkg> --gtfs <root>/inputs/gtfs/gtfs-nl.zip

stores the plain mode `pt` (`time`, `rail_km`, `other_km`,
`other_boardings`). The shared-bicycle variants need the bicycle-leg modes:

    STORE=<root>/intermediate/skims/utrecht_nl
    COMMON="--kwb <gpkg> --gtfs <zip> --data-root <root>"
    python -m ikob2.cli.skims build-pt $STORE $COMMON --access bike --mode-name pt_bw
    for kind in lime ovfiets; do
      python -m ikob2.cli.skims build-pt $STORE $COMMON --egress bike \
          --egress-hubs file --hub-kind $kind --mode-name pt_wb_$kind
      python -m ikob2.cli.skims build-pt $STORE $COMMON --access bike \
          --egress bike --egress-hubs file --hub-kind $kind --mode-name pt_bb_$kind
    done

(`skims.md`: frequency model, fares, bicycle legs, hub files.)

## 8. Accessibility

    python -m ikob2.cli.accessibility --data-root <root> \
        --study utrecht_nl --run s0 --modes car bike pt \
        --ownership --shared-bike v0 v1 v2 v3

With `--data-root` the paths are filled from the layout: the KWB file of
`--kwb-year`, the store `intermediate/skims/<study>`, the sector jobs of
`--jobs-year`, the StatLine snapshots, the detour model, car availability,
bicycle ownership and the output folder `outputs/runs/<run>`. Each can also
be given explicitly (`--kwb`, `--skims`, `--sector-jobs`, `--out`, ...).
The reference files named in the parameters (`paths.budgets`,
`paths.survey_margins`, `paths.lime_price_scales`, `paths.pt_fare_scales`)
are used as given when that path exists and are otherwise looked up in their
folder of the data layout. `run.json` records the path and SHA-256 of every
input file (`input_files`), so an archived data folder can be matched to a
run.

What a run does:

1. computes the 44 segments per buurt from KWB and the StatLine snapshots and
   keeps the 40 with a budget (`--censored atom|drop` for decile 1);
2. matches the sector jobs to income deciles (`sector_income_weights`) and
   splits them by job type (`split_jobs_by_wfh`), or gives every segment all
   jobs (`--common-jobs`);
3. loads the Weibull margins and the reference budgets;
4. builds the mode matrices: car time plus parking search time and car cost
   (`--car-model`, `--no-parking-search`), bicycle time, PT time and fare
   (`--pt-*` flags), and the shared-bicycle option sets (`--shared-bike`);
5. evaluates the specification (`--spec`, `--copula`, `--theta`,
   `--no-cost-gate`, `--time-shape`) for every mode, segment and origin;
6. applies availability (`--ownership`), price scales by segment, and the
   scenario settings (`--lime-scale`, `--dockless-model`, `--pt-fare-scales`,
   `--report-usage`);
7. writes the outputs.

Outputs in `outputs/runs/<run>/`: `accessibility.csv` (one row per origin x
mode x segment), `summary_income.csv`, `summary_household.csv` (and
`*_expected.csv` with `--ownership`), `run.json`, and unless `--set
accessibility.export=false` the analysis products (`segments.csv`,
`time_margins.csv`, `hubs.csv`, `origins.gpkg`, `money_gate_*.csv`,
`accessibility.parquet` when pyarrow is installed, `dictionary.csv`, and a
README with an R snippet). Columns: `data_specification.md`.

### Car time and cost

`skims.car.car_time_and_cost` returns time and money separately:

* **time**: drive time plus parking search time, arrival search at the
  destination by KWB urbanisation class (12 / 8 / 4 / 0 / 0 minutes for
  classes 1-5) plus a departure search at the origin of a quarter of that;
* **money**: variable cost per km x distance, plus optional road charge and
  per-zone parking cost (supported, no data wired in). Car models in
  `[car.models]`: `fossil` 0.16 EUR/km (default), `electric` 0.05,
  `shared` 0.33 EUR/km + 0.05 EUR/min, `taxi` 2.40 EUR/km + 0.40 EUR/min.

Distances are the store's routed `distance`; without it, the crow-fly
distance times the detour model (`--detour`, or the constant 1.3).

## 9. Comparing runs

    python -m ikob2.cli.compare --data-root <root> run_a run_b [--mode pt_v2]

writes to `outputs/comparisons/<a>__vs__<b>/`: Spearman correlations over
origin x segment cells and over origin means, the top-10% overlap, and level
ratios by income class and mode. When both runs were made with
`--report-usage`, it also writes who gains against who bears the cost
(`effectiveness_income_class.csv`, `effectiveness_household_type.csv`).

`cli.interchange` and `cli.paper_tables` are described in `scenarios.md`.
