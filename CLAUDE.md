# IKOB Aporometrica: working notes for AI assistants and contributors

Threshold-gate accessibility model (`src/ikob2`). Read `README.md` and
`docs/architecture.md` first; the model is in `docs/model_theory.md`.

## Commands

    pip install -r requirements-lock.txt && pip install -e . --no-deps
    pytest -q                 # ~550 tests, ~25 s; must pass before a commit
    pytest -q --cov           # as CI: fails below 80% coverage
    ruff check src tests examples validation
    pre-commit install --hook-type pre-commit --hook-type pre-push

CI (`.github/workflows/tests.yml`) runs ruff and pytest on the locked
environment on every push. After a dependency change, regenerate
`requirements-lock.txt` (instructions in its header) and keep the ruff version
in `.pre-commit-config.yaml` equal to the locked one.

## Rules for changes

* A change that can move results must leave them unchanged unless that is
  the intent: run `tests/test_example.py` (the synthetic end-to-end run) and,
  for larger changes, rerun a real configuration before and after and compare
  `accessibility.csv` (all rows, exact). Say what was compared.
* Every bug fix comes with a test that fails without it. Tests assert values
  (hand computations, invariants), not only shapes or "no exception".
* Parameters live in `src/ikob2/defaults.toml`; the code holds no numeric
  assumptions of its own. Unknown keys are errors; keep it that way.
* Documentation describes the current version only (no "earlier", "used to",
  "not yet"); update the docs in the same commit as the code.
* The output format (`accessibility.csv` columns, `run.json`,
  `accessibility.parquet`, `origins.gpkg`, `dictionary.csv`) is read by a
  separate R analysis pipeline that runs without AI tools on another machine:
  do not rename or drop columns without saying so.

## Data

* The data folder is outside the repository (`--data-root` or
  `$IKOB_DATA_ROOT`; layout in `src/ikob2/utils/paths.py`). Never commit data
  from it. The code never writes under `inputs/`, except that
  `cli.layout create` adds missing reference files and never overwrites.
* Licences: CBS (KWB, StatLine) and LISA municipal data
  (https://www.lisa.nl/gratis-data/, the 'gemeenten' table) are open;
  OpenStreetMap is ODbL; GTFS is open.
* **ODiN microdata requires a DANS permission.** Never paste, print or send
  rows of the ODiN files (`inputs/odin/`) to an AI tool or into logs, issues,
  commits or documentation. Work with code and aggregate outputs only (the
  tables written by `cli.segments car-availability`, `pt-spend` and
  `cli.envelope aggregates`). The envelope aggregates in
  `data/envelope/odin/` are published with the author's approval
  (`data/envelope/odin/README.md`); publishing other ODiN-derived tables
  needs the author's decision.

## Modelling decisions not to change without asking

* The reference budgets are per home-based tour and are divided by the
  number of one-way journeys per tour, `legs_per_tour` = 2.2 (ODiN 2022-23,
  `envelope/journeys_per_tour.py`); other values only as a sensitivity.
* Results are framed as capability / equity (gain per euro, share of the gap
  closed); no fiscal-return estimates.
* Runs that set bicycle ownership to zero ("nobike") describe accessibility in
  a bicycle-and-PT world; car ownership is a separate axis and is not netted
  out (no independence assumption between car and bicycle ownership).
