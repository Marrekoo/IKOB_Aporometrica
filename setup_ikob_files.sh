#!/usr/bin/env bash

set -e

PROJECT_NAME="IKOB_Aporometrica"

echo "Creating project structure for $PROJECT_NAME..."

# Root files
touch pyproject.toml
touch README.md
touch LICENSE
touch .gitignore

# -------------------------
# docs
# -------------------------
mkdir -p docs/examples
touch docs/architecture.md
touch docs/model_theory.md
touch docs/data_specification.md

# -------------------------
# examples
# -------------------------
mkdir -p examples/minimal_project/data
mkdir -p examples/advanced_project
touch examples/minimal_project/project.yaml
touch examples/minimal_project/runs.yaml

# -------------------------
# tests
# -------------------------
mkdir -p tests
touch tests/test_travel_time.py
touch tests/test_weights.py
touch tests/test_accessibility.py
touch tests/test_variants.py
touch tests/test_runner.py

# -------------------------
# src structure
# -------------------------
mkdir -p src/ikob2
touch src/ikob2/__init__.py

# -------------------------
# 1️⃣ Domain Layer
# -------------------------
mkdir -p src/ikob2/domain
touch src/ikob2/domain/__init__.py
touch src/ikob2/domain/project.py
touch src/ikob2/domain/dataset.py
touch src/ikob2/domain/zones.py
touch src/ikob2/domain/mappings.py
touch src/ikob2/domain/variant.py
touch src/ikob2/domain/run_config.py
touch src/ikob2/domain/state.py

# -------------------------
# 2️⃣ Data Layer
# -------------------------
mkdir -p src/ikob2/data
touch src/ikob2/data/__init__.py
touch src/ikob2/data/loader.py
touch src/ikob2/data/validator.py
touch src/ikob2/data/schema.py
touch src/ikob2/data/column_mapper.py
touch src/ikob2/data/geopackage.py
touch src/ikob2/data/errors.py

# -------------------------
# 3️⃣ Core Model
# -------------------------
mkdir -p src/ikob2/core
touch src/ikob2/core/__init__.py
touch src/ikob2/core/travel_time.py
touch src/ikob2/core/decay_curves.py
touch src/ikob2/core/weights.py
touch src/ikob2/core/accessibility.py
touch src/ikob2/core/competition.py
touch src/ikob2/core/math_utils.py

# -------------------------
# 4️⃣ Variants
# -------------------------
mkdir -p src/ikob2/variants
touch src/ikob2/variants/__init__.py
touch src/ikob2/variants/base.py
touch src/ikob2/variants/cost_variants.py
touch src/ikob2/variants/decay_variants.py
touch src/ikob2/variants/zone_variants.py
touch src/ikob2/variants/stochastic.py

# -------------------------
# 5️⃣ Engine
# -------------------------
mkdir -p src/ikob2/engine
touch src/ikob2/engine/__init__.py
touch src/ikob2/engine/runner.py
touch src/ikob2/engine/scheduler.py
touch src/ikob2/engine/experiment_expander.py
touch src/ikob2/engine/progress.py
touch src/ikob2/engine/cache.py
touch src/ikob2/engine/logging.py

# -------------------------
# 6️⃣ Outputs
# -------------------------
mkdir -p src/ikob2/outputs
touch src/ikob2/outputs/__init__.py
touch src/ikob2/outputs/writer.py
touch src/ikob2/outputs/tidy_export.py
touch src/ikob2/outputs/metadata.py
touch src/ikob2/outputs/geopackage_export.py
touch src/ikob2/outputs/r_export_helpers.py

# -------------------------
# 7️⃣ CLI
# -------------------------
mkdir -p src/ikob2/cli
touch src/ikob2/cli/__init__.py
touch src/ikob2/cli/run.py
touch src/ikob2/cli/validate.py
touch src/ikob2/cli/inspect.py
touch src/ikob2/cli/create_project.py

# -------------------------
# 8️⃣ Utilities
# -------------------------
mkdir -p src/ikob2/utils
touch src/ikob2/utils/__init__.py
touch src/ikob2/utils/config_loader.py
touch src/ikob2/utils/typing.py
touch src/ikob2/utils/random.py
touch src/ikob2/utils/paths.py

echo "✅ Project structure created successfully."
echo ""
echo "Next steps:"
echo "  cd $PROJECT_NAME"
echo "  git init"
echo "  python -m venv .venv"
echo "  source .venv/bin/activate"
echo "  pip install -e ."
