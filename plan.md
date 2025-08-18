Publishing Plan: Enterprise-Grade ML Repo (Fraud Detection)
0) Goal & Scope

Publish a clean, well-engineered ML project that’s reproducible, typed, tested, and CI-validated. Not productionized; focused on professional standards and clarity.

1) Prerequisites

GitHub account + local Git.

Python 3.11 or 3.12.

Poetry for env/deps: pipx install poetry (or pip install poetry).

(Optional) Kaggle API for data fetch.

2) Create Repository & Metadata

Create a private repo named fraud-detection-ml.

Add License (MIT or Apache-2.0).

Set repo description + topics: machine-learning, fraud-detection, streamlit, sklearn.

Enable branch protection on main (require PR + checks).

3) Target Structure
fraud-detection-ml/
├─ src/fraud_detection/
│  ├─ __init__.py
│  ├─ data.py            # load/split/validate schema
│  ├─ features.py        # transforms/encoders
│  ├─ model.py           # train/evaluate/save
│  ├─ predict.py         # batch/CLI inference
│  └─ utils.py           # logging, config, seeding
├─ app/streamlit_app.py  # UI (imports src/*)
├─ configs/
│  ├─ train.yaml
│  ├─ infer.yaml
│  └─ app.yaml
├─ notebooks/EDA.ipynb   # outputs stripped via pre-commit
├─ data/sample.csv       # tiny sample for tests & smoke
├─ artifacts/            # local models/metrics (git-ignored)
├─ scripts/
│  ├─ download_data.py   # pull dataset subset or full via Kaggle
│  └─ run_all.sh         # lint → tests → train → eval → (optional) app
├─ tests/
│  ├─ test_schema.py
│  ├─ test_features.py
│  ├─ test_model_smoke.py
│  └─ test_cli_predict.py
├─ docs/
│  ├─ MODEL_CARD.md
│  └─ DATA_CARD.md
├─ .github/workflows/ci.yml
├─ .gitignore
├─ .pre-commit-config.yaml
├─ pyproject.toml
├─ README.md
├─ CHANGELOG.md
├─ CONTRIBUTING.md
└─ CODE_OF_CONDUCT.md

File migration (from your current assets)

fraud_detection.py → move UI parts to app/streamlit_app.py; move logic into src/fraud_detection/ modules.

analysis_model.ipynb → notebooks/EDA.ipynb (strip outputs).

AIML Dataset.csv → do not commit; place a small data/sample.csv.

fraud_detection_pipeline.pkl → do not commit; keep under artifacts/ locally or attach to a Release.

4) Initialize Project
git init
poetry init -n
poetry add pandas numpy scikit-learn pydantic pyyaml streamlit
poetry add -D black ruff mypy pytest pytest-cov nbstripout

5) pyproject.toml (tooling & CLI)

Add the following sections to pyproject.toml:

[tool.black]
line-length = 100
target-version = ["py311"]

[tool.ruff]
line-length = 100
select = ["E","F","I"]  # I = import sorting
fix = true

[tool.mypy]
python_version = "3.11"
ignore_missing_imports = true
strict = false

[tool.poetry.scripts]
fd-train = "src.fraud_detection.model:main"
fd-predict = "src.fraud_detection.predict:main"
fd-app = "app.streamlit_app:main"


main functions should exist in the referenced modules.

6) Pre-commit & Hygiene

.pre-commit-config.yaml

repos:
  - repo: https://github.com/psf/black
    rev: 24.4.2
    hooks: [{id: black}]
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.5.7
    hooks:
      - id: ruff
        args: [--fix]
  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.10.0
    hooks: [{id: mypy}]
  - repo: https://github.com/kynan/nbstripout
    rev: 0.6.1
    hooks: [{id: nbstripout}]


Enable:

poetry run pre-commit install


.gitignore (key entries)

artifacts/
data/*.csv
!data/sample.csv
*.pkl
.env
.venv
.ipynb_checkpoints/

7) Minimal Configs

configs/train.yaml

random_seed: 42
data:
  train_path: data/sample.csv
  target: is_fraud
model:
  type: sklearn.LogisticRegression
  params: { max_iter: 1000 }
cv:
  folds: 3
metrics: [roc_auc, f1, precision, recall]


configs/infer.yaml

input_path: data/sample.csv
output_path: artifacts/preds.csv
model_path: artifacts/model.pkl


configs/app.yaml

title: "Fraud Detection Demo"
sample_path: data/sample.csv

8) Scripts

scripts/run_all.sh

#!/usr/bin/env bash
set -euo pipefail
poetry run ruff check .
poetry run black --check .
poetry run mypy .
poetry run pytest -q --cov=src --cov-report=term-missing
poetry run fd-train --config configs/train.yaml
poetry run fd-predict --config configs/infer.yaml
# Optional: poetry run streamlit run app/streamlit_app.py


chmod +x scripts/run_all.sh

scripts/download_data.py (requirements)

Support: direct HTTP or kaggle datasets download.

Write to data/ and validate schema (columns, dtypes).

Never store creds in code; read KAGGLE_USERNAME/KAGGLE_KEY.

9) Testing (what to cover)

test_schema.py: validate columns, non-null constraints, basic stats.

test_features.py: transformers are pure & shape-stable.

test_model_smoke.py: tiny train on sample.csv runs end-to-end and yields metrics.

test_cli_predict.py: run fd-predict on sample.csv and assert preds.csv exists & schema ok.

Run:

poetry run pytest -q --cov=src --cov-report=term-missing

10) CI (GitHub Actions)

.github/workflows/ci.yml

name: ci
on:
  push: { branches: [main] }
  pull_request: { branches: [main] }
jobs:
  build:
    runs-on: ubuntu-latest
    strategy:
      matrix: { python-version: ["3.11", "3.12"] }
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: ${{ matrix.python-version }} }
      - name: Install Poetry
        run: pipx install poetry
      - name: Install deps
        run: poetry install --no-interaction
      - name: Lint
        run: |
          poetry run ruff check .
          poetry run black --check .
      - name: Type check
        run: poetry run mypy .
      - name: Tests
        run: poetry run pytest -q --cov=src --cov-report=xml

11) Documentation

README.md (outline)

One-line problem statement + 2–3 line overview.

Data: link/source, license, how to fetch; clarify sample.csv.

Approach: features, model choice, CV, metrics.

Results: small table (ROC-AUC/F1/etc.) + limitations.

Quickstart:

poetry install
poetry run python scripts/download_data.py   # optional
poetry run fd-train --config configs/train.yaml
poetry run fd-predict --config configs/infer.yaml
poetry run streamlit run app/streamlit_app.py


Repo map, Model/Data Cards links, License.

MODEL_CARD.md: intended use, data, metrics, caveats, ethical notes.
DATA_CARD.md: fields, splits, source link, license, known biases.

12) Data & Model Assets Policy

Do not commit large CSVs or .pkl.

Keep local outputs in artifacts/ (ignored).

If sharing a trained model, attach it to a GitHub Release (e.g., v0.1.0) and document how to download.

13) Badges & Repo Polish

Add shields to README: Python version, CI status, license.

Add a GIF/screenshot of the Streamlit app.

Add .github/ISSUE_TEMPLATE/ (bug/feature) and PULL_REQUEST_TEMPLATE.md.

Adopt Conventional Commits; maintain CHANGELOG.md.

14) First Release

Update README with final metrics & screenshots.

Tag: git tag v0.1.0 && git push --tags.

Create GitHub Release; attach model (optional) + release notes.

15) Resume Snippet (ready to paste)

Built an end-to-end fraud-detection ML pipeline (data validation → feature engineering → training/evaluation → CLI & Streamlit app) with typed, tested Python and CI (lint/type/test/coverage). Published model + data cards and reproducible configs; achieved [ROC-AUC X.XX] on hold-out data.

16) Go-Live Checklist

 Clean tree matches target structure

 Pre-commit hooks installed & passing

 poetry install succeeds; scripts/run_all.sh passes locally

 CI green on PR

 README + cards complete; screenshots added

 Large files excluded; optional model attached to Release

 v0.1.0 published