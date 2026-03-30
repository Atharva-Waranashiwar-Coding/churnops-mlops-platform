# ChurnOps

ChurnOps is a production-style MLOps project for customer churn prediction. Phase 03 extends the modular baseline with MLflow-backed experiment tracking, structured run logging, and a gated model registry workflow that stays compatible with future deployment automation.

## Phase 03 Scope

- keep the modular training pipeline from Phase 02 intact
- track training runs in MLflow without leaking tracking code into preprocessing or model modules
- log searchable parameters, per-split metrics, model artifacts, and evaluation outputs
- register only the best model version according to a configurable comparison rule
- keep tracking and registry settings externalized so local and remote backends can be swapped later

## Repository Layout

```text
.
├── artifacts/                # local training outputs (gitignored)
├── configs/
│   └── base.yaml             # default training configuration
├── data/
│   ├── processed/            # reserved for later phases
│   └── raw/                  # local input dataset location (gitignored)
├── src/
│   └── churnops/
│       ├── artifacts/        # artifact persistence logic
│       ├── config/           # settings models, loading, and runtime overrides
│       ├── data/             # dataset ingestion and validation
│       ├── features/         # preprocessing and dataset splitting
│       ├── models/           # estimator training and metrics
│       ├── pipeline/         # runner orchestration and CLI entrypoint
│       └── tracking/         # tracker interface and MLflow implementation
└── tests/                    # unit and integration tests
```

## Prerequisites

- Python 3.10+
- a local churn dataset in CSV format

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Dataset Contract

The default configuration expects a telecom churn CSV at `data/raw/customer_churn.csv`. The shipped `configs/base.yaml` is aligned to the common Telco churn export variant with columns such as `CustomerID`, `Tenure Months`, `Monthly Charges`, and `Churn Label`.

At ingestion time, those raw headers are renamed into a stable internal schema such as `customerID`, `tenure`, `MonthlyCharges`, `TotalCharges`, and `Churn`. The target column is `Churn`, the positive class is `Yes`, and `customerID` is excluded from modeling.

If your dataset lives elsewhere or uses different split ratios, model settings, or tracking backends, copy `configs/base.yaml` and update the relevant sections there.

## Local Training

Place the dataset at `data/raw/customer_churn.csv`, then run:

```bash
make train
```

Installed CLI entrypoint:

```bash
churnops-train --config configs/base.yaml
```

Equivalent module entrypoint:

```bash
PYTHONPATH=src python -m churnops.pipeline.train --config configs/base.yaml
```

If the dataset lives somewhere else on your machine, override the configured path at runtime:

```bash
make train DATA_PATH=/absolute/path/to/customer_churn.csv
```

On success, the pipeline writes a timestamped run directory under `artifacts/training/` containing:

- `model/model.joblib`: serialized sklearn pipeline with preprocessing and classifier
- `metrics/metrics.json`: train, validation, and test metrics
- `metadata/run.json`: run metadata, config provenance, and artifact paths
- `metadata/validation.json`: raw dataset validation summary
- `config/training.yaml`: snapshot of the resolved training config

## Experiment Tracking

The default configuration also enables MLflow with a local SQLite-backed metadata store at `artifacts/mlflow/mlflow.db`. Every successful training run opens an MLflow run, logs searchable parameters and per-split metrics, records the full local run bundle as artifacts, and logs an MLflow-native sklearn model artifact for downstream serving workflows.

Tracking settings live under the `tracking:` section in the YAML config. The important knobs are:

- `tracking.enabled`: enable or disable experiment tracking without changing pipeline code
- `tracking.experiment_name`: MLflow experiment name for training runs
- `tracking.tracking_uri`: local SQLite URI or remote MLflow tracking backend URI
- `tracking.registry_uri`: model registry backend URI; defaults to the tracking backend
- `tracking.artifact_location`: where MLflow should write tracked run artifacts
- `tracking.model_registry.*`: model registration policy, including the comparison split and metric

The registry flow is gated. ChurnOps only registers a new model version when the current run beats the incumbent registered version on the configured comparison metric. The default rule uses validation-set `f1`.

To inspect the local MLflow store with the UI:

```bash
mlflow ui --backend-store-uri sqlite:///artifacts/mlflow/mlflow.db
```

## Training Flow

The local training runner executes these stages:

1. read and normalize the raw dataset
2. validate the dataset contract and target availability
3. prepare features and targets, then split train/validation/test data
4. train the configured baseline estimator
5. evaluate each split and persist a structured local training run
6. track the run in MLflow and register the model if it is the current best candidate

## Running Tests

```bash
make test
```

## Design Notes

- `src/` package layout keeps the codebase ready for packaging, CI, Docker, and service integration.
- configuration loading, settings models, and runtime overrides are centralized under `churnops.config`.
- the pipeline runner is orchestration-only; domain logic stays in dedicated validation, preprocessing, training, and evaluation modules.
- experiment tracking is isolated under `churnops.tracking`, so the rest of the codebase stays MLflow-agnostic.
- the feature contract is explicit by default, which prevents accidental training on unexpected or leakage-prone columns.
- the persisted model artifact is a full sklearn pipeline, which keeps future FastAPI inference integration straightforward.
- the MLflow registry flow is metric-driven and can be repointed to a remote backend without changing pipeline orchestration.
