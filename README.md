# ChurnOps

ChurnOps is a production-style MLOps project for customer churn prediction. Phase 06 adds GitHub Actions-based quality gates so linting, tests, package build checks, and Docker build validation run automatically against the current local-platform architecture.

## Phase 06 Scope

- keep training, inference, tracking, and containerization intact from earlier phases
- add automated GitHub Actions checks for linting, tests, and package build validation
- validate the inference Docker image in CI without pushing artifacts
- keep CI feedback fast enough for normal pull request work
- document the relationship between local commands and CI expectations

## Repository Layout

```text
.
├── .github/
│   └── workflows/            # GitHub Actions quality gates
├── artifacts/                # local training outputs (gitignored)
├── configs/
│   └── base.yaml             # default training configuration
├── docker/
│   └── entrypoint.sh         # runtime bootstrap for container services
├── data/
│   ├── processed/            # reserved for later phases
│   └── raw/                  # local input dataset location (gitignored)
├── docker-compose.yml        # local platform stack for API, training, and MLflow UI
├── Dockerfile                # production-minded inference service image
├── src/
│   └── churnops/
│       ├── artifacts/        # artifact persistence logic
│       ├── config/           # settings models, loading, and runtime overrides
│       ├── data/             # dataset ingestion and validation
│       ├── features/         # preprocessing and dataset splitting
│       ├── models/           # estimator training and metrics
│       ├── pipeline/         # runner orchestration and CLI entrypoint
│       ├── inference/        # model loading and prediction service layer
│       ├── api/              # FastAPI app bootstrap, routes, and schemas
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

## Inference API

The inference service loads the trained sklearn pipeline behind a dedicated service layer. By default it serves the most recent local artifact from `artifacts/training/`, but the same API can be pointed at an MLflow model URI or an MLflow registered model alias/version through the `inference:` config section.

Endpoints:

- `GET /health`: liveness and readiness-style status for the API and model loader
- `GET /v1/model/metadata`: loaded model source, feature contract, and class-label metadata
- `POST /v1/predictions`: batch churn prediction endpoint

Run the API locally:

```bash
make serve
```

Equivalent module entrypoint:

```bash
PYTHONPATH=src python -m churnops.api.app --config configs/base.yaml
```

The default service binding comes from the config:

- `inference.host`: API bind host
- `inference.port`: API bind port
- `inference.model_source`: `local_artifact`, `mlflow_model_uri`, or `mlflow_registry`
- `inference.prediction_threshold`: threshold used to mark `predicted_churn`
- `inference.preload_model`: whether to load the model at startup

Example prediction request based on the shipped churn fixture:

```json
{
  "instances": [
    {
      "gender": "Female",
      "SeniorCitizen": 0,
      "Partner": "Yes",
      "Dependents": "No",
      "tenure": 1,
      "PhoneService": "No",
      "MultipleLines": "No phone service",
      "InternetService": "DSL",
      "OnlineSecurity": "Yes",
      "OnlineBackup": "No",
      "DeviceProtection": "No",
      "TechSupport": "Yes",
      "StreamingTV": "No",
      "StreamingMovies": "No",
      "Contract": "Month-to-month",
      "PaperlessBilling": "Yes",
      "PaymentMethod": "Electronic check",
      "MonthlyCharges": 29.85,
      "TotalCharges": ""
    }
  ]
}
```

## Containerized Local Platform

The repository now includes a Docker image for the inference service and a `docker-compose.yml` stack for local platform workflows. Runtime behavior is driven through `CHURNOPS_*` environment variables so the same image can support local development, CI smoke tests, and later deployment targets.

Copy the sample environment file before using Docker:

```bash
cp .env.example .env
```

Build and run the local inference service:

```bash
make platform-up
```

Bootstrap or refresh the local model artifact from Docker:

```bash
make platform-train
```

Stop the local platform:

```bash
make platform-down
```

The compose stack includes:

- `inference-api`: long-running FastAPI service for prediction traffic
- `trainer`: one-shot training service under the `ops` profile
- `mlflow`: MLflow UI under the `ops` profile for experiment inspection

Useful environment variables:

- `CHURNOPS_CONFIG`: config file path inside the container
- `CHURNOPS_DATA_PATH`: dataset path override
- `CHURNOPS_INFERENCE_MODEL_SOURCE`: `local_artifact`, `mlflow_model_uri`, or `mlflow_registry`
- `CHURNOPS_INFERENCE_LOCAL_RUN_ID` / `CHURNOPS_INFERENCE_LOCAL_MODEL_PATH`: select the local model artifact
- `CHURNOPS_INFERENCE_MODEL_URI`: explicit MLflow model URI
- `CHURNOPS_INFERENCE_REGISTERED_MODEL_NAME`, `CHURNOPS_INFERENCE_REGISTERED_MODEL_ALIAS`, `CHURNOPS_INFERENCE_REGISTERED_MODEL_VERSION`: MLflow registry model selection
- `CHURNOPS_TRACKING_URI`, `CHURNOPS_REGISTRY_URI`, `CHURNOPS_TRACKING_ARTIFACT_LOCATION`: shared tracking backend configuration
- `MLFLOW_UI_PORT`: host port for the MLflow UI service

The image uses a non-root runtime user, an explicit healthcheck, and a small entrypoint bootstrap so empty bind mounts still produce the required local runtime directories.

## Continuous Integration

GitHub Actions now enforces the main quality gates for pushes to `develop`, pushes to `phase/**` branches, and pull requests:

- `.github/workflows/ci.yml`: package build validation plus separate lint and test jobs
- `.github/workflows/docker-build.yml`: inference image build validation for Docker-relevant changes

The CI design mirrors local developer commands instead of introducing a separate automation-only path:

- `make lint`: Ruff import/style/static checks across `src/` and `tests/`
- `make test`: pytest suite against the local fixture-backed workflow
- `python -m build --no-isolation`: package build validation
- `docker build --file Dockerfile --target runtime --tag churnops/inference-api:ci .`: image build validation

The Python workflow uses pip caching, timeout limits, and concurrency cancellation to reduce wasted CI time. The Docker workflow is path-scoped so documentation-only or unrelated config changes do not pay for a full image build.

## Running Quality Checks

```bash
make lint
make test
```

## Design Notes

- `src/` package layout keeps the codebase ready for packaging, CI, Docker, and service integration.
- configuration loading, settings models, and runtime overrides are centralized under `churnops.config`.
- the pipeline runner is orchestration-only; domain logic stays in dedicated validation, preprocessing, training, and evaluation modules.
- experiment tracking is isolated under `churnops.tracking`, so the rest of the codebase stays MLflow-agnostic.
- the inference API is thin by design; model loading and prediction execution live under `churnops.inference`.
- environment-based runtime overrides keep the same image usable across local Docker workflows, CI, and future deployment targets.
- the feature contract is explicit by default, which prevents accidental training on unexpected or leakage-prone columns.
- the persisted model artifact is a full sklearn pipeline, which keeps future FastAPI inference integration straightforward.
- the MLflow registry flow is metric-driven and can be repointed to a remote backend without changing pipeline orchestration.
