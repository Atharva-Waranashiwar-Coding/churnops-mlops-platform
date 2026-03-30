# ChurnOps

ChurnOps is a production-style MLOps project for customer churn prediction. Phase 01 establishes the local baseline: configuration-driven data ingestion, preprocessing, model training, evaluation, and artifact persistence without relying on notebooks as the main implementation path.

## Phase 01 Scope

- bootstrap a maintainable Python project structure
- ingest a local churn dataset with schema checks
- preprocess mixed numeric and categorical features with sklearn pipelines
- train a baseline logistic regression classifier
- evaluate train, validation, and test performance
- persist the trained pipeline and run metadata to a structured artifacts directory

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
│       ├── data/             # dataset ingestion
│       ├── features/         # preprocessing and dataset splitting
│       ├── models/           # estimator training and metrics
│       ├── pipeline/         # CLI entrypoints
│       └── config.py         # typed settings loader
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

If your dataset lives elsewhere or uses different split ratios or model settings, copy `configs/base.yaml` and update the path and parameters there.

## Local Training

Place the dataset at `data/raw/customer_churn.csv`, then run:

```bash
make train
```

Equivalent direct command:

```bash
PYTHONPATH=src python -m churnops.pipeline.train --config configs/base.yaml
```

If the dataset lives somewhere else on your machine, override the configured path at runtime:

```bash
make train DATA_PATH=/absolute/path/to/customer_churn.csv
```

On success, the pipeline writes a timestamped run directory under `artifacts/training/` containing:

- `model.joblib`: serialized sklearn pipeline with preprocessing and classifier
- `metrics.json`: train, validation, and test metrics
- `metadata.json`: dataset, feature, split, and model metadata
- `config.yaml`: snapshot of the resolved training config

## Running Tests

```bash
make test
```

## Design Notes

- `src/` package layout keeps the codebase ready for packaging, CI, Docker, and service integration.
- configuration is separated from implementation so later phases can extend the training workflow without rewriting module boundaries.
- the feature contract is explicit by default, which prevents accidental training on unexpected or leakage-prone columns.
- the persisted model artifact is a full sklearn pipeline, which keeps future FastAPI inference integration straightforward.
