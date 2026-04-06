# ChurnOps

## Project Description

ChurnOps is an end-to-end MLOps project for customer churn prediction. It includes model training, experiment tracking with MLflow, a FastAPI inference API, Prometheus and Grafana monitoring, drift detection, Airflow-based retraining orchestration, and Kubernetes deployment assets.

## Purpose

This repository is meant to show a production-minded MLOps workflow in one codebase. It keeps training, serving, monitoring, orchestration, and deployment concerns separate so the system stays easier to operate and extend.

## Documentation

- [Architecture](docs/architecture.md)
- [Contributing](CONTRIBUTING.md)
- [Release Notes](docs/release-notes/v0.1.0.md)
- [Changelog](CHANGELOG.md)

## Prerequisites

- Python 3.10 or newer
- Docker Desktop or another local Docker runtime for container workflows
- a churn CSV if you want to train on your own dataset instead of the checked-in fixture

## Setup

Create a virtual environment and install the project:

```bash
python -m venv .venv
source .venv/bin/activate
make install-dev
```

Optional environment file for Docker and platform workflows:

```bash
cp .env.example .env
```

## Running Locally

Train with the checked-in fixture dataset:

```bash
make train-fixture
```

Train with your own dataset:

```bash
make train DATA_PATH=/absolute/path/to/customer_churn.csv
```

Start the inference API:

```bash
make serve
```

Start the local platform stack with the API, Prometheus, and Grafana:

```bash
make platform-up
```

Start the local Airflow services:

```bash
make airflow-init
make airflow-up
```

Render the Kubernetes manifests locally:

```bash
make k8s-render-staging
make k8s-render-production
```

## Testing And Validation

Run lint checks:

```bash
make lint
```

Run the test suite:

```bash
make test
```

Run the full local verification flow:

```bash
make verify
```

`make verify` runs linting, tests, packaging, and Kubernetes manifest rendering.

## Main Local Workflow

If you want the shortest useful local path:

1. `make install-dev`
2. `make train-fixture`
3. `make serve`
4. `make test`
5. `make verify`
