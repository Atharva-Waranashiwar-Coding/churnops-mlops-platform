# Contributing

ChurnOps is structured as a portfolio-grade MLOps repository, but it is still intended to be easy to run and extend locally. This guide covers the expected development workflow.

## Prerequisites

- Python 3.10 or newer
- Docker Desktop or another local Docker runtime for container-based workflows
- a churn CSV if you want to run training outside the checked-in test fixture

## Local Setup

```bash
python -m venv .venv
source .venv/bin/activate
make install-dev
```

Optional local environment file for Docker and platform workflows:

```bash
cp .env.example .env
```

## Common Commands

Use `make help` to see the main local commands. The most common ones are:

- `make train DATA_PATH=/path/to/customer_churn.csv`
- `make serve`
- `make test`
- `make verify`
- `make platform-up`
- `make airflow-up`
- `make k8s-render-staging`
- `make k8s-render-production`

For quick end-to-end validation without bringing your own dataset:

```bash
make train-fixture
```

## Development Workflow

1. Start from `develop`.
2. Create a focused branch for the change.
3. Keep business logic separate from orchestration, API route glue, and deployment assets.
4. Add or update tests with the code change.
5. Run `make verify` before finalizing the branch.

## Code And Repository Expectations

- prefer extending the existing training, inference, tracking, and drift layers instead of bypassing them
- keep configuration externalized instead of hardcoding environment-specific values
- prefer production-minded defaults and explicit failure handling
- keep Docker, Kubernetes, and Airflow assets aligned with the Python runtime contract
- avoid placeholder examples and tutorial-only shortcuts

## Testing Expectations

The current suite mixes unit and integration-style tests. When touching a runtime-critical path, prioritize:

- API behavior and error handling
- training pipeline persistence and tracking outputs
- config loading and environment overrides
- drift monitoring and retraining triggers
- deployment asset renderability where practical

## Documentation Expectations

Public-facing changes should keep the repository presentation coherent:

- update `README.md` when behavior or workflows change
- update `docs/architecture.md` if component boundaries or runtime flow change
- add release-facing notes when the user-visible story of the repo changes materially
