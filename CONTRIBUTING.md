# Contributing

`README.md` is the primary guide for setup, running, and testing. This document only covers the contribution workflow and repository expectations.

## Development Workflow

1. Start from `develop`.
2. Create a focused branch for the change.
3. Keep business logic separate from orchestration, API route glue, and deployment assets.
4. Add or update tests with the code change.
5. Run `make verify` before finalizing the branch.

## Repository Expectations

- extend the existing training, inference, tracking, drift, and orchestration layers instead of bypassing them
- keep configuration externalized instead of hardcoding environment-specific values
- preserve the separation between business logic and infrastructure concerns
- keep Docker, Kubernetes, and Airflow assets aligned with the Python runtime contract
- avoid placeholder examples and tutorial-only shortcuts

## Testing Expectations

When touching runtime-critical behavior, prioritize:

- API behavior and error handling
- training pipeline persistence and tracking outputs
- config loading and environment overrides
- drift monitoring and retraining triggers
- deployment asset renderability where practical

## Documentation Expectations

- keep `README.md` accurate for setup, running, and testing
- update `docs/architecture.md` when system boundaries or runtime flow change
- update release-facing docs when the external story of the repo changes materially
