# Changelog

All notable changes to ChurnOps are documented here.

## [0.1.0] - 2026-04-06

Initial portfolio release covering the full local MLOps lifecycle:

- modular churn-model training pipeline with validation, preprocessing, evaluation, and artifact persistence
- MLflow experiment tracking and model-registry integration
- FastAPI inference API with strong schemas, model metadata, and readiness/liveness probes
- Docker Compose local platform with MLflow, Prometheus, Grafana, and Airflow
- Airflow-based orchestrated retraining flow
- drift detection with persisted baselines, PSI-based evaluation, event logging, and Airflow-triggered retraining
- Kubernetes deployment assets for staged and production-style inference deployment
- expanded docs, onboarding guidance, and release-readiness automation
