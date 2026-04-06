.DEFAULT_GOAL := help

.PHONY: airflow-init airflow-up help install-dev k8s-render k8s-render-production k8s-render-staging \
	package platform-down platform-train platform-up serve test train train-fixture verify

CONFIG ?= configs/base.yaml
FIXTURE_CONFIG ?= configs/fixture.yaml
DATA_PATH ?=
K8S_OVERLAY ?= deploy/kubernetes/overlays/staging
FIXTURE_DATA ?= tests/fixtures/customer_churn.csv

help: ## Show the common local development commands
	@grep -E '^[a-zA-Z0-9_.-]+:.*## ' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*## "}; {printf "%-24s %s\n", $$1, $$2}'

install-dev: ## Install the project with development dependencies
	python -m pip install -e ".[dev]"

lint: ## Run the Ruff lint checks
	python -m ruff check src tests airflow/dags

k8s-render: ## Render the selected Kubernetes overlay with kubectl kustomize
	kubectl kustomize $(K8S_OVERLAY)

k8s-render-staging: ## Render the staging Kubernetes overlay
	$(MAKE) k8s-render K8S_OVERLAY=deploy/kubernetes/overlays/staging

k8s-render-production: ## Render the production Kubernetes overlay
	$(MAKE) k8s-render K8S_OVERLAY=deploy/kubernetes/overlays/production

package: ## Build the sdist and wheel artifacts
	python -m build --no-isolation

train: ## Run the local training workflow
	PYTHONPATH=src python -m churnops.pipeline.train --config $(CONFIG) $(if $(DATA_PATH),--data-path $(DATA_PATH))

train-fixture: ## Train against the checked-in churn fixture dataset
	PYTHONPATH=src python -m churnops.pipeline.train --config $(FIXTURE_CONFIG) --data-path $(FIXTURE_DATA)

serve: ## Start the FastAPI inference service locally
	PYTHONPATH=src python -m churnops.api.app --config $(CONFIG)

platform-up: ## Start the local inference, Prometheus, and Grafana stack
	docker compose up --build inference-api prometheus grafana

platform-train: ## Run the trainer container once against the configured dataset
	docker compose run --rm --profile ops trainer

airflow-init: ## Bootstrap the local Airflow metadata database and admin account
	docker compose --profile ops up airflow-db airflow-init

airflow-up: ## Start the local Airflow scheduler and webserver
	docker compose --profile ops up --build airflow-db airflow-scheduler airflow-webserver

platform-down: ## Stop the local Docker platform stack
	docker compose down --remove-orphans

test: ## Run the pytest suite
	PYTHONPATH=src pytest -q

verify: ## Run the main local quality gates and manifest validation steps
	$(MAKE) lint
	$(MAKE) test
	$(MAKE) package
	$(MAKE) k8s-render-staging
	$(MAKE) k8s-render-production
