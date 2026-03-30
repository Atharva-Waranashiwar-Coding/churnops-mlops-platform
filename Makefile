.PHONY: airflow-init airflow-up install-dev lint platform-down platform-train platform-up serve test train

CONFIG ?= configs/base.yaml
DATA_PATH ?=

install-dev:
	python -m pip install -e ".[dev]"

lint:
	python -m ruff check src tests airflow/dags

train:
	PYTHONPATH=src python -m churnops.pipeline.train --config $(CONFIG) $(if $(DATA_PATH),--data-path $(DATA_PATH))

serve:
	PYTHONPATH=src python -m churnops.api.app --config $(CONFIG)

platform-up:
	docker compose up --build inference-api

platform-train:
	docker compose run --rm --profile ops trainer

airflow-init:
	docker compose --profile ops up airflow-db airflow-init

airflow-up:
	docker compose --profile ops up --build airflow-db airflow-scheduler airflow-webserver

platform-down:
	docker compose down --remove-orphans

test:
	PYTHONPATH=src pytest -q
