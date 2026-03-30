.PHONY: install-dev train test

CONFIG ?= configs/base.yaml
DATA_PATH ?=

install-dev:
	pip install -e ".[dev]"

train:
	PYTHONPATH=src python -m churnops.pipeline.train --config $(CONFIG) $(if $(DATA_PATH),--data-path $(DATA_PATH))

test:
	PYTHONPATH=src pytest
