.PHONY: install-dev serve train test

CONFIG ?= configs/base.yaml
DATA_PATH ?=

install-dev:
	pip install -e ".[dev]"

train:
	PYTHONPATH=src python -m churnops.pipeline.train --config $(CONFIG) $(if $(DATA_PATH),--data-path $(DATA_PATH))

serve:
	PYTHONPATH=src python -m churnops.api.app --config $(CONFIG)

test:
	PYTHONPATH=src pytest
