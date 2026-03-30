.PHONY: install-dev train test

install-dev:
	pip install -e ".[dev]"

train:
	PYTHONPATH=src python -m churnops.pipeline.train --config configs/base.yaml

test:
	PYTHONPATH=src pytest
