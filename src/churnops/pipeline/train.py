"""CLI entrypoint for the local churn baseline training workflow."""

from __future__ import annotations

import argparse
from dataclasses import replace
import logging
from pathlib import Path

from churnops.artifacts.persistence import PersistedRun, persist_training_run
from churnops.config import Settings, load_settings
from churnops.data.ingestion import load_raw_dataset
from churnops.features.preprocessing import prepare_training_dataset, split_dataset
from churnops.models.training import TrainingResult, train_baseline_model

LOGGER = logging.getLogger(__name__)


def run_training(
    config_path: str | Path,
    data_path: str | Path | None = None,
) -> tuple[TrainingResult, PersistedRun]:
    """Execute the end-to-end local baseline training workflow."""

    settings = load_settings(config_path)
    settings = apply_runtime_overrides(settings, data_path=data_path)
    raw_dataset = load_raw_dataset(settings.data)
    prepared_dataset = prepare_training_dataset(raw_dataset, settings.data)
    data_splits = split_dataset(prepared_dataset.features, prepared_dataset.target, settings.split)
    training_result = train_baseline_model(
        data_splits=data_splits,
        feature_spec=prepared_dataset.feature_spec,
        config=settings.model,
    )
    persisted_run = persist_training_run(
        settings=settings,
        model_pipeline=training_result.model_pipeline,
        metrics=training_result.metrics,
        split_sizes=training_result.split_sizes,
        feature_spec=training_result.feature_spec,
        source_row_count=int(raw_dataset.shape[0]),
    )
    return training_result, persisted_run


def apply_runtime_overrides(
    settings: Settings,
    data_path: str | Path | None = None,
) -> Settings:
    """Apply CLI-level runtime overrides without mutating the loaded config object."""

    if data_path is None:
        return settings

    resolved_data_path = Path(data_path).expanduser()
    if not resolved_data_path.is_absolute():
        resolved_data_path = (settings.project.root_dir / resolved_data_path).resolve()

    return replace(
        settings,
        data=replace(
            settings.data,
            raw_data_path=resolved_data_path,
        ),
    )


def build_argument_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for local training."""

    parser = argparse.ArgumentParser(description="Train the ChurnOps baseline churn model.")
    parser.add_argument(
        "--config",
        default="configs/base.yaml",
        help="Path to the YAML configuration file.",
    )
    parser.add_argument(
        "--data-path",
        help="Optional CSV path that overrides data.raw_data_path from the config.",
    )
    return parser


def configure_logging() -> None:
    """Configure application logging for CLI execution."""

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def main() -> int:
    """Run the training CLI."""

    parser = build_argument_parser()
    args = parser.parse_args()

    configure_logging()

    try:
        training_result, persisted_run = run_training(args.config, data_path=args.data_path)
    except FileNotFoundError as error:
        LOGGER.error(
            "%s. Place a churn CSV at that path or rerun with --data-path /path/to/customer_churn.csv.",
            error,
        )
        return 1
    except Exception:
        LOGGER.exception("Training failed.")
        return 1

    test_metrics = training_result.metrics["test"]
    LOGGER.info("Training completed successfully.")
    LOGGER.info("Artifact directory: %s", persisted_run.run_directory)
    LOGGER.info(
        "Test metrics | accuracy=%.4f precision=%.4f recall=%.4f f1=%.4f roc_auc=%s",
        test_metrics["accuracy"],
        test_metrics["precision"],
        test_metrics["recall"],
        test_metrics["f1"],
        f"{test_metrics['roc_auc']:.4f}" if test_metrics["roc_auc"] is not None else "n/a",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
