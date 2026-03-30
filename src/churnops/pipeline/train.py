"""CLI entrypoint for the local churn baseline training workflow."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from churnops.config import apply_runtime_overrides, load_settings
from churnops.pipeline.runner import TrainingPipelineResult, run_local_training

LOGGER = logging.getLogger(__name__)


def run_training(
    config_path: str | Path,
    data_path: str | Path | None = None,
) -> TrainingPipelineResult:
    """Execute the end-to-end local baseline training workflow."""

    settings = load_settings(config_path)
    settings = apply_runtime_overrides(settings, data_path=data_path)
    return run_local_training(settings)


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
        pipeline_result = run_training(args.config, data_path=args.data_path)
    except FileNotFoundError as error:
        LOGGER.error(
            "%s. Place a churn CSV at that path or rerun with --data-path /path/to/customer_churn.csv.",
            error,
        )
        return 1
    except Exception:
        LOGGER.exception("Training failed.")
        return 1

    test_metrics = pipeline_result.evaluation_result.metrics["test"]
    LOGGER.info("Training completed successfully.")
    LOGGER.info("Artifact directory: %s", pipeline_result.persisted_run.run_directory)
    if pipeline_result.tracking_result.enabled and pipeline_result.tracking_result.run_id:
        LOGGER.info(
            "MLflow run | experiment=%s run_id=%s",
            pipeline_result.tracking_result.experiment_name,
            pipeline_result.tracking_result.run_id,
        )
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
