"""Pipeline orchestration interfaces for ChurnOps."""

from churnops.pipeline.runner import TrainingPipelineResult, run_local_training

__all__ = ["TrainingPipelineResult", "run_local_training"]
