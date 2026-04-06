"""Airflow DAG entrypoint for scheduled ChurnOps retraining."""

from churnops.orchestration.airflow import build_training_dag

dag = build_training_dag()
