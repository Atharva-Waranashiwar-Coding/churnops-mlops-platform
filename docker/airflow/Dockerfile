# syntax=docker/dockerfile:1.7

ARG AIRFLOW_VERSION=2.10.4

FROM apache/airflow:${AIRFLOW_VERSION}-python3.11

ARG AIRFLOW_VERSION

USER root

COPY pyproject.toml README.md /opt/churnops/
COPY src /opt/churnops/src
COPY docker/airflow/init.sh /opt/churnops/docker/airflow/init.sh

RUN chown -R airflow:0 /opt/churnops \
    && chmod +x /opt/churnops/docker/airflow/init.sh

USER airflow

RUN pip install --no-cache-dir \
    --constraint "https://raw.githubusercontent.com/apache/airflow/constraints-${AIRFLOW_VERSION}/constraints-3.11.txt" \
    "psycopg2-binary>=2.9,<3.0" \
    /opt/churnops

ENV PYTHONPATH=/opt/churnops/src
