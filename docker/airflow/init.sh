#!/bin/sh
set -eu

airflow db migrate

if ! airflow users list | awk 'NR > 2 {print $1}' | grep -qx "${AIRFLOW_ADMIN_USERNAME}"; then
  airflow users create \
    --username "${AIRFLOW_ADMIN_USERNAME}" \
    --password "${AIRFLOW_ADMIN_PASSWORD}" \
    --firstname "${AIRFLOW_ADMIN_FIRSTNAME}" \
    --lastname "${AIRFLOW_ADMIN_LASTNAME}" \
    --role Admin \
    --email "${AIRFLOW_ADMIN_EMAIL}"
fi
