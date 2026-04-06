#!/bin/sh
set -eu

mkdir -p /app/artifacts /app/data/raw /app/data/processed

if [ -n "${CHURNOPS_TRACKING_URI:-}" ]; then
  case "${CHURNOPS_TRACKING_URI}" in
    sqlite:///*)
      tracking_path="${CHURNOPS_TRACKING_URI#sqlite:///}"
      mkdir -p "$(dirname "${tracking_path}")"
      ;;
  esac
fi

if [ -n "${CHURNOPS_TRACKING_ARTIFACT_LOCATION:-}" ]; then
  case "${CHURNOPS_TRACKING_ARTIFACT_LOCATION}" in
    file://*)
      artifact_path="${CHURNOPS_TRACKING_ARTIFACT_LOCATION#file://}"
      mkdir -p "${artifact_path}"
      ;;
    *://*)
      ;;
    *)
      mkdir -p "${CHURNOPS_TRACKING_ARTIFACT_LOCATION}"
      ;;
  esac
fi

exec "$@"
