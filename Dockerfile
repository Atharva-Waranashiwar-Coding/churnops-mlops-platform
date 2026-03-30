# syntax=docker/dockerfile:1.7

FROM python:3.11-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    VIRTUAL_ENV=/opt/venv \
    PATH="/opt/venv/bin:${PATH}"

WORKDIR /build

RUN python -m venv "${VIRTUAL_ENV}" \
    && pip install --upgrade pip setuptools wheel

COPY pyproject.toml README.md ./
COPY src ./src

RUN pip install --no-cache-dir .


FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    VIRTUAL_ENV=/opt/venv \
    PATH="/opt/venv/bin:${PATH}" \
    CHURNOPS_CONFIG=/app/configs/base.yaml

RUN groupadd --system churnops \
    && useradd --system --gid churnops --create-home --home-dir /home/churnops churnops

WORKDIR /app

COPY --from=builder /opt/venv /opt/venv
COPY --chmod=755 docker/entrypoint.sh /usr/local/bin/docker-entrypoint.sh
COPY configs ./configs

RUN mkdir -p /app/artifacts /app/data/raw /app/data/processed \
    && chown -R churnops:churnops /app /home/churnops

USER churnops

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD python -c "import os, urllib.request; port=os.environ.get('CHURNOPS_INFERENCE_PORT', '8000'); urllib.request.urlopen(f'http://127.0.0.1:{port}/health', timeout=3)"

ENTRYPOINT ["docker-entrypoint.sh"]
CMD ["churnops-serve"]
