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
COPY configs ./configs

RUN chown -R churnops:churnops /app /home/churnops

USER churnops

EXPOSE 8000

CMD ["churnops-serve"]
