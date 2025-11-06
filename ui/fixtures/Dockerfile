# This Dockerfile is used to load fixtures into a (separate) ClickHouse server
FROM clickhouse:24.12-alpine

RUN apk add --no-cache python3
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Copy Python project files and install dependencies
WORKDIR /fixtures
COPY ui/fixtures/pyproject.toml ui/fixtures/uv.lock ./
RUN uv sync --frozen
