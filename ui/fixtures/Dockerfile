# This Dockerfile is used to load fixtures into a (separate) ClickHouse server
FROM clickhouse:lts

RUN apt-get update && apt-get install -y cmake
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Copy Python project files and install dependencies
WORKDIR /fixtures
COPY ui/fixtures/pyproject.toml ui/fixtures/uv.lock ui/fixtures/.python-version ./
RUN uv sync --frozen
