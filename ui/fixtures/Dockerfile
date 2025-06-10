# This Dockerfile is used to load fixtures into a (separate) ClickHouse server
FROM clickhouse/clickhouse-server:24.12-alpine

RUN apk add --no-cache python3
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/
