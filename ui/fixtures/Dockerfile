# This Dockerfile is used to load fixtures into a (separate) ClickHouse server
# We need the AWS CLI available to download our large fixtures from s3
FROM clickhouse/clickhouse-server:24.12

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/
RUN uv python install
RUN uvx parquet-tools
