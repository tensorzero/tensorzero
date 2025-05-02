# This Dockerfile is used to load fixtures into a (separate) ClickHouse server
# We need the AWS CLI available to download our large fixtures from s3
FROM clickhouse/clickhouse-server:24.12-alpine

RUN apk add --no-cache python3 py3-pip aws-cli
RUN pip install parquet-tools --break-system-packages