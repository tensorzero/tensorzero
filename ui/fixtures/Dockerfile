# This Dockerfile is used to load fixtures into (separate) ClickHouse and Postgres servers
FROM debian:trixie-slim

# Install system dependencies and clickhouse-client
RUN apt-get update && apt-get install -y --no-install-recommends \
    apt-transport-https \
    ca-certificates \
    curl \
    gnupg \
    postgresql-client \
    unzip \
    && curl -fsSL 'https://packages.clickhouse.com/rpm/lts/repodata/repomd.xml.key' \
       | gpg --dearmor -o /usr/share/keyrings/clickhouse-keyring.gpg \
    && ARCH=$(dpkg --print-architecture) \
    && echo "deb [signed-by=/usr/share/keyrings/clickhouse-keyring.gpg arch=${ARCH}] https://packages.clickhouse.com/deb lts main" \
       > /etc/apt/sources.list.d/clickhouse.list \
    && apt-get update \
    && apt-get install -y --no-install-recommends clickhouse-client \
    && rm -rf /var/lib/apt/lists/*

# Install AWS CLI v2
RUN ARCH=$(uname -m) && \
    if [ "$ARCH" = "x86_64" ]; then \
      AWSCLI_URL="https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip"; \
    elif [ "$ARCH" = "aarch64" ]; then \
      AWSCLI_URL="https://awscli.amazonaws.com/awscli-exe-linux-aarch64.zip"; \
    else \
      echo "Unsupported architecture: $ARCH" && exit 1; \
    fi && \
    curl "$AWSCLI_URL" -o "awscliv2.zip" && \
    unzip awscliv2.zip && \
    ./aws/install && \
    rm -rf awscliv2.zip aws

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Copy Python project files and install dependencies
WORKDIR /fixtures

COPY ui/fixtures/pyproject.toml ui/fixtures/uv.lock ui/fixtures/.python-version ./

RUN uv sync --frozen
