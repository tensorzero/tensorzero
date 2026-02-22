# This Dockerfile is used to load fixtures into (separate) ClickHouse and Postgres servers
FROM debian:trixie-slim

# Install system dependencies, clickhouse-client, and s5cmd
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    gnupg \
    postgresql-client \
    && curl -fsSL 'https://packages.clickhouse.com/rpm/lts/repodata/repomd.xml.key' \
       | gpg --dearmor -o /usr/share/keyrings/clickhouse-keyring.gpg \
    && ARCH=$(dpkg --print-architecture) \
    && echo "deb [signed-by=/usr/share/keyrings/clickhouse-keyring.gpg arch=${ARCH}] https://packages.clickhouse.com/deb lts main" \
       > /etc/apt/sources.list.d/clickhouse.list \
    && apt-get update \
    && apt-get install -y --no-install-recommends clickhouse-client \
    && ARCH=$(uname -m) \
    && if [ "$ARCH" = "x86_64" ]; then \
         S5CMD_ARCH="64bit"; \
       elif [ "$ARCH" = "aarch64" ]; then \
         S5CMD_ARCH="arm64"; \
       else \
         echo "Unsupported architecture: $ARCH" && exit 1; \
       fi \
    && curl -fsSL "https://github.com/peak/s5cmd/releases/download/v2.3.0/s5cmd_2.3.0_Linux-${S5CMD_ARCH}.tar.gz" \
       | tar xz -C /usr/local/bin s5cmd \
    && apt-get purge -y --auto-remove gnupg \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.9.27 /uv /uvx /bin/

# Copy Python project files and install dependencies
WORKDIR /fixtures

COPY ui/fixtures/pyproject.toml ui/fixtures/uv.lock ui/fixtures/.python-version ./

RUN uv sync --frozen
