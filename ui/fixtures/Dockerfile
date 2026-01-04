# This Dockerfile is used to load fixtures into a (separate) ClickHouse server
FROM clickhouse:lts

RUN apt-get update && apt-get install -y cmake curl
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
    ./aws/install
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Copy Python project files and install dependencies
WORKDIR /fixtures
COPY ui/fixtures/pyproject.toml ui/fixtures/uv.lock ui/fixtures/.python-version ./
RUN uv sync --frozen
