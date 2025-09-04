# ========== builder ==========
FROM rust:1.88.0-slim AS builder

WORKDIR /src

# Install dependencies for building
RUN apt-get update && apt-get install -y \
    clang \
    libc++-dev \
    pkg-config \
    libssl-dev \
    python3 \
    python3-pip \
    python3-venv \
    && rm -rf /var/lib/apt/lists/*

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Copy the entire workspace
COPY . .

# Install maturin and build the python client wheel with e2e_tests feature
RUN --mount=type=cache,id=tensorzero-python-release,sharing=locked,target=/usr/local/cargo/registry \
    --mount=type=cache,id=tensorzero-python-release,sharing=locked,target=/usr/local/cargo/git \
    uv tool install maturin && \
    cd clients/python && \
    uv tool run maturin build --features e2e_tests --release --out dist --find-interpreter

# ========== runtime ==========
FROM ghcr.io/astral-sh/uv:debian-slim AS runtime

# Install system dependencies including build tools
RUN apt-get update && apt-get install -y \
    ca-certificates \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy only the Python client files needed for tests (excluding any built artifacts)
COPY clients/python/pyproject.toml /app/clients/python/
COPY clients/python/uv.lock /app/clients/python/
COPY clients/python/pytest.ini /app/clients/python/
COPY clients/python/tests/ /app/clients/python/tests/
COPY clients/python/test-no-rebuild.sh /app/clients/python/

# Change to the Python client directory
WORKDIR /app/clients/python

# Copy the wheel from builder stage
COPY --from=builder /src/clients/python/dist/*.whl /tmp/

# Create virtual environment and install dependencies (excluding the local tensorzero package)
RUN uv sync --frozen --no-install-project && uv pip install $(ls /tmp/tensorzero-*.whl | head -1)

# Set default command to run tests without rebuilding
CMD ["./test-no-rebuild.sh"]
