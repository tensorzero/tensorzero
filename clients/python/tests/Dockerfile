# ========== builder ==========
FROM rust:1.88.0-slim AS builder

WORKDIR /src

# Install dependencies for building
RUN apt-get update && apt-get install -y \
    clang \
    libc++-dev \
    pkg-config \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy the entire workspace
COPY . .

# Build the tensorzero-python crate with e2e_tests feature
RUN --mount=type=cache,id=tensorzero-python-release,sharing=locked,target=/usr/local/cargo/registry \
    --mount=type=cache,id=tensorzero-python-release,sharing=locked,target=/usr/local/cargo/git \
    cargo build --features e2e_tests -p tensorzero-python && \
    cp -r /src/target/debug /release

# ========== runtime ==========
FROM ghcr.io/astral-sh/uv:debian-slim AS runtime

# Install system dependencies including build tools
RUN apt-get update && apt-get install -y \
    ca-certificates \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy the entire workspace for the runtime build
COPY . .

# Change to the Python client directory
WORKDIR /app/clients/python

# Copy built Rust binary from builder stage
COPY --from=builder /release/ /app/target/debug/

# Install dependencies and the project in editable mode (this will initially fail to build the extension)
RUN uv sync --frozen || true

# Copy the pre-built library directly into the Python site-packages, overriding any stub
RUN python_site_packages=$(find .venv/lib -name "site-packages" -type d | head -1) && \
    tensorzero_so=$(find /app/target/debug -name "*tensorzero*.so" | head -1) && \
    if [ -n "$tensorzero_so" ] && [ -n "$python_site_packages" ]; then \
        cp "$tensorzero_so" "$python_site_packages/"; \
        echo "Copied $tensorzero_so to $python_site_packages/"; \
        ls -la "$python_site_packages"/*tensorzero* || echo "No tensorzero files found after copy"; \
    else \
        echo "Could not find tensorzero .so file or site-packages directory"; \
        find /app/target/debug -name "*.so" || echo "No .so files found"; \
        find .venv -name "site-packages" -type d || echo "No site-packages found"; \
        exit 1; \
    fi

# Install the project properly now that we have the pre-built extension
RUN uv sync --frozen

# Set default command to run tests without rebuilding
CMD ["./test-no-rebuild.sh"]
