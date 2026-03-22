ARG GATEWAY_TAG=latest

# Stage 1: Build config-applier-cli
FROM rust:1.93 AS rust-builder

WORKDIR /build
COPY config-applier-cli/ ./
RUN cargo build --release

# Stage 2: Eval container
FROM tensorzero/gateway:${GATEWAY_TAG}

USER root

# Install git (needed by uv for git dependencies)
RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*

# Install Python via uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
ENV UV_PYTHON_INSTALL_DIR=/opt/python
RUN uv python install 3.13

# Copy config-applier CLI from builder
COPY --from=rust-builder /build/target/release/config-applier-cli /usr/local/bin/config-applier-cli

# Set up Python project
WORKDIR /app
RUN chown -R gateway:gateway /app

USER gateway

COPY --chown=gateway:gateway pyproject.toml uv.lock* ./
RUN uv sync --frozen --no-dev || uv sync --no-dev

COPY --chown=gateway:gateway src/ src/
COPY --chown=gateway:gateway interlocutor_config/ interlocutor_config/
COPY --chown=gateway:gateway configs/ configs/

ENTRYPOINT ["uv", "run", "autopilot-benchmark"]
CMD ["run", "--config", "configs/ner.yaml", "--work-dir", "/app/output", "--verbose"]
