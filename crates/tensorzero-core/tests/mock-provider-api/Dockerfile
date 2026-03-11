# ========== builder ==========

FROM lukemathwalker/cargo-chef:latest-rust-1.93-trixie AS chef

WORKDIR /src

FROM chef AS planner

# Only copy workspace manifests and mock-provider-api to avoid cache invalidation from unrelated crate changes
COPY crates/Cargo.toml crates/Cargo.lock ./
COPY crates/tensorzero-core/tests/mock-provider-api/ tensorzero-core/tests/mock-provider-api/
# Trim workspace to only include mock-provider-api
RUN sed -i '/^members = \[/,/^\]/c\members = ["tensorzero-core/tests/mock-provider-api"]' Cargo.toml
RUN cargo chef prepare --recipe-path recipe.json

FROM chef AS builder
ENV CARGO_INCREMENTAL=0

RUN apt-get update && apt-get install -y mold && rm -rf /var/lib/apt/lists/*

# Set up the Rust toolchain
COPY crates/rust-toolchain.toml ./rust-toolchain.toml
RUN rustup show

# Copy the recipe and install dependencies (trim workspace to just this crate)
COPY --from=planner /src/recipe.json ./recipe.json
COPY crates/Cargo.toml crates/Cargo.lock ./
RUN sed -i '/^members = \[/,/^\]/c\members = ["tensorzero-core/tests/mock-provider-api"]' Cargo.toml
ARG CARGO_BUILD_FLAGS=""
RUN cargo chef cook --profile dev -p mock-provider-api ${CARGO_BUILD_FLAGS}

# Re-copy the real workspace Cargo.toml (cargo chef cook overwrites it with a minimal version)
COPY crates/Cargo.toml crates/Cargo.lock ./
RUN sed -i '/^members = \[/,/^\]/c\members = ["tensorzero-core/tests/mock-provider-api"]' Cargo.toml

# Copy the crate itself and build it
COPY crates/tensorzero-core/tests/mock-provider-api/ tensorzero-core/tests/mock-provider-api/

RUN cargo build -p mock-provider-api ${CARGO_BUILD_FLAGS} && \
    mkdir -p /release && \
    cp target/debug/mock-provider-api /release/mock-provider-api

# ========== mock-provider-api ==========

FROM gcr.io/distroless/cc-debian13 AS mock-provider-api

COPY --from=builder /release/mock-provider-api /usr/local/bin/mock-provider-api

WORKDIR /app

EXPOSE 3030

USER nonroot:nonroot

HEALTHCHECK --start-period=30s --start-interval=1s --timeout=1s CMD ["mock-provider-api", "health-check"]

ENTRYPOINT ["mock-provider-api"]
