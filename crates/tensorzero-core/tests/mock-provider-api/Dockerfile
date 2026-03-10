# ========== builder ==========

FROM lukemathwalker/cargo-chef:latest-rust-1.93-trixie AS chef

WORKDIR /src

FROM chef AS planner
WORKDIR /app/crates
# Only copy workspace manifests and mock-provider-api's Cargo.toml
# to avoid cache invalidation from unrelated crate changes
COPY crates/Cargo.toml crates/Cargo.lock ./
COPY crates/tensorzero-core/tests/mock-provider-api/ tensorzero-core/tests/mock-provider-api/
# Trim workspace to only include mock-provider-api
RUN sed -i '/^members = \[/,/^\]/c\members = ["tensorzero-core/tests/mock-provider-api"]' Cargo.toml
RUN cargo chef prepare --recipe-path recipe.json

FROM chef AS builder
WORKDIR /tensorzero
ENV CARGO_INCREMENTAL=0
COPY --from=planner /app/crates/recipe.json /tensorzero/crates/recipe.json

# Copy the crate and relevant files; trim workspace to remove other crates
COPY crates/rust-toolchain.toml /tensorzero/crates/rust-toolchain.toml
COPY crates/Cargo.toml crates/Cargo.lock /tensorzero/crates/
RUN sed -i '/^members = \[/,/^\]/c\members = ["tensorzero-core/tests/mock-provider-api"]' /tensorzero/crates/Cargo.toml

WORKDIR /tensorzero/crates
ARG CARGO_BUILD_FLAGS=""
RUN cargo chef cook --profile dev -p mock-provider-api ${CARGO_BUILD_FLAGS}

# Re-copy the real workspace Cargo.toml (cargo chef cook overwrites it with a minimal version)
COPY crates/Cargo.toml crates/Cargo.lock /tensorzero/crates/
RUN sed -i '/^members = \[/,/^\]/c\members = ["tensorzero-core/tests/mock-provider-api"]' /tensorzero/crates/Cargo.toml
COPY crates/tensorzero-core/tests/mock-provider-api/ /tensorzero/crates/tensorzero-core/tests/mock-provider-api/

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
