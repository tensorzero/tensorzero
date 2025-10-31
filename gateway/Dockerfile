# ========== builder ==========

FROM lukemathwalker/cargo-chef:latest-rust-1.90-bookworm AS chef

WORKDIR /src

FROM chef AS planner
WORKDIR /app
COPY . .
RUN git rev-parse HEAD
RUN cargo chef prepare --recipe-path recipe.json

ARG CARGO_BUILD_FLAGS=""
ARG PROFILE="performance"

FROM chef AS builder
WORKDIR /tensorzero
RUN apt-get update && apt-get install -y clang libc++-dev && rm -rf /var/lib/apt/lists/*
COPY --from=planner /app/recipe.json recipe.json
RUN cargo chef cook --profile $PROFILE -p gateway $CARGO_BUILD_FLAGS
COPY . /tensorzero

RUN cargo build --profile $PROFILE -p gateway $CARGO_BUILD_FLAGS
RUN cp -r /tensorzero/target/$PROFILE /release

# ========== base ==========

FROM debian:bookworm-slim AS base

RUN apt-get update && apt-get install -y ca-certificates openssl wget && rm -rf /var/lib/apt/lists/*

# ========== gateway ==========

FROM base AS gateway

RUN useradd -m -s /bin/bash gateway

USER gateway

COPY --from=builder /release/gateway /usr/local/bin/gateway

WORKDIR /app

EXPOSE 3000

ENTRYPOINT ["gateway"]
