# ========== builder ==========
ARG CARGO_BUILD_FLAGS=""

FROM lukemathwalker/cargo-chef:latest-rust-1 AS chef
WORKDIR /app

FROM chef as planner
COPY . .
RUN cargo chef prepare --recipe-path recipe.json --bin gateway

FROM chef AS builder
RUN apt-get update && apt-get install -y clang libc++-dev && rm -rf /var/lib/apt/lists/*
WORKDIR /src
COPY --from=planner /app/recipe.json recipe.json
RUN cargo chef cook --release -p gateway
COPY . .

RUN cargo build --release -p gateway $CARGO_BUILD_FLAGS && cp -r /src/target/release /release

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

CMD ["--warn-default-cmd", "--config-file", "config/tensorzero.toml"]
