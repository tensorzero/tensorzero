# ========== builder ==========

FROM rust:1.88.0 AS builder

WORKDIR /src

RUN apt-get update && apt-get install -y clang libc++-dev && rm -rf /var/lib/apt/lists/*

COPY . .

ARG CARGO_BUILD_FLAGS=""

RUN cargo build --release -p evaluations $CARGO_BUILD_FLAGS && \
    cp -r /src/target/release /release

# ========== base ==========

FROM debian:bookworm-slim AS base

RUN apt-get update && apt-get install -y ca-certificates openssl wget && rm -rf /var/lib/apt/lists/*

# ========== evaluations ==========

FROM base AS evaluations

RUN useradd -m -s /bin/bash evaluations

USER evaluations

COPY --from=builder /release/evaluations /usr/local/bin/evaluations

WORKDIR /app

ENTRYPOINT ["evaluations"]
