# Global build args
ARG CARGO_BUILD_FLAGS=""
ARG PROFILE="release"

# ========== builder ==========

FROM rust:1.88.0 AS builder
ARG CARGO_BUILD_FLAGS
ARG PROFILE

WORKDIR /src

RUN apt-get update && apt-get install -y clang libc++-dev && rm -rf /var/lib/apt/lists/*

COPY . .

RUN cargo build --profile ${PROFILE} -p evaluations $CARGO_BUILD_FLAGS
RUN if [ "${PROFILE}" = "dev" ]; then cp -r /src/target/debug /release; else cp -r /src/target/${PROFILE} /release; fi

# ========== base ==========

FROM debian:trixie-slim AS base

RUN apt-get update && apt-get install -y ca-certificates openssl wget && rm -rf /var/lib/apt/lists/*

# ========== evaluations ==========

FROM base AS evaluations

RUN useradd -m -s /bin/bash evaluations

USER evaluations

COPY --from=builder /release/evaluations /usr/local/bin/evaluations

WORKDIR /app

ENTRYPOINT ["evaluations"]
