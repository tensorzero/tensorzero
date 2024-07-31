# ========== builder ==========

FROM rust:latest AS builder

WORKDIR /src

RUN apt-get update && apt-get install -y clang libc++-dev && rm -rf /var/lib/apt/lists/*

COPY ./api .

ARG CARGO_BUILD_FLAGS=""
RUN echo "Building with flags: ${CARGO_BUILD_FLAGS}"

RUN --mount=type=cache,id=tensorzero-api-release,sharing=shared,target=/usr/local/cargo/registry \
    --mount=type=cache,id=tensorzero-api-release,sharing=shared,target=/usr/local/cargo/git \
    --mount=type=cache,id=tensorzero-api-release,sharing=locked,target=/src/target \
    cargo build --release $CARGO_BUILD_FLAGS && \
    cp -r /src/target/release /release

# ========== base ==========

FROM debian:bookworm-slim AS base

RUN apt-get update && apt-get install -y openssl ca-certificates && rm -rf /var/lib/apt/lists/*

# ========== api ==========

FROM base AS api

COPY --from=builder /release/api /usr/local/bin/api

COPY ./functions ./functions

CMD ["api"]
