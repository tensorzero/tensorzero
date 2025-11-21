# ========== builder ==========

FROM rust:1.88.0 AS builder

WORKDIR /src

RUN apt-get update && apt-get install -y clang libc++-dev && rm -rf /var/lib/apt/lists/*

COPY . .
RUN git rev-parse HEAD

ARG CARGO_BUILD_FLAGS=""

RUN cargo build -p provider-proxy $CARGO_BUILD_FLAGS && \
    cp -r /src/target/debug/provider-proxy /release/provider-proxy

# ========== base ==========

FROM debian:bookworm-slim AS base

RUN apt-get update && apt-get install -y ca-certificates openssl wget && rm -rf /var/lib/apt/lists/*

# ========== provider-proxy ==========

FROM base AS provider-proxy

COPY --from=builder /release/provider-proxy /usr/local/bin/provider-proxy

WORKDIR /app

EXPOSE 3003 3004

ENTRYPOINT ["provider-proxy"]

CMD ["--help"]
