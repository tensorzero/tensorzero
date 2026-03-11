# ========== builder ==========

FROM rust:1.93.0-slim AS builder

WORKDIR /src
ENV CARGO_INCREMENTAL=0

RUN apt-get update && apt-get install -y clang libc++-dev mold && rm -rf /var/lib/apt/lists/*
ENV CARGO_TARGET_X86_64_UNKNOWN_LINUX_GNU_RUSTFLAGS="-C link-arg=-fuse-ld=mold"
ENV CARGO_TARGET_AARCH64_UNKNOWN_LINUX_GNU_RUSTFLAGS="-C link-arg=-fuse-ld=mold"

COPY crates/ crates/

ARG CARGO_BUILD_FLAGS=""

WORKDIR /src/crates
RUN cargo build -p provider-proxy $CARGO_BUILD_FLAGS && \
    mkdir -p /release && cp -r /src/crates/target/debug/provider-proxy /release/provider-proxy

# ========== provider-proxy ==========

FROM gcr.io/distroless/cc-debian13 AS provider-proxy

COPY --from=builder /release/provider-proxy /usr/local/bin/provider-proxy

WORKDIR /app

USER nonroot:nonroot

# Port 3003: proxy
# Port 3004: health check
EXPOSE 3003 3004

HEALTHCHECK --start-period=30s --start-interval=1s --timeout=1s CMD ["provider-proxy", "health-check"]

ENTRYPOINT ["provider-proxy"]

CMD ["--help"]
