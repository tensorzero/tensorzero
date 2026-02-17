# ========== builder ==========

FROM rust:1.93.0-slim AS builder

WORKDIR /src

RUN apt-get update && apt-get install -y clang libc++-dev git && rm -rf /var/lib/apt/lists/*

COPY . .
RUN git rev-parse HEAD

ARG CARGO_BUILD_FLAGS=""

RUN cargo build -p provider-proxy $CARGO_BUILD_FLAGS && \
    mkdir -p /release && cp -r /src/target/debug/provider-proxy /release/provider-proxy

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
