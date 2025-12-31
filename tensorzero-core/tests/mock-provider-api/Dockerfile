# ========== builder ==========

FROM rust:1.88.0 AS builder

WORKDIR /src
COPY . .

ARG CARGO_BUILD_FLAGS=""

RUN cargo build -p mock-provider-api $CARGO_BUILD_FLAGS && \
    mkdir -p /release && \
    cp -r /src/target/debug/mock-provider-api /release/mock-provider-api

# ========== mock-provider-api ==========

FROM gcr.io/distroless/cc-debian12:debug AS mock-provider-api

COPY --from=builder /release/mock-provider-api /usr/local/bin/mock-provider-api

WORKDIR /app

EXPOSE 3030

USER nonroot:nonroot

HEALTHCHECK --start-period=30s --start-interval=1s --timeout=1s CMD ["/busybox/wget", "--spider", "--tries=1", "http://localhost:3030/status"]

ENTRYPOINT ["mock-provider-api"]
