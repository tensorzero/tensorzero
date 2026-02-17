# ========== builder ==========

FROM rust:1.93.0 AS builder

WORKDIR /src
ENV CARGO_INCREMENTAL=0
COPY . .

ARG CARGO_BUILD_FLAGS=""

RUN cargo build -p mock-provider-api $CARGO_BUILD_FLAGS && \
    mkdir -p /release && \
    cp -r /src/target/debug/mock-provider-api /release/mock-provider-api

# ========== mock-provider-api ==========

FROM gcr.io/distroless/cc-debian13 AS mock-provider-api

COPY --from=builder /release/mock-provider-api /usr/local/bin/mock-provider-api

WORKDIR /app

EXPOSE 3030

USER nonroot:nonroot

HEALTHCHECK --start-period=30s --start-interval=1s --timeout=1s CMD ["mock-provider-api", "health-check"]

ENTRYPOINT ["mock-provider-api"]
