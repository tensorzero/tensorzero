# ========== builder ==========

FROM rust:1.88.0 AS builder

WORKDIR /src
COPY . .

ARG CARGO_BUILD_FLAGS=""

RUN cargo build --release -p mock-inference-provider $CARGO_BUILD_FLAGS && \
    cp -r /src/target/release /release

# ========== mock-inference-provider ==========

FROM gcr.io/distroless/cc-debian12:debug AS mock-inference-provider

COPY --from=builder /release/mock-inference-provider /usr/local/bin/mock-inference-provider

WORKDIR /app

EXPOSE 3030

USER nonroot:nonroot

HEALTHCHECK --start-period=30s --start-interval=1s --timeout=1s CMD ["/busybox/wget", "--spider", "--tries=1", "http://localhost:3030/status"]

ENTRYPOINT ["mock-inference-provider"]
