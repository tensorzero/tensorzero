# ========== builder ==========

FROM rust:1.88.0 AS builder

WORKDIR /src
COPY . .

ARG CARGO_BUILD_FLAGS=""

RUN cargo build --release -p mock-inference-provider $CARGO_BUILD_FLAGS && \
    cp -r /src/target/release /release

# ========== health-probe ==========

FROM gcr.io/grpc-ecosystem/grpc-health-probe:latest AS health-probe

# ========== mock-inference-provider ==========

FROM gcr.io/distroless/cc-debian12 AS mock-inference-provider

COPY --from=health-probe /ko-app/grpc-health-probe /bin/grpc-health-probe
COPY --from=builder /release/mock-inference-provider /usr/local/bin/mock-inference-provider

WORKDIR /app

EXPOSE 3030

USER nonroot:nonroot

HEALTHCHECK --start-period=1s --start-interval=1s --timeout=1s CMD ["/bin/grpc-health-probe", "-addr=localhost:3030"]

ENTRYPOINT ["mock-inference-provider"]
