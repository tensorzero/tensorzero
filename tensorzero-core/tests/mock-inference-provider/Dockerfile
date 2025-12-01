# ========== builder ==========

FROM rust:1.88.0 AS builder

WORKDIR /src
COPY . .

ARG CARGO_BUILD_FLAGS=""

RUN cargo build -p mock-inference-provider $CARGO_BUILD_FLAGS && \
    mkdir -p /release && \
    cp -r /src/target/debug/mock-inference-provider /release/mock-inference-provider

# ========== mock-inference-provider ==========

FROM debian:bookworm-slim AS mock-inference-provider

RUN apt-get update && apt-get install -y wget

RUN useradd -m -s /bin/bash mock-inference-provider

USER mock-inference-provider

COPY --from=builder /release/mock-inference-provider /usr/local/bin/mock-inference-provider

WORKDIR /app

EXPOSE 3030

ENTRYPOINT ["mock-inference-provider"]
