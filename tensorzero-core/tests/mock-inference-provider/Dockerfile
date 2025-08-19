# ========== builder ==========

FROM rust:1.88.0 AS builder

WORKDIR /src
COPY . .

ARG CARGO_BUILD_FLAGS=""

RUN --mount=type=cache,id=tensorzero-mock-inference-provider-release,sharing=shared,target=/usr/local/cargo/registry \
    --mount=type=cache,id=tensorzero-mock-inference-provider-release,sharing=shared,target=/usr/local/cargo/git \
    cargo build --release -p mock-inference-provider $CARGO_BUILD_FLAGS && \
    cp -r /src/target/release /release

# ========== mock-inference-provider ==========

FROM debian:bookworm-slim AS mock-inference-provider

RUN apt-get update && apt-get install -y wget

RUN useradd -m -s /bin/bash mock-inference-provider

USER mock-inference-provider

COPY --from=builder /release/mock-inference-provider /usr/local/bin/mock-inference-provider

WORKDIR /app

EXPOSE 3030

ENTRYPOINT ["mock-inference-provider"]
