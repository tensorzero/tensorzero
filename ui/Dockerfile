# ========== base ==========

FROM node:23-bookworm-slim AS base

COPY --from=ghcr.io/astral-sh/uv:0.6.6 /uv /uvx /bin/

RUN npm install -g pnpm

# ========== development-dependencies-env ==========

FROM base AS development-dependencies-env

COPY ./ui/package.json ./ui/pnpm-lock.yaml /app/

WORKDIR /app

RUN pnpm install --frozen-lockfile

# ========== production-dependencies-env ==========

FROM base AS production-dependencies-env

COPY ./ui/package.json ./ui/pnpm-lock.yaml /app/

WORKDIR /app

RUN pnpm install --frozen-lockfile --prod

# ========== rust-sccache-env ==========

FROM rust:latest AS rust-sccache-env
ENV SCCACHE_VERSION=v0.10.0
RUN SCCACHE_PLATFORM=$(uname -m)-unknown-linux-musl && \
wget https://github.com/mozilla/sccache/releases/download/${SCCACHE_VERSION}/sccache-${SCCACHE_VERSION}-${SCCACHE_PLATFORM}.tar.gz && \
    tar -xzf sccache-${SCCACHE_VERSION}-${SCCACHE_PLATFORM}.tar.gz && \
    mv sccache-${SCCACHE_VERSION}-${SCCACHE_PLATFORM}/sccache /usr/local/bin/ && \
    chmod +x /usr/local/bin/sccache && \
    rm -rf sccache-${SCCACHE_VERSION}-${SCCACHE_PLATFORM} sccache-${SCCACHE_VERSION}-${SCCACHE_PLATFORM}.tar.gz
ENV RUSTC_WRAPPER=sccache SCCACHE_DIR=/sccache

# ========== minijinja-build-env ==========

FROM rust-sccache-env AS minijinja-build-env

COPY . /build

WORKDIR /build/ui/app/utils/minijinja
RUN curl https://rustwasm.github.io/wasm-pack/installer/init.sh -sSf | sh
RUN --mount=type=cache,id=tensorzero-evaluations-release,sharing=shared,target=/sccache \
    wasm-pack build --features console_error_panic_hook

# ========== evaluations-build-env ==========

FROM rust-sccache-env AS evaluations-build-env

RUN apt-get update && apt-get install -y clang libc++-dev && rm -rf /var/lib/apt/lists/*

COPY . /tensorzero

WORKDIR /tensorzero

ARG CARGO_BUILD_FLAGS=""

RUN --mount=type=cache,id=tensorzero-evaluations-release,sharing=shared,target=/sccache \
    cargo build --release -p evaluations $CARGO_BUILD_FLAGS && \
    sccache --show-stats && \
    cp -r /tensorzero/target/release /release


# ========== build-env ==========

FROM base AS build-env

COPY ./ui /app/

COPY --from=minijinja-build-env /build/ui/app/utils/minijinja/pkg /app/app/utils/minijinja/pkg

COPY --from=development-dependencies-env /app/node_modules /app/node_modules

WORKDIR /app

RUN pnpm run build

# ========== python-build-env ==========

FROM rust-sccache-env AS python-build-env
RUN apt-get update && apt-get install -y python3
COPY --from=ghcr.io/astral-sh/uv:0.6.6 /uv /uvx /bin/
COPY . /build
WORKDIR /build/optimization-server
# TODO - figure out how to cache the rust build here
# UV_PYTHON_DOWNLOADS=never
# https://hynek.me/articles/docker-uv/
ENV UV_COMPILE_BYTECODE=1
RUN --mount=type=cache,id=tensorzero-evaluations-release,sharing=shared,target=/sccache \
    uv --verbose sync --locked --no-dev

# ========== ui ==========

FROM base AS ui

RUN useradd -m -s /bin/sh ui

RUN apt-get update && apt-get install -y curl python3 && rm -rf /var/lib/apt/lists/*

COPY ./ui/package.json ./ui/pnpm-lock.yaml /app/

COPY ./ui/entrypoint.sh /app/entrypoint.sh

RUN chmod +x /app/entrypoint.sh

COPY ./optimization-server /build/optimization-server
COPY --from=python-build-env /build/clients/python /build/clients/python
COPY --from=python-build-env /build/optimization-server/.venv  /build/optimization-server/.venv
# Workaround for uv issue
#COPY ./clients/python /clients/python

USER ui

COPY --from=production-dependencies-env /app/node_modules /app/node_modules

COPY --from=build-env /app/build /app/build

COPY --from=evaluations-build-env /release/evaluations /usr/local/bin/evaluations

WORKDIR /app

EXPOSE 4000

ENV HOST=0.0.0.0

ENTRYPOINT ["./entrypoint.sh"]
