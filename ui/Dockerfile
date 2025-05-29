# ========== base ==========

FROM node:23-bookworm-slim AS base

COPY --from=ghcr.io/astral-sh/uv:0.6.6 /uv /uvx /bin/

RUN npm install -g pnpm

# ========== cargo-chef-planner ==========
FROM lukemathwalker/cargo-chef:latest-rust-1 AS chef
WORKDIR /app

FROM chef AS planner
COPY . .
RUN cargo chef prepare --recipe-path recipe.json

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

# ========== minijinja-build-env ==========

FROM rust:latest AS minijinja-build-env

COPY . /build

WORKDIR /build/ui/app/utils/minijinja
RUN curl https://rustwasm.github.io/wasm-pack/installer/init.sh -sSf | sh
RUN wasm-pack build --features console_error_panic_hook

# ========== evaluations-build-env ==========

FROM chef AS evaluations-build-env

RUN apt-get update && apt-get install -y clang libc++-dev && rm -rf /var/lib/apt/lists/*
WORKDIR /tensorzero
COPY --from=planner /app/recipe.json recipe.json
RUN cargo chef cook --release -p evaluations --recipe-path recipe.json
COPY . /tensorzero

ARG CARGO_BUILD_FLAGS=""

RUN cargo build --release -p evaluations $CARGO_BUILD_FLAGS && \
    cp -r /tensorzero/target/release /release


# ========== build-env ==========

FROM base AS build-env

COPY ./ui /app/

COPY --from=minijinja-build-env /build/ui/app/utils/minijinja/pkg /app/app/utils/minijinja/pkg

COPY --from=development-dependencies-env /app/node_modules /app/node_modules

WORKDIR /app

RUN pnpm run build

# ========== python-build-env ==========

FROM rust:latest AS python-build-env
RUN apt-get update && apt-get install -y python3
COPY --from=ghcr.io/astral-sh/uv:0.6.6 /uv /uvx /bin/
COPY . /build
WORKDIR /build/optimization-server
# TODO - figure out how to cache the rust build here
# UV_PYTHON_DOWNLOADS=never
# https://hynek.me/articles/docker-uv/
ENV UV_COMPILE_BYTECODE=1
RUN uv  --verbose sync --locked --no-dev

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
