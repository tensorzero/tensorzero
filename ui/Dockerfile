# ========== base ==========

FROM node:23-bookworm-slim AS base

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

# ========== minijinja-build-env ==========

FROM rust:latest AS minijinja-build-env

COPY ./ui/app/utils/minijinja /minijinja/

WORKDIR /minijinja
RUN curl https://rustwasm.github.io/wasm-pack/installer/init.sh -sSf | sh
RUN wasm-pack build --features console_error_panic_hook

# ========== evaluations-build-env ==========

FROM rust:latest AS evaluations-build-env

COPY . /tensorzero

WORKDIR /tensorzero

RUN apt-get update && apt-get install -y clang libc++-dev && rm -rf /var/lib/apt/lists/*

ARG CARGO_BUILD_FLAGS=""

RUN --mount=type=cache,id=tensorzero-evaluations-release,sharing=shared,target=/usr/local/cargo/registry \
    --mount=type=cache,id=tensorzero-evaluations-release,sharing=shared,target=/usr/local/cargo/git \
    --mount=type=cache,id=tensorzero-evaluations-release,sharing=locked,target=/src/target \
    cargo build --release -p evaluations $CARGO_BUILD_FLAGS && \
    cp -r /tensorzero/target/release /release


# ========== build-env ==========

FROM base AS build-env

COPY ./ui /app/

COPY --from=minijinja-build-env /minijinja/pkg /app/app/utils/minijinja/pkg

COPY --from=development-dependencies-env /app/node_modules /app/node_modules

WORKDIR /app

RUN pnpm run build

# ========== ui ==========

FROM base AS ui

RUN useradd -m -s /bin/sh ui

USER ui

COPY ./ui/package.json ./ui/pnpm-lock.yaml /app/

COPY --from=production-dependencies-env /app/node_modules /app/node_modules

COPY --from=build-env /app/build /app/build

COPY --from=evaluations-build-env /release/evaluations /usr/local/bin/evaluations

WORKDIR /app

EXPOSE 4000

ENV HOST=0.0.0.0

CMD ["pnpm", "run", "start"]
