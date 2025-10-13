# ========== base ==========

FROM node:23-bookworm-slim AS base

COPY --from=ghcr.io/astral-sh/uv:0.6.6 /uv /uvx /bin/

RUN npm install -g pnpm

# ========== cargo-chef-planner ==========

FROM lukemathwalker/cargo-chef:latest-rust-1.90-bookworm AS chef
WORKDIR /app

FROM chef AS planner
COPY . .
RUN cargo chef prepare --recipe-path recipe.json

# ========== package-jsons-layer ==========

FROM base AS package-jsons-layer
WORKDIR /src
COPY . .

# Archive only package.json and pnpm-lock.yaml recursively, skipping .venv/ and node_modules and .vite
# pnpm needs these files for the workspace to work
RUN find . -type d \( -name .venv -o -name node_modules -o -name .vite \) -prune -false -o \
    -type f \( -name package.json -o -name pnpm-lock.yaml -o -name pnpm-workspace.yaml \) \
    -exec tar --transform='s|^\./||' --no-recursion -rf /tmp/packages.tar {} \;

# ========== development-dependencies-env ==========

FROM base AS development-dependencies-env

WORKDIR /app

COPY --from=package-jsons-layer /tmp/packages.tar /tmp/packages.tar
RUN tar -vxf /tmp/packages.tar -C .

RUN pnpm install --frozen-lockfile

# ========== production-dependencies-env ==========

FROM base AS production-dependencies-env

WORKDIR /app

COPY --from=package-jsons-layer /tmp/packages.tar /tmp/packages.tar
RUN tar -vxf /tmp/packages.tar -C .

RUN pnpm install --frozen-lockfile --prod

# ========== tensorzero-node-build-env ==========

FROM rust:1.88.0 AS tensorzero-node-build-env

RUN apt-get update && apt-get install -y nodejs npm && rm -rf /var/lib/apt/lists/*
RUN npm install -g pnpm

COPY . /build
WORKDIR /build/internal/tensorzero-node

RUN pnpm install --frozen-lockfile && pnpm run build

# ========== build-env ==========

FROM base AS build-env

WORKDIR /app

# Copy dependencies from development-dependencies-env
COPY --from=development-dependencies-env /app/node_modules /app/node_modules
COPY --from=development-dependencies-env /app/ui/node_modules* /app/ui/node_modules
# Copy tensorzero-node binding
COPY --from=tensorzero-node-build-env /build/internal/tensorzero-node/dist /app/internal/tensorzero-node/dist
COPY --from=tensorzero-node-build-env /build/internal/tensorzero-node/index.cjs /app/internal/tensorzero-node/index.cjs
COPY --from=tensorzero-node-build-env /build/internal/tensorzero-node/*linux*.node /app/internal/tensorzero-node/

COPY . .

RUN pnpm --filter=tensorzero-ui run build

# ========== ui ==========

FROM base AS ui

RUN useradd -m -s /bin/sh ui

RUN apt-get update && apt-get install -y curl wget python3 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy package.json files for workspace structure
COPY --from=package-jsons-layer /tmp/packages.tar /tmp/packages.tar
RUN tar -vxf /tmp/packages.tar -C .

# Copy production dependencies
COPY --from=production-dependencies-env /app/node_modules /app/node_modules
COPY --from=production-dependencies-env /app/ui/node_modules /app/ui/node_modules

# Copy the built UI
COPY --from=build-env /app/ui/build /app/ui/build

# Copy tensorzero-node bindings for runtime (from build-env which has complete node modules)
COPY --from=build-env /app/internal/tensorzero-node/index.cjs /app/ui/build/index.cjs
COPY --from=build-env /app/internal/tensorzero-node/*.node /app/ui/build/

# Copy entrypoint script
# This script calls into the /app/ui directory using `pnpm --filter=tensorzero-ui run start`
COPY ./ui/entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

USER ui

EXPOSE 4000

ENV HOST=0.0.0.0
ENV NODE_ENV=production
ENV PORT=4000
ENV RUST_LOG=warn
ENV TENSORZERO_UI_CONFIG_PATH=/app/config/tensorzero.toml

ENTRYPOINT ["./entrypoint.sh"]
HEALTHCHECK --start-period=10s --start-interval=1s --timeout=1s CMD wget --no-verbose --tries=1 --spider http://localhost:4000/health
