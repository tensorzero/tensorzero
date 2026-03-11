# Global build args
ARG DEBUG_BUILD=0

# ========== base ==========

FROM node:24.13.0-trixie-slim AS base

RUN npm install -g pnpm

# ========== development-dependencies-env ==========

FROM base AS development-dependencies-env

WORKDIR /app

COPY package.json pnpm-lock.yaml pnpm-workspace.yaml ./
COPY ui/package.json ./ui/
COPY crates/tensorzero-node/package.json ./crates/tensorzero-node/

RUN pnpm install --frozen-lockfile

# ========== production-dependencies-env ==========

FROM base AS production-dependencies-env

WORKDIR /app

COPY package.json pnpm-lock.yaml pnpm-workspace.yaml ./
COPY ui/package.json ./ui/
COPY crates/tensorzero-node/package.json ./crates/tensorzero-node/

RUN pnpm install --frozen-lockfile --prod

# ========== tensorzero-node-build-env ==========

FROM rust:1.93.0 AS tensorzero-node-build-env
ARG DEBUG_BUILD

# Install Node.js 24 (required by @napi-rs/cli v3)
RUN curl -fsSL https://deb.nodesource.com/setup_24.x | bash - && \
    apt-get install -y nodejs mold && \
    rm -rf /var/lib/apt/lists/*
ENV CARGO_TARGET_X86_64_UNKNOWN_LINUX_GNU_RUSTFLAGS="-C link-arg=-fuse-ld=mold"
ENV CARGO_TARGET_AARCH64_UNKNOWN_LINUX_GNU_RUSTFLAGS="-C link-arg=-fuse-ld=mold"
RUN npm install -g pnpm

COPY package.json pnpm-lock.yaml pnpm-workspace.yaml /build/
COPY crates/ /build/crates/
WORKDIR /build/crates/tensorzero-node

RUN pnpm install --frozen-lockfile && if [ "$DEBUG_BUILD" = "1" ]; then pnpm run build:debug; else pnpm run build; fi

# ========== build-env ==========

FROM base AS build-env

WORKDIR /app

# Copy dependencies from development-dependencies-env
COPY --from=development-dependencies-env /app/node_modules /app/node_modules
COPY --from=development-dependencies-env /app/ui/node_modules* /app/ui/node_modules
# Copy tensorzero-node binding
COPY --from=tensorzero-node-build-env /build/crates/tensorzero-node/dist /app/crates/tensorzero-node/dist
COPY --from=tensorzero-node-build-env /build/crates/tensorzero-node/index.cjs /app/crates/tensorzero-node/index.cjs
COPY --from=tensorzero-node-build-env /build/crates/tensorzero-node/*linux*.node /app/crates/tensorzero-node/

COPY package.json pnpm-lock.yaml pnpm-workspace.yaml ./
COPY crates/tensorzero-node/package.json ./crates/tensorzero-node/
COPY ui/ ./ui/

RUN pnpm --filter=tensorzero-ui run build

# ========== ui ==========

FROM base AS ui

RUN useradd -m -s /bin/sh ui

RUN apt-get update && apt-get install -y wget && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy package.json files for workspace structure
COPY package.json pnpm-lock.yaml pnpm-workspace.yaml ./
COPY ui/package.json ./ui/
COPY crates/tensorzero-node/package.json ./crates/tensorzero-node/

# Copy production dependencies
COPY --from=production-dependencies-env /app/node_modules /app/node_modules
COPY --from=production-dependencies-env /app/ui/node_modules /app/ui/node_modules

# Copy the built UI
COPY --from=build-env /app/ui/build /app/ui/build

# Copy tensorzero-node bindings for runtime (from build-env which has complete node modules)
COPY --from=build-env /app/crates/tensorzero-node/index.cjs /app/ui/build/index.cjs
COPY --from=build-env /app/crates/tensorzero-node/*.node /app/ui/build/

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

ENTRYPOINT ["./entrypoint.sh"]
HEALTHCHECK --start-period=10s --start-interval=1s --timeout=1s CMD wget --no-verbose --tries=1 --spider http://localhost:4000/health
