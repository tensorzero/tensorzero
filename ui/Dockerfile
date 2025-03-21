# ========== base ==========

FROM node:20-alpine AS base

RUN npm install -g pnpm

# ========== development-dependencies-env ==========

FROM base AS development-dependencies-env

COPY ./package.json pnpm-lock.yaml /app/

WORKDIR /app

RUN pnpm install --frozen-lockfile

# ========== production-dependencies-env ==========

FROM base AS production-dependencies-env

COPY ./package.json pnpm-lock.yaml /app/

WORKDIR /app

RUN pnpm install --frozen-lockfile --prod

# ========== rust-build-env ==========

FROM rust:latest AS rust-build-env

COPY ./app/utils/minijinja /minijinja/

WORKDIR /minijinja
RUN curl https://rustwasm.github.io/wasm-pack/installer/init.sh -sSf | sh
RUN wasm-pack build --features console_error_panic_hook

# ========== build-env ==========

FROM base AS build-env

COPY . /app/
COPY --from=rust-build-env /minijinja/pkg /app/app/utils/minijinja/pkg
COPY --from=development-dependencies-env /app/node_modules /app/node_modules

WORKDIR /app

RUN pnpm run build

# ========== ui ==========

FROM base AS ui

RUN adduser -D -s /bin/sh ui

USER ui

COPY ./package.json pnpm-lock.yaml /app/

COPY --from=production-dependencies-env /app/node_modules /app/node_modules

COPY --from=build-env /app/build /app/build

WORKDIR /app

EXPOSE 4000

ENV HOST=0.0.0.0

CMD ["pnpm", "run", "start"]
