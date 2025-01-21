FROM node:20-alpine AS base
RUN npm install -g pnpm

FROM base AS development-dependencies-env
COPY . /app
WORKDIR /app
RUN pnpm install --frozen-lockfile

FROM base AS production-dependencies-env
COPY ./package.json pnpm-lock.yaml /app/
WORKDIR /app
RUN pnpm install --frozen-lockfile --prod

FROM base AS build-env
COPY . /app/
COPY --from=development-dependencies-env /app/node_modules /app/node_modules
WORKDIR /app
RUN pnpm run build

FROM base
COPY ./package.json pnpm-lock.yaml /app/
COPY --from=production-dependencies-env /app/node_modules /app/node_modules
COPY --from=build-env /app/build /app/build
WORKDIR /app
EXPOSE 4000
CMD ["pnpm", "run", "start"]
