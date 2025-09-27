# ========== base ==========

FROM node:20-alpine AS base

RUN npm install -g pnpm

# ========== package-jsons-layer ==========

FROM base AS package-jsons-layer
WORKDIR /src
COPY . .

# Archive only package.json and pnpm-lock.yaml recursively, skipping node_modules
# pnpm needs these files for the workspace to work
RUN find . -name node_modules -prune -o -name .vite -prune -o \
    -name package.json -o -name pnpm-lock.yaml -o -name pnpm-workspace.yaml | \
    grep -E '\.(json|yaml)$' | \
    tar -cf /tmp/packages.tar -T -

# ========== dependencies-env ==========

FROM base AS dependencies-env

WORKDIR /app

COPY --from=package-jsons-layer /tmp/packages.tar /tmp/packages.tar
RUN tar -vxf /tmp/packages.tar -C .

RUN pnpm install --frozen-lockfile

# ========== test-runner ==========

FROM base AS test-runner

WORKDIR /app

# Copy package.json files for workspace structure
COPY --from=package-jsons-layer /tmp/packages.tar /tmp/packages.tar
RUN tar -vxf /tmp/packages.tar -C .

# Copy dependencies
COPY --from=dependencies-env /app/node_modules /app/node_modules
COPY --from=dependencies-env /app/clients/openai-node/node_modules /app/clients/openai-node/node_modules

# Copy source code
COPY clients/openai-node/ ./clients/openai-node/

# Copy test assets
COPY tensorzero-core/tests/e2e/providers/ferris.png ./tensorzero-core/tests/e2e/providers/ferris.png
COPY tensorzero-core/tests/e2e/providers/deepseek_paper.pdf ./tensorzero-core/tests/e2e/providers/deepseek_paper.pdf

WORKDIR /app/clients/openai-node

# Set environment variables
ENV NODE_ENV=ci

# Run the tests
CMD ["pnpm", "test", "--run"]
