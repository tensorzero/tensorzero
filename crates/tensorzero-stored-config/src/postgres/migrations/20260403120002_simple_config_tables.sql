CREATE SCHEMA IF NOT EXISTS tensorzero;

CREATE TABLE IF NOT EXISTS tensorzero.tools_configs (
    id UUID PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    schema_revision INT NOT NULL,
    config JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS tensorzero.evaluations_configs (
    id UUID PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    schema_revision INT NOT NULL,
    config JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE tensorzero.gateway_configs (
    id UUID PRIMARY KEY,
    schema_revision INT NOT NULL,
    config JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE tensorzero.clickhouse_configs (
    id UUID PRIMARY KEY,
    schema_revision INT NOT NULL,
    config JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE tensorzero.postgres_configs (
    id UUID PRIMARY KEY,
    schema_revision INT NOT NULL,
    config JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE tensorzero.object_storage_configs (
    id UUID PRIMARY KEY,
    schema_revision INT NOT NULL,
    config JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE tensorzero.models_configs (
    id UUID PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    schema_revision INT NOT NULL,
    config JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE tensorzero.embedding_models_configs (
    id UUID PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    schema_revision INT NOT NULL,
    config JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE tensorzero.metrics_configs (
    id UUID PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    schema_revision INT NOT NULL,
    config JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE tensorzero.rate_limiting_configs (
    id UUID PRIMARY KEY,
    schema_revision INT NOT NULL,
    config JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE tensorzero.autopilot_configs (
    id UUID PRIMARY KEY,
    schema_revision INT NOT NULL,
    config JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE tensorzero.provider_types_configs (
    id UUID PRIMARY KEY,
    schema_revision INT NOT NULL,
    config JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE tensorzero.optimizers_configs (
    id UUID PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    schema_revision INT NOT NULL,
    config JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
