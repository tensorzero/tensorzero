-- Config Snapshots
CREATE TABLE IF NOT EXISTS tensorzero.config_snapshots (
    hash BYTEA PRIMARY KEY,
    config TEXT NOT NULL,
    extra_templates JSONB NOT NULL DEFAULT '{}',
    tensorzero_version TEXT NOT NULL,
    tags JSONB NOT NULL DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    last_used TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Deployment ID (singleton table)
CREATE TABLE IF NOT EXISTS tensorzero.deployment_id (
    deployment_id TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    dummy BIGINT NOT NULL DEFAULT 0,
    CONSTRAINT deploymentid_dummy_is_zero CHECK (dummy = 0),
    CONSTRAINT deploymentid_singleton UNIQUE (dummy)
);
