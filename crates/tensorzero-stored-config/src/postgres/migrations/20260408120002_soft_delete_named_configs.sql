-- Add `deleted_at` tombstone columns to every upsert-per-name config table so
-- the full-config apply path can mark items the user removed from the TOML
-- as deleted (instead of silently leaving stale rows behind). The read path
-- filters out rows where `deleted_at IS NOT NULL`. Re-adding a previously
-- deleted name revives the row by clearing `deleted_at` in the upsert.
--
-- `function_configs` already has `deleted_at` (added in the initial migration),
-- so it is not included here.

ALTER TABLE tensorzero.tools_configs
    ADD COLUMN IF NOT EXISTS deleted_at TIMESTAMPTZ;

ALTER TABLE tensorzero.evaluations_configs
    ADD COLUMN IF NOT EXISTS deleted_at TIMESTAMPTZ;

ALTER TABLE tensorzero.models_configs
    ADD COLUMN IF NOT EXISTS deleted_at TIMESTAMPTZ;

ALTER TABLE tensorzero.embedding_models_configs
    ADD COLUMN IF NOT EXISTS deleted_at TIMESTAMPTZ;

ALTER TABLE tensorzero.metrics_configs
    ADD COLUMN IF NOT EXISTS deleted_at TIMESTAMPTZ;

ALTER TABLE tensorzero.optimizers_configs
    ADD COLUMN IF NOT EXISTS deleted_at TIMESTAMPTZ;
