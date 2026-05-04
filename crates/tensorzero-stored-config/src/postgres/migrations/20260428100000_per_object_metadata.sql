-- Per-object metadata column.
--
-- Every per-object config row carries a `metadata JSONB NOT NULL DEFAULT
-- '{}'` field for editor / Autopilot / file-import provenance:
--
--   metadata.notes              — block-level TOML comments (file mode) or
--                                 the "Notes" textarea (UI mode)
--   metadata.created_by         — "autopilot/session-abc", "ui/user-123",
--                                 "file-import"
--   metadata.tags               — UI tag editor, grouping, filtering
--   metadata.source_file        — TOML-mode provenance
--   metadata.created_at_source  — TOML-mode provenance
--
-- IMPORTANT: `metadata` is NOT included in the snapshot canonical hash.
-- Two rows whose configs hash identically but whose metadata differ are
-- considered the same logical config; metadata is purely descriptive.
--
-- The default `'{}'::jsonb` covers existing rows without an explicit
-- backfill. NULL is disallowed so callers don't need to special-case
-- absent metadata.

ALTER TABLE tensorzero.function_configs
    ADD COLUMN metadata JSONB NOT NULL DEFAULT '{}'::jsonb;

ALTER TABLE tensorzero.variant_configs
    ADD COLUMN metadata JSONB NOT NULL DEFAULT '{}'::jsonb;

ALTER TABLE tensorzero.stored_files
    ADD COLUMN metadata JSONB NOT NULL DEFAULT '{}'::jsonb;

ALTER TABLE tensorzero.tools_configs
    ADD COLUMN metadata JSONB NOT NULL DEFAULT '{}'::jsonb;

ALTER TABLE tensorzero.evaluations_configs
    ADD COLUMN metadata JSONB NOT NULL DEFAULT '{}'::jsonb;

ALTER TABLE tensorzero.models_configs
    ADD COLUMN metadata JSONB NOT NULL DEFAULT '{}'::jsonb;

ALTER TABLE tensorzero.embedding_models_configs
    ADD COLUMN metadata JSONB NOT NULL DEFAULT '{}'::jsonb;

ALTER TABLE tensorzero.metrics_configs
    ADD COLUMN metadata JSONB NOT NULL DEFAULT '{}'::jsonb;

ALTER TABLE tensorzero.optimizers_configs
    ADD COLUMN metadata JSONB NOT NULL DEFAULT '{}'::jsonb;

-- Singletons (one row per gateway deployment).
ALTER TABLE tensorzero.gateway_configs
    ADD COLUMN metadata JSONB NOT NULL DEFAULT '{}'::jsonb;

ALTER TABLE tensorzero.clickhouse_configs
    ADD COLUMN metadata JSONB NOT NULL DEFAULT '{}'::jsonb;

ALTER TABLE tensorzero.postgres_configs
    ADD COLUMN metadata JSONB NOT NULL DEFAULT '{}'::jsonb;

ALTER TABLE tensorzero.object_storage_configs
    ADD COLUMN metadata JSONB NOT NULL DEFAULT '{}'::jsonb;

ALTER TABLE tensorzero.rate_limiting_configs
    ADD COLUMN metadata JSONB NOT NULL DEFAULT '{}'::jsonb;

ALTER TABLE tensorzero.autopilot_configs
    ADD COLUMN metadata JSONB NOT NULL DEFAULT '{}'::jsonb;

ALTER TABLE tensorzero.provider_types_configs
    ADD COLUMN metadata JSONB NOT NULL DEFAULT '{}'::jsonb;

-- GIN index on `metadata.tags` keys for the future "filter by tag" UI.
-- Uses the default GIN op class (not jsonb_path_ops) because tag filters
-- typically use `?` / `?&` / `?|` (key-existence operators) rather than
-- `@>` containment.
CREATE INDEX IF NOT EXISTS function_configs_metadata_tags_gin
    ON tensorzero.function_configs USING gin ((metadata -> 'tags'));
CREATE INDEX IF NOT EXISTS variant_configs_metadata_tags_gin
    ON tensorzero.variant_configs USING gin ((metadata -> 'tags'));
