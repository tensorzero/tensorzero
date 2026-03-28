-- Tracks background data migrations that run post-startup in Rust.
-- Used to ensure each migration runs exactly once across all gateway instances.
-- Distributed locking is handled via pg_try_advisory_lock in Rust code;
-- this table records completion status.
CREATE TABLE IF NOT EXISTS tensorzero.background_migrations (
    name TEXT PRIMARY KEY,
    started_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    completed_at TIMESTAMPTZ,
    rows_affected BIGINT NOT NULL DEFAULT 0
);
