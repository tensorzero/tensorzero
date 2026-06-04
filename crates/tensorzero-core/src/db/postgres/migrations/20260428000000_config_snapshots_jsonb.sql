-- Snapshot table evolution toward "JSON is the source of truth, structural
-- hashing is stable across roundtrips":
--
-- 1. `config_jsonb` (JSONB)              — canonical config representation
--                                         (replaces `config TEXT` for reads;
--                                          TOML column kept for migration
--                                          until `hash_v2` cutover)
-- 2. `canonical_hash` (BYTEA)            — structural Blake3 over the
--                                         canonical JSON Value tree (see
--                                         `tensorzero_core::config::snapshot::canonical_hash`).
--                                         Stable across serialize/deserialize
--                                         round-trips; the legacy `hash`
--                                         column is bound to canonical-TOML
--                                         bytes and drifts under float
--                                         reformatting.
-- 3. GIN index on `config_jsonb`         — supports `@>`, `@?`, `@@`
--                                         containment / JSONPath queries.
-- 4. Partial unique index on `canonical_hash` — content-addressed identity
--                                         for new lookups, but tolerant of
--                                         `NULL` on legacy rows pre-backfill.
--
-- Read path (Rust): `get_config_snapshot(SnapshotHash)` dispatches based on
-- the `SnapshotHashScheme` carried by the hash:
--   - `LegacyToml`  → `WHERE hash = $1`           (existing rows; back-compat)
--   - `Canonical`   → `WHERE canonical_hash = $1`  (new lookups; stable basis)
-- Both columns are populated on every new write so either lookup succeeds.
ALTER TABLE tensorzero.config_snapshots
    ADD COLUMN config_jsonb JSONB,
    ADD COLUMN canonical_hash BYTEA;

-- GIN index on the JSON column. `jsonb_path_ops` is smaller and faster than
-- the default GIN op class and supports `@>`, `@?`, `@@` (covers
-- containment + JSONPath). Does NOT support `?`, `?&`, `?|` (key-existence)
-- — add a default-class index later if those are needed.
CREATE INDEX IF NOT EXISTS config_snapshots_config_jsonb_gin
    ON tensorzero.config_snapshots
    USING gin (config_jsonb jsonb_path_ops);

-- Partial unique index on `canonical_hash`: deduplicates content-addressed
-- snapshots while tolerating NULLs on legacy rows the backfill hasn't yet
-- touched. Once backfill completes, every row has a non-NULL value and
-- this becomes a full unique index in spirit.
CREATE UNIQUE INDEX IF NOT EXISTS config_snapshots_canonical_hash_unique
    ON tensorzero.config_snapshots (canonical_hash)
    WHERE canonical_hash IS NOT NULL;
