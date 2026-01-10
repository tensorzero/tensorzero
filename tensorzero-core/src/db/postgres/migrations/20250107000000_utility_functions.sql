-- =============================================================================
-- Migration: Utility Functions
-- Description: Creates utility functions needed by other migrations.
-- =============================================================================

-- =============================================================================
-- UUID MIN/MAX AGGREGATES
-- PostgreSQL doesn't have built-in min/max for UUID, so we create our own.
-- UUIDs are compared lexicographically (which works well for UUIDv7).
-- =============================================================================

CREATE OR REPLACE FUNCTION tensorzero_uuid_smaller(uuid, uuid)
RETURNS uuid AS $$
    SELECT CASE WHEN $1 IS NULL THEN $2
                WHEN $2 IS NULL THEN $1
                WHEN $1 < $2 THEN $1
                ELSE $2
           END;
$$ LANGUAGE SQL IMMUTABLE PARALLEL SAFE;

CREATE OR REPLACE FUNCTION tensorzero_uuid_larger(uuid, uuid)
RETURNS uuid AS $$
    SELECT CASE WHEN $1 IS NULL THEN $2
                WHEN $2 IS NULL THEN $1
                WHEN $1 > $2 THEN $1
                ELSE $2
           END;
$$ LANGUAGE SQL IMMUTABLE PARALLEL SAFE;

CREATE AGGREGATE tensorzero_min(uuid) (
    SFUNC = tensorzero_uuid_smaller,
    STYPE = uuid,
    PARALLEL = SAFE
);

CREATE AGGREGATE tensorzero_max(uuid) (
    SFUNC = tensorzero_uuid_larger,
    STYPE = uuid,
    PARALLEL = SAFE
);

-- =============================================================================
-- UUID V7 TIMESTAMP EXTRACTION
-- Extracts the embedded Unix timestamp from a UUIDv7.
-- UUIDv7 stores milliseconds since Unix epoch in the first 48 bits.
-- =============================================================================

CREATE OR REPLACE FUNCTION uuid_v7_to_timestamp(id UUID)
RETURNS TIMESTAMPTZ AS $$
DECLARE
    hex_str TEXT;
    ts_ms BIGINT;
BEGIN
    -- Remove hyphens and get the hex string
    hex_str := replace(id::text, '-', '');
    -- First 48 bits (12 hex chars) are Unix timestamp in milliseconds
    ts_ms := ('x' || substring(hex_str, 1, 12))::bit(48)::bigint;
    RETURN to_timestamp(ts_ms / 1000.0);
END;
$$ LANGUAGE plpgsql IMMUTABLE STRICT PARALLEL SAFE;
