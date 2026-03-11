-- Replace `model_provider_statistics` materialized view with an incrementally maintained table:
-- each row is (model, provider, minute, total input tokens, total output tokens, inference count)
--
-- The online path for computing model statistics over a time window becomes:
-- SELECT .. FROM model_provider_statistics WHERE minute >= (NOW() - INTERVAL window)
--
-- The offline aggregation path is:
-- SELECT (aggregate values)
-- FROM model_inferences
-- WHERE timestamp in window
-- GROUP BY model, provider, (timestamp truncated to minute)

-- Convert the existing materialized view into a regular table, preserving existing data.
DO $$
BEGIN
    IF EXISTS (
        SELECT 1
        FROM pg_matviews
        WHERE schemaname = 'tensorzero'
          AND matviewname = 'model_provider_statistics'
    ) THEN
        CREATE TABLE tensorzero.model_provider_statistics_table AS
        SELECT *
        FROM tensorzero.model_provider_statistics;

        DROP MATERIALIZED VIEW tensorzero.model_provider_statistics;

        ALTER TABLE tensorzero.model_provider_statistics_table
            RENAME TO model_provider_statistics;
    END IF;
END $$;

-- Recreate indexes that were previously attached to the materialized view.
CREATE UNIQUE INDEX IF NOT EXISTS idx_model_provider_stats_pk
    ON tensorzero.model_provider_statistics(model_name, model_provider_name, minute);

CREATE INDEX IF NOT EXISTS idx_model_provider_stats_minute
    ON tensorzero.model_provider_statistics(minute);

-- Watermark state for incremental refresh.
CREATE TABLE IF NOT EXISTS tensorzero.model_provider_statistics_refresh_state (
    singleton BOOLEAN PRIMARY KEY DEFAULT TRUE CHECK (singleton),
    last_processed_created_at TIMESTAMPTZ,
    last_processed_id UUID
);

-- Initialize watermark at the current tail to avoid a full-table first refresh.
INSERT INTO tensorzero.model_provider_statistics_refresh_state (
    singleton,
    last_processed_created_at,
    last_processed_id
)
SELECT
    TRUE,
    latest.created_at,
    latest.id
FROM (
    SELECT created_at, id
    FROM tensorzero.model_inferences
    ORDER BY created_at DESC, id DESC
    LIMIT 1
) AS latest
ON CONFLICT (singleton) DO NOTHING;

-- Incremental refresh for `model_provider_statistics`.
-- We reprocess a trailing lookback window to absorb slightly late-arriving rows.
CREATE OR REPLACE FUNCTION tensorzero.refresh_model_provider_statistics_incremental(
    lookback INTERVAL DEFAULT INTERVAL '10 minutes',
    full_refresh BOOLEAN DEFAULT FALSE
)
RETURNS VOID
LANGUAGE plpgsql
AS $$
DECLARE
    state_created_at TIMESTAMPTZ;
    refresh_from TIMESTAMPTZ;
    refresh_to TIMESTAMPTZ := NOW();
    newest_created_at TIMESTAMPTZ;
    newest_id UUID;
    oldest_created_at TIMESTAMPTZ;
BEGIN
    INSERT INTO tensorzero.model_provider_statistics_refresh_state (singleton)
    VALUES (TRUE)
    ON CONFLICT (singleton) DO NOTHING;

    SELECT last_processed_created_at
    INTO state_created_at
    FROM tensorzero.model_provider_statistics_refresh_state
    WHERE singleton = TRUE
    FOR UPDATE;

    IF full_refresh THEN
        TRUNCATE TABLE tensorzero.model_provider_statistics;
        refresh_from := TIMESTAMPTZ '-infinity';
    ELSIF state_created_at IS NULL THEN
        refresh_from := TIMESTAMPTZ '-infinity';
    ELSE
        refresh_from := date_trunc('minute', state_created_at - lookback);
    END IF;

    -- Upsert only affected minute buckets in the trailing refresh window.
    INSERT INTO tensorzero.model_provider_statistics (
        model_name,
        model_provider_name,
        minute,
        total_input_tokens,
        total_output_tokens,
        inference_count
    )
    SELECT
        model_name,
        model_provider_name,
        date_trunc('minute', created_at) AS minute,
        -- Don't coalesce NULL values to 0
        SUM(input_tokens)::BIGINT AS total_input_tokens,
        SUM(output_tokens)::BIGINT AS total_output_tokens,
        COUNT(*)::BIGINT AS inference_count
    FROM tensorzero.model_inferences
    WHERE created_at >= refresh_from
      AND created_at <= refresh_to
    GROUP BY model_name, model_provider_name, date_trunc('minute', created_at)
    ON CONFLICT (model_name, model_provider_name, minute) DO UPDATE
    SET
        total_input_tokens = EXCLUDED.total_input_tokens,
        total_output_tokens = EXCLUDED.total_output_tokens,
        inference_count = EXCLUDED.inference_count;

    -- Keep retention behavior correct: if old source partitions were dropped,
    -- remove stale stats buckets that are now older than the earliest source row.
    SELECT MIN(created_at)
    INTO oldest_created_at
    FROM tensorzero.model_inferences;

    IF oldest_created_at IS NULL THEN
        TRUNCATE TABLE tensorzero.model_provider_statistics;
    ELSE
        DELETE FROM tensorzero.model_provider_statistics
        WHERE minute < date_trunc('minute', oldest_created_at);
    END IF;

    SELECT created_at, id
    INTO newest_created_at, newest_id
    FROM tensorzero.model_inferences
    ORDER BY created_at DESC, id DESC
    LIMIT 1;

    UPDATE tensorzero.model_provider_statistics_refresh_state
    SET
        last_processed_created_at = newest_created_at,
        last_processed_id = newest_id
    WHERE singleton = TRUE;
END;
$$;
