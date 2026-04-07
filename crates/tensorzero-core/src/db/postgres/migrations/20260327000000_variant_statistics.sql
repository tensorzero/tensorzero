-- Create the `variant_statistics` rollup table, aggregating metrics by
-- (function_name, variant_name, minute). This is the variant-level counterpart
-- of `model_provider_statistics`.
--
-- Token/cost metrics come from `model_inferences` joined via inference_id.
-- Inference count comes from `chat_inferences` UNION ALL `json_inferences`.
--
-- Latency quantiles (processing_time_ms, ttft_ms) are NOT included here —
-- they can be added in a follow-up via separate materialized views, matching
-- the `model_latency_quantiles` pattern.

CREATE TABLE IF NOT EXISTS tensorzero.variant_statistics (
    function_name TEXT NOT NULL,
    variant_name TEXT NOT NULL,
    minute TIMESTAMPTZ NOT NULL,
    total_input_tokens BIGINT,
    total_output_tokens BIGINT,
    inference_count BIGINT NOT NULL,
    total_cost NUMERIC(18, 9),
    count_with_cost BIGINT,
    total_provider_cache_read_input_tokens BIGINT,
    total_provider_cache_write_input_tokens BIGINT,
    PRIMARY KEY (function_name, variant_name, minute)
);

CREATE INDEX IF NOT EXISTS idx_variant_stats_minute
    ON tensorzero.variant_statistics(minute);

-- Watermark state for incremental refresh.
CREATE TABLE IF NOT EXISTS tensorzero.variant_statistics_refresh_state (
    singleton BOOLEAN PRIMARY KEY DEFAULT TRUE CHECK (singleton),
    last_processed_created_at TIMESTAMPTZ,
    last_processed_id UUID
);

-- Initialize watermark so the refresh function has a starting point.
-- We use the latest created_at from both chat_inferences and json_inferences.
INSERT INTO tensorzero.variant_statistics_refresh_state (
    singleton,
    last_processed_created_at,
    last_processed_id
)
SELECT
    TRUE,
    latest.created_at,
    latest.id
FROM (
    (SELECT created_at, id FROM tensorzero.chat_inferences ORDER BY created_at DESC, id DESC LIMIT 1)
    UNION ALL
    (SELECT created_at, id FROM tensorzero.json_inferences ORDER BY created_at DESC, id DESC LIMIT 1)
) AS latest
ORDER BY created_at DESC, id DESC
LIMIT 1
ON CONFLICT (singleton) DO NOTHING;

-- Incremental refresh for `variant_statistics`.
-- Reprocesses a trailing lookback window to absorb slightly late-arriving rows.
CREATE OR REPLACE FUNCTION tensorzero.refresh_variant_statistics_incremental(
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
    INSERT INTO tensorzero.variant_statistics_refresh_state (singleton)
    VALUES (TRUE)
    ON CONFLICT (singleton) DO NOTHING;

    SELECT last_processed_created_at
    INTO state_created_at
    FROM tensorzero.variant_statistics_refresh_state
    WHERE singleton = TRUE
    FOR UPDATE;

    IF full_refresh THEN
        TRUNCATE TABLE tensorzero.variant_statistics;
        refresh_from := TIMESTAMPTZ '-infinity';
    ELSIF state_created_at IS NULL THEN
        refresh_from := TIMESTAMPTZ '-infinity';
    ELSE
        refresh_from := date_trunc('minute', state_created_at - lookback);
    END IF;

    -- Upsert affected minute buckets in the trailing refresh window.
    -- Source: chat_inferences + json_inferences LEFT JOIN model_inferences.
    INSERT INTO tensorzero.variant_statistics (
        function_name,
        variant_name,
        minute,
        total_input_tokens,
        total_output_tokens,
        inference_count,
        total_cost,
        count_with_cost,
        total_provider_cache_read_input_tokens,
        total_provider_cache_write_input_tokens
    )
    SELECT
        ci.function_name,
        ci.variant_name,
        date_trunc('minute', ci.created_at) AS minute,
        SUM(mi.input_tokens)::BIGINT AS total_input_tokens,
        SUM(mi.output_tokens)::BIGINT AS total_output_tokens,
        COUNT(DISTINCT ci.id)::BIGINT AS inference_count,
        SUM(mi.cost) AS total_cost,
        COUNT(mi.cost)::BIGINT AS count_with_cost,
        SUM(mi.provider_cache_read_input_tokens)::BIGINT AS total_provider_cache_read_input_tokens,
        SUM(mi.provider_cache_write_input_tokens)::BIGINT AS total_provider_cache_write_input_tokens
    FROM (
        SELECT id, function_name, variant_name, created_at
        FROM tensorzero.chat_inferences
        WHERE created_at >= refresh_from AND created_at <= refresh_to
        UNION ALL
        SELECT id, function_name, variant_name, created_at
        FROM tensorzero.json_inferences
        WHERE created_at >= refresh_from AND created_at <= refresh_to
    ) ci
    LEFT JOIN tensorzero.model_inferences mi ON mi.inference_id = ci.id
    GROUP BY ci.function_name, ci.variant_name, date_trunc('minute', ci.created_at)
    ON CONFLICT (function_name, variant_name, minute) DO UPDATE
    SET
        total_input_tokens = EXCLUDED.total_input_tokens,
        total_output_tokens = EXCLUDED.total_output_tokens,
        inference_count = EXCLUDED.inference_count,
        total_cost = EXCLUDED.total_cost,
        count_with_cost = EXCLUDED.count_with_cost,
        total_provider_cache_read_input_tokens = EXCLUDED.total_provider_cache_read_input_tokens,
        total_provider_cache_write_input_tokens = EXCLUDED.total_provider_cache_write_input_tokens;

    -- Keep retention correct: remove stale stats older than the earliest source row.
    SELECT LEAST(
        (SELECT MIN(created_at) FROM tensorzero.chat_inferences),
        (SELECT MIN(created_at) FROM tensorzero.json_inferences)
    )
    INTO oldest_created_at;

    IF oldest_created_at IS NULL THEN
        TRUNCATE TABLE tensorzero.variant_statistics;
    ELSE
        DELETE FROM tensorzero.variant_statistics
        WHERE minute < date_trunc('minute', oldest_created_at);
    END IF;

    -- Update watermark to the latest inference across both tables.
    -- Push ORDER BY + LIMIT into each branch so Postgres can use index scans
    -- on each table independently instead of sorting the full union.
    SELECT created_at, id
    INTO newest_created_at, newest_id
    FROM (
        (SELECT created_at, id FROM tensorzero.chat_inferences ORDER BY created_at DESC, id DESC LIMIT 1)
        UNION ALL
        (SELECT created_at, id FROM tensorzero.json_inferences ORDER BY created_at DESC, id DESC LIMIT 1)
    ) latest
    ORDER BY created_at DESC, id DESC
    LIMIT 1;

    UPDATE tensorzero.variant_statistics_refresh_state
    SET
        last_processed_created_at = newest_created_at,
        last_processed_id = newest_id
    WHERE singleton = TRUE;
END;
$$;

-- Backfill all historical data into variant_statistics.
SELECT tensorzero.refresh_variant_statistics_incremental(full_refresh => TRUE);
