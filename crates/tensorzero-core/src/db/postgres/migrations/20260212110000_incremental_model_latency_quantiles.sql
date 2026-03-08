-- Replace `model_latency_quantiles*` materialized views with incrementally maintained
-- sparse histogram rollups. Query-time quantile computation now happens in application SQL.
--
-- We use log2 buckets with 64 buckets per power-of-two: from the latenc number, bucket is computed as
-- log2(latency_ms) * 64, rounded down to the nearest integer. We also add a bucket -1 for <= 0. In the
-- worst case, our latencies are (2^k - epsilon) and they fall into the widest bucket for that power of two,
-- our error from the lower bound is 2^(1/64) - 1 = 1.1%. We can cut it in half by using the geometric mean
-- of the bucket as the estimated latency.
--
-- Computing estimated tail latency quantiles from the histogram is done by finding the bucket with the
-- cumulative count >= the rank target.

-- Bucket assignment helper.
CREATE OR REPLACE FUNCTION tensorzero.latency_histogram_bucket_id(
    latency_ms INTEGER,
    buckets_per_power_of_two INTEGER DEFAULT 64
)
RETURNS INTEGER
LANGUAGE sql
IMMUTABLE
AS $$
SELECT
    CASE
        WHEN latency_ms IS NULL THEN NULL
        WHEN latency_ms <= 0 THEN -1
        ELSE FLOOR((LN(GREATEST(latency_ms, 1)::DOUBLE PRECISION) / LN(2.0)) * buckets_per_power_of_two)::INTEGER
    END
$$;

-- Sparse per-minute latency histogram (stores only non-empty buckets).
CREATE TABLE IF NOT EXISTS tensorzero.model_latency_histogram_minute (
    model_name TEXT NOT NULL,
    minute TIMESTAMPTZ NOT NULL,
    metric TEXT NOT NULL CHECK (metric IN ('response_time_ms', 'ttft_ms')),
    bucket_id INTEGER NOT NULL,
    bucket_count BIGINT NOT NULL CHECK (bucket_count > 0),
    PRIMARY KEY (model_name, minute, metric, bucket_id)
);

CREATE INDEX IF NOT EXISTS idx_model_latency_histogram_minute_window
    ON tensorzero.model_latency_histogram_minute(minute, metric, model_name, bucket_id);

-- Sparse per-hour rollup.
CREATE TABLE IF NOT EXISTS tensorzero.model_latency_histogram_hour (
    model_name TEXT NOT NULL,
    hour TIMESTAMPTZ NOT NULL,
    metric TEXT NOT NULL CHECK (metric IN ('response_time_ms', 'ttft_ms')),
    bucket_id INTEGER NOT NULL,
    bucket_count BIGINT NOT NULL CHECK (bucket_count > 0),
    PRIMARY KEY (model_name, hour, metric, bucket_id)
);

CREATE INDEX IF NOT EXISTS idx_model_latency_histogram_hour_window
    ON tensorzero.model_latency_histogram_hour(hour, metric, model_name, bucket_id);

-- Refresh watermarks.
CREATE TABLE IF NOT EXISTS tensorzero.model_latency_histogram_minute_refresh_state (
    singleton BOOLEAN PRIMARY KEY DEFAULT TRUE CHECK (singleton),
    last_processed_created_at TIMESTAMPTZ,
    last_processed_id UUID
);

CREATE TABLE IF NOT EXISTS tensorzero.model_latency_histogram_hour_refresh_state (
    singleton BOOLEAN PRIMARY KEY DEFAULT TRUE CHECK (singleton),
    last_processed_minute TIMESTAMPTZ
);

-- Initialize minute watermark at the current tail to avoid full-table first refresh.
INSERT INTO tensorzero.model_latency_histogram_minute_refresh_state (
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

-- Initialize hour watermark at current tail.
INSERT INTO tensorzero.model_latency_histogram_hour_refresh_state (
    singleton,
    last_processed_minute
)
SELECT
    TRUE,
    latest.minute
FROM (
    SELECT minute
    FROM tensorzero.model_latency_histogram_minute
    ORDER BY minute DESC
    LIMIT 1
) AS latest
ON CONFLICT (singleton) DO NOTHING;

-- Incremental refresh for minute histogram.
CREATE OR REPLACE FUNCTION tensorzero.refresh_model_latency_histogram_minute_incremental(
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
    refresh_from_minute TIMESTAMPTZ;
    refresh_to_minute TIMESTAMPTZ;
    newest_created_at TIMESTAMPTZ;
    newest_id UUID;
    oldest_created_at TIMESTAMPTZ;
BEGIN
    INSERT INTO tensorzero.model_latency_histogram_minute_refresh_state (singleton)
    VALUES (TRUE)
    ON CONFLICT (singleton) DO NOTHING;

    SELECT last_processed_created_at
    INTO state_created_at
    FROM tensorzero.model_latency_histogram_minute_refresh_state
    WHERE singleton = TRUE
    FOR UPDATE;

    IF full_refresh THEN
        TRUNCATE TABLE tensorzero.model_latency_histogram_minute;
        refresh_from := TIMESTAMPTZ '-infinity';
    ELSIF state_created_at IS NULL THEN
        refresh_from := TIMESTAMPTZ '-infinity';
    ELSE
        refresh_from := date_trunc('minute', state_created_at - lookback);
    END IF;

    refresh_from_minute := date_trunc('minute', refresh_from);
    refresh_to_minute := date_trunc('minute', refresh_to);

    -- Recompute affected minute buckets in trailing window.
    DELETE FROM tensorzero.model_latency_histogram_minute
    WHERE minute >= refresh_from_minute
      AND minute <= refresh_to_minute;

    INSERT INTO tensorzero.model_latency_histogram_minute (
        model_name,
        minute,
        metric,
        bucket_id,
        bucket_count
    )
    SELECT
        model_name,
        minute,
        metric,
        bucket_id,
        COUNT(*)::BIGINT AS bucket_count
    FROM (
        SELECT
            mi.model_name,
            date_trunc('minute', mi.created_at) AS minute,
            value.metric,
            value.bucket_id
        FROM tensorzero.model_inferences mi
        CROSS JOIN LATERAL (
            VALUES
                (
                    'response_time_ms'::TEXT,
                    tensorzero.latency_histogram_bucket_id(mi.response_time_ms)
                ),
                (
                    'ttft_ms'::TEXT,
                    tensorzero.latency_histogram_bucket_id(mi.ttft_ms)
                )
        ) AS value(metric, bucket_id)
        WHERE mi.created_at >= refresh_from
          AND mi.created_at <= refresh_to
          AND value.bucket_id IS NOT NULL
    ) source
    GROUP BY model_name, minute, metric, bucket_id
    ON CONFLICT (model_name, minute, metric, bucket_id) DO UPDATE
    SET bucket_count = EXCLUDED.bucket_count;

    -- Keep retention behavior correct.
    SELECT MIN(created_at)
    INTO oldest_created_at
    FROM tensorzero.model_inferences;

    IF oldest_created_at IS NULL THEN
        TRUNCATE TABLE tensorzero.model_latency_histogram_minute;
    ELSE
        DELETE FROM tensorzero.model_latency_histogram_minute
        WHERE minute < date_trunc('minute', oldest_created_at);
    END IF;

    SELECT created_at, id
    INTO newest_created_at, newest_id
    FROM tensorzero.model_inferences
    ORDER BY created_at DESC, id DESC
    LIMIT 1;

    UPDATE tensorzero.model_latency_histogram_minute_refresh_state
    SET
        last_processed_created_at = newest_created_at,
        last_processed_id = newest_id
    WHERE singleton = TRUE;
END;
$$;

-- Incremental refresh for hour histogram.
CREATE OR REPLACE FUNCTION tensorzero.refresh_model_latency_histogram_hour_incremental(
    lookback INTERVAL DEFAULT INTERVAL '2 hours',
    full_refresh BOOLEAN DEFAULT FALSE
)
RETURNS VOID
LANGUAGE plpgsql
AS $$
DECLARE
    state_minute TIMESTAMPTZ;
    refresh_from TIMESTAMPTZ;
    refresh_to TIMESTAMPTZ := date_trunc('hour', NOW());
    newest_minute TIMESTAMPTZ;
    oldest_minute TIMESTAMPTZ;
BEGIN
    INSERT INTO tensorzero.model_latency_histogram_hour_refresh_state (singleton)
    VALUES (TRUE)
    ON CONFLICT (singleton) DO NOTHING;

    SELECT last_processed_minute
    INTO state_minute
    FROM tensorzero.model_latency_histogram_hour_refresh_state
    WHERE singleton = TRUE
    FOR UPDATE;

    IF full_refresh THEN
        TRUNCATE TABLE tensorzero.model_latency_histogram_hour;
        refresh_from := TIMESTAMPTZ '-infinity';
    ELSIF state_minute IS NULL THEN
        refresh_from := TIMESTAMPTZ '-infinity';
    ELSE
        refresh_from := date_trunc('hour', state_minute - lookback);
    END IF;

    DELETE FROM tensorzero.model_latency_histogram_hour
    WHERE hour >= refresh_from
      AND hour <= refresh_to;

    INSERT INTO tensorzero.model_latency_histogram_hour (
        model_name,
        hour,
        metric,
        bucket_id,
        bucket_count
    )
    SELECT
        model_name,
        date_trunc('hour', minute) AS hour,
        metric,
        bucket_id,
        SUM(bucket_count)::BIGINT AS bucket_count
    FROM tensorzero.model_latency_histogram_minute
    WHERE minute >= refresh_from
      AND minute < refresh_to + INTERVAL '1 hour'
    GROUP BY model_name, date_trunc('hour', minute), metric, bucket_id
    ON CONFLICT (model_name, hour, metric, bucket_id) DO UPDATE
    SET bucket_count = EXCLUDED.bucket_count;

    SELECT MIN(minute)
    INTO oldest_minute
    FROM tensorzero.model_latency_histogram_minute;

    IF oldest_minute IS NULL THEN
        TRUNCATE TABLE tensorzero.model_latency_histogram_hour;
    ELSE
        DELETE FROM tensorzero.model_latency_histogram_hour
        WHERE hour < date_trunc('hour', oldest_minute);
    END IF;

    SELECT minute
    INTO newest_minute
    FROM tensorzero.model_latency_histogram_minute
    ORDER BY minute DESC
    LIMIT 1;

    UPDATE tensorzero.model_latency_histogram_hour_refresh_state
    SET last_processed_minute = newest_minute
    WHERE singleton = TRUE;
END;
$$;

-- Drop old materialized views (if they still exist as materialized views).
DO $$
BEGIN
    IF EXISTS (
        SELECT 1
        FROM pg_matviews
        WHERE schemaname = 'tensorzero'
          AND matviewname = 'model_latency_quantiles'
    ) THEN
        DROP MATERIALIZED VIEW tensorzero.model_latency_quantiles;
    END IF;

    IF EXISTS (
        SELECT 1
        FROM pg_matviews
        WHERE schemaname = 'tensorzero'
          AND matviewname = 'model_latency_quantiles_hour'
    ) THEN
        DROP MATERIALIZED VIEW tensorzero.model_latency_quantiles_hour;
    END IF;

    IF EXISTS (
        SELECT 1
        FROM pg_matviews
        WHERE schemaname = 'tensorzero'
          AND matviewname = 'model_latency_quantiles_day'
    ) THEN
        DROP MATERIALIZED VIEW tensorzero.model_latency_quantiles_day;
    END IF;

    IF EXISTS (
        SELECT 1
        FROM pg_matviews
        WHERE schemaname = 'tensorzero'
          AND matviewname = 'model_latency_quantiles_week'
    ) THEN
        DROP MATERIALIZED VIEW tensorzero.model_latency_quantiles_week;
    END IF;

    IF EXISTS (
        SELECT 1
        FROM pg_matviews
        WHERE schemaname = 'tensorzero'
          AND matviewname = 'model_latency_quantiles_month'
    ) THEN
        DROP MATERIALIZED VIEW tensorzero.model_latency_quantiles_month;
    END IF;
END $$;
