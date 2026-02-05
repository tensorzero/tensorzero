-- Materialized view for model provider statistics aggregation
-- This view pre-aggregates model_inferences data by minute for faster dashboard queries.

CREATE MATERIALIZED VIEW tensorzero.model_provider_statistics AS
SELECT
    model_name,
    model_provider_name,
    date_trunc('minute', created_at) AS minute,
    SUM(input_tokens)::BIGINT AS total_input_tokens,
    SUM(output_tokens)::BIGINT AS total_output_tokens,
    COUNT(*)::BIGINT AS inference_count
FROM tensorzero.model_inferences
GROUP BY model_name, model_provider_name, date_trunc('minute', created_at);

-- Unique index required for REFRESH MATERIALIZED VIEW CONCURRENTLY
CREATE UNIQUE INDEX idx_model_provider_stats_pk
    ON tensorzero.model_provider_statistics(model_name, model_provider_name, minute);

-- Additional index for time-based queries
CREATE INDEX idx_model_provider_stats_minute
    ON tensorzero.model_provider_statistics(minute);

-- Materialized views for model latency quantiles
-- These precompute latency percentiles per model for dashboard queries.
-- Quantiles: 0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.15, 0.25, 0.5, 0.75, 0.85, 0.9, 0.95, 0.98, 0.99, 0.995, 0.999
--
-- We create separate views for different time windows since quantiles cannot be aggregated.
-- Minute queries compute from raw data since the data volume is small.

-- Cumulative latency quantiles (all time)
CREATE MATERIALIZED VIEW tensorzero.model_latency_quantiles AS
SELECT
    model_name,
    COALESCE(
        percentile_cont(ARRAY[0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.15, 0.25, 0.5, 0.75, 0.85, 0.9, 0.95, 0.98, 0.99, 0.995, 0.999])
            WITHIN GROUP (ORDER BY response_time_ms),
        array_fill(NULL::double precision, ARRAY[17])
    ) AS response_time_ms_quantiles,
    COALESCE(
        percentile_cont(ARRAY[0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.15, 0.25, 0.5, 0.75, 0.85, 0.9, 0.95, 0.98, 0.99, 0.995, 0.999])
            WITHIN GROUP (ORDER BY ttft_ms),
        array_fill(NULL::double precision, ARRAY[17])
    ) AS ttft_ms_quantiles,
    COUNT(*)::BIGINT AS count
FROM tensorzero.model_inferences
GROUP BY model_name;

CREATE UNIQUE INDEX idx_model_latency_quantiles_pk
    ON tensorzero.model_latency_quantiles(model_name);

-- Last hour latency quantiles (rolling 1 hour)
CREATE MATERIALIZED VIEW tensorzero.model_latency_quantiles_hour AS
SELECT
    model_name,
    COALESCE(
        percentile_cont(ARRAY[0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.15, 0.25, 0.5, 0.75, 0.85, 0.9, 0.95, 0.98, 0.99, 0.995, 0.999])
            WITHIN GROUP (ORDER BY response_time_ms),
        array_fill(NULL::double precision, ARRAY[17])
    ) AS response_time_ms_quantiles,
    COALESCE(
        percentile_cont(ARRAY[0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.15, 0.25, 0.5, 0.75, 0.85, 0.9, 0.95, 0.98, 0.99, 0.995, 0.999])
            WITHIN GROUP (ORDER BY ttft_ms),
        array_fill(NULL::double precision, ARRAY[17])
    ) AS ttft_ms_quantiles,
    COUNT(*)::BIGINT AS count
FROM tensorzero.model_inferences
WHERE created_at >= NOW() - INTERVAL '1 hour'
GROUP BY model_name;

CREATE UNIQUE INDEX idx_model_latency_quantiles_hour_pk
    ON tensorzero.model_latency_quantiles_hour(model_name);

-- Last day latency quantiles (rolling 24 hours)
CREATE MATERIALIZED VIEW tensorzero.model_latency_quantiles_day AS
SELECT
    model_name,
    COALESCE(
        percentile_cont(ARRAY[0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.15, 0.25, 0.5, 0.75, 0.85, 0.9, 0.95, 0.98, 0.99, 0.995, 0.999])
            WITHIN GROUP (ORDER BY response_time_ms),
        array_fill(NULL::double precision, ARRAY[17])
    ) AS response_time_ms_quantiles,
    COALESCE(
        percentile_cont(ARRAY[0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.15, 0.25, 0.5, 0.75, 0.85, 0.9, 0.95, 0.98, 0.99, 0.995, 0.999])
            WITHIN GROUP (ORDER BY ttft_ms),
        array_fill(NULL::double precision, ARRAY[17])
    ) AS ttft_ms_quantiles,
    COUNT(*)::BIGINT AS count
FROM tensorzero.model_inferences
WHERE created_at >= NOW() - INTERVAL '1 day'
GROUP BY model_name;

CREATE UNIQUE INDEX idx_model_latency_quantiles_day_pk
    ON tensorzero.model_latency_quantiles_day(model_name);

-- Last week latency quantiles (rolling 7 days)
CREATE MATERIALIZED VIEW tensorzero.model_latency_quantiles_week AS
SELECT
    model_name,
    COALESCE(
        percentile_cont(ARRAY[0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.15, 0.25, 0.5, 0.75, 0.85, 0.9, 0.95, 0.98, 0.99, 0.995, 0.999])
            WITHIN GROUP (ORDER BY response_time_ms),
        array_fill(NULL::double precision, ARRAY[17])
    ) AS response_time_ms_quantiles,
    COALESCE(
        percentile_cont(ARRAY[0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.15, 0.25, 0.5, 0.75, 0.85, 0.9, 0.95, 0.98, 0.99, 0.995, 0.999])
            WITHIN GROUP (ORDER BY ttft_ms),
        array_fill(NULL::double precision, ARRAY[17])
    ) AS ttft_ms_quantiles,
    COUNT(*)::BIGINT AS count
FROM tensorzero.model_inferences
WHERE created_at >= NOW() - INTERVAL '1 week'
GROUP BY model_name;

CREATE UNIQUE INDEX idx_model_latency_quantiles_week_pk
    ON tensorzero.model_latency_quantiles_week(model_name);

-- Last month latency quantiles (rolling 30 days)
CREATE MATERIALIZED VIEW tensorzero.model_latency_quantiles_month AS
SELECT
    model_name,
    COALESCE(
        percentile_cont(ARRAY[0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.15, 0.25, 0.5, 0.75, 0.85, 0.9, 0.95, 0.98, 0.99, 0.995, 0.999])
            WITHIN GROUP (ORDER BY response_time_ms),
        array_fill(NULL::double precision, ARRAY[17])
    ) AS response_time_ms_quantiles,
    COALESCE(
        percentile_cont(ARRAY[0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.15, 0.25, 0.5, 0.75, 0.85, 0.9, 0.95, 0.98, 0.99, 0.995, 0.999])
            WITHIN GROUP (ORDER BY ttft_ms),
        array_fill(NULL::double precision, ARRAY[17])
    ) AS ttft_ms_quantiles,
    COUNT(*)::BIGINT AS count
FROM tensorzero.model_inferences
WHERE created_at >= NOW() - INTERVAL '1 month'
GROUP BY model_name;

CREATE UNIQUE INDEX idx_model_latency_quantiles_month_pk
    ON tensorzero.model_latency_quantiles_month(model_name);
