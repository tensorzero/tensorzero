-- Idempotent setup script for pg_cron extension and partition management jobs.
-- This is NOT a migration - it runs every time migrations are run.

-- Try to create pg_cron extension (catches errors if not available)
-- TODO(#6176): Promote this to a migration and require pg_cron extension to be installed.
DO $$
BEGIN
    CREATE EXTENSION IF NOT EXISTS pg_cron;
    RAISE NOTICE 'pg_cron extension created or already exists';
EXCEPTION
    WHEN OTHERS THEN
        RAISE NOTICE 'pg_cron not enabled: % (%)', SQLERRM, SQLSTATE;
END $$;

-- Schedule pg_cron jobs for partition management if pg_cron is available
DO $cron_setup$
BEGIN
    -- Check if pg_cron extension is available
    IF EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'pg_cron') THEN
        -- Create future partitions daily at 00:05 UTC
        PERFORM cron.schedule(
            'tensorzero_create_inference_partitions',
            '5 0 * * *',
            $$
            SELECT tensorzero.create_partitions('chat_inferences');
            SELECT tensorzero.create_partitions('json_inferences');
            SELECT tensorzero.create_partitions('model_inferences');
            $$
        );

        -- Drop old partitions daily at 00:30 UTC (only acts if retention is configured)
        PERFORM cron.schedule(
            'tensorzero_drop_old_inference_partitions',
            '30 0 * * *',
            $$
            SELECT tensorzero.drop_old_partitions('chat_inferences', 'inference_retention_days');
            SELECT tensorzero.drop_old_partitions('json_inferences', 'inference_retention_days');
            SELECT tensorzero.drop_old_partitions('model_inferences', 'inference_retention_days');
            $$
        );

        -- Refresh materialized views every 5 minutes
        PERFORM cron.schedule(
            'tensorzero_refresh_materialized_views',
            '*/5 * * * *',
            $$
            REFRESH MATERIALIZED VIEW tensorzero.model_provider_statistics;
            REFRESH MATERIALIZED VIEW tensorzero.model_latency_quantiles;
            REFRESH MATERIALIZED VIEW tensorzero.model_latency_quantiles_hour;
            REFRESH MATERIALIZED VIEW tensorzero.model_latency_quantiles_day;
            REFRESH MATERIALIZED VIEW tensorzero.model_latency_quantiles_week;
            REFRESH MATERIALIZED VIEW tensorzero.model_latency_quantiles_month;
            $$
        );

        RAISE NOTICE 'pg_cron jobs scheduled for partition management and materialized view refresh';
    ELSE
        RAISE WARNING 'pg_cron extension not available';
    END IF;
END $cron_setup$;
