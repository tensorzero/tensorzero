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
        -- Inference metadata tables use monthly partitions; IO tables use daily partitions
        PERFORM cron.schedule(
            'tensorzero_create_inference_partitions',
            '5 0 * * *',
            $$
            SELECT tensorzero.create_monthly_partitions('chat_inferences');
            SELECT tensorzero.create_monthly_partitions('json_inferences');
            SELECT tensorzero.create_monthly_partitions('model_inferences');
            SELECT tensorzero.create_monthly_partitions('batch_requests');
            SELECT tensorzero.create_monthly_partitions('batch_model_inferences');
            SELECT tensorzero.create_partitions('chat_inference_data');
            SELECT tensorzero.create_partitions('json_inference_data');
            SELECT tensorzero.create_partitions('model_inference_data');
            SELECT tensorzero.create_partitions('batch_request_data');
            SELECT tensorzero.create_partitions('batch_model_inference_data');
            $$
        );

        -- Drop old metadata partitions daily at 00:30 UTC (only acts if retention is configured)
        PERFORM cron.schedule(
            'tensorzero_drop_old_inference_metadata_partitions',
            '30 0 * * *',
            $$
            SELECT tensorzero.drop_old_monthly_partitions('chat_inferences', 'inference_metadata_retention_days');
            SELECT tensorzero.drop_old_monthly_partitions('json_inferences', 'inference_metadata_retention_days');
            SELECT tensorzero.drop_old_monthly_partitions('model_inferences', 'inference_metadata_retention_days');
            SELECT tensorzero.drop_old_monthly_partitions('batch_requests', 'inference_metadata_retention_days');
            SELECT tensorzero.drop_old_monthly_partitions('batch_model_inferences', 'inference_metadata_retention_days');
            $$
        );

        -- Drop old data partitions daily at 00:35 UTC (only acts if retention is configured)
        PERFORM cron.schedule(
            'tensorzero_drop_old_inference_data_partitions',
            '35 0 * * *',
            $$
            SELECT tensorzero.drop_old_partitions('chat_inference_data', 'inference_data_retention_days');
            SELECT tensorzero.drop_old_partitions('json_inference_data', 'inference_data_retention_days');
            SELECT tensorzero.drop_old_partitions('model_inference_data', 'inference_data_retention_days');
            SELECT tensorzero.drop_old_partitions('batch_request_data', 'inference_data_retention_days');
            SELECT tensorzero.drop_old_partitions('batch_model_inference_data', 'inference_data_retention_days');
            $$
        );

        -- Remove the legacy combined job if it exists
        PERFORM cron.unschedule(jobid)
        FROM cron.job
        WHERE jobname = 'tensorzero_drop_old_inference_partitions';

        -- Incrementally refresh model provider statistics every 5 minutes, with a 10 minute lookback window
        PERFORM cron.schedule(
            'tensorzero_refresh_model_provider_statistics_incremental',
            '*/5 * * * *',
            $$
            SELECT tensorzero.refresh_model_provider_statistics_incremental(INTERVAL '10 minutes');
            $$
        );

        -- Ensure legacy materialized view refresh job is removed after migration to incremental histogram refresh.
        PERFORM cron.unschedule(jobid)
        FROM cron.job
        WHERE jobname = 'tensorzero_refresh_materialized_views';

        -- Ensure legacy wrapper histogram refresh job is removed.
        PERFORM cron.unschedule(jobid)
        FROM cron.job
        WHERE jobname = 'tensorzero_refresh_model_latency_histograms_incremental';

        -- Incrementally refresh minute latency histogram rollups every 5 minutes.
        PERFORM cron.schedule(
            'tensorzero_refresh_model_latency_histogram_minute_incremental',
            '*/5 * * * *',
            $$
            SELECT tensorzero.refresh_model_latency_histogram_minute_incremental(INTERVAL '10 minutes');
            $$
        );

        -- Incrementally refresh hour latency histogram rollups every 5 minutes, offset after minute refresh.
        PERFORM cron.schedule(
            'tensorzero_refresh_model_latency_histogram_hour_incremental',
            '2-59/5 * * * *',
            $$
            SELECT tensorzero.refresh_model_latency_histogram_hour_incremental(INTERVAL '2 hours');
            $$
        );

        RAISE NOTICE 'pg_cron jobs scheduled for partition management, incremental stats refresh, and separate minute/hour latency histogram refresh';
    ELSE
        RAISE WARNING 'pg_cron extension not available';
    END IF;
END $cron_setup$;
