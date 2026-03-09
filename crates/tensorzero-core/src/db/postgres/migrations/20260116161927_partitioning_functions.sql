-- Schema wrapping all tensorzero constructs
CREATE SCHEMA IF NOT EXISTS tensorzero;

-- Retention configuration table
-- Gateway writes retention_days on startup; pg_cron jobs read it
CREATE TABLE IF NOT EXISTS tensorzero.retention_config (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Partition management: create future partitions for a given table
-- Called by pg_cron daily or manually, or when creating new partitioned tables
-- p_table_name should be unqualified (e.g., 'chat_inferences'), tables are created in tensorzero schema
CREATE OR REPLACE FUNCTION tensorzero.create_partitions(p_table_name TEXT)
RETURNS void AS $$
DECLARE
    partition_date DATE;
    partition_name TEXT;
BEGIN
    FOR i IN 0..7 LOOP
        partition_date := CURRENT_DATE + i;
        partition_name := p_table_name || '_' || to_char(partition_date, 'YYYY_MM_DD');
        IF NOT EXISTS (
            SELECT 1 FROM pg_tables WHERE schemaname = 'tensorzero' AND tablename = partition_name
        ) THEN
            EXECUTE format(
                'CREATE TABLE IF NOT EXISTS tensorzero.%I PARTITION OF tensorzero.%I FOR VALUES FROM (%L) TO (%L)',
                partition_name,
                p_table_name,
                partition_date,
                partition_date + 1
            );
        END IF;
    END LOOP;
END;
$$ LANGUAGE plpgsql;

-- Partition management: drop old partitions for a given table
-- Called by pg_cron daily or manually
-- p_retention_key: key in tensorzero.retention_config (e.g., 'inference_retention_days')
CREATE OR REPLACE FUNCTION tensorzero.drop_old_partitions(p_table_name TEXT, p_retention_key TEXT)
RETURNS void AS $$
DECLARE
    retention_days INT;
    cutoff_date DATE;
    partition_record RECORD;
    pattern TEXT;
BEGIN
    SELECT value::INT INTO retention_days
    FROM tensorzero.retention_config
    WHERE key = p_retention_key;

    IF retention_days IS NULL THEN
        RAISE NOTICE '% not configured, skipping partition cleanup for %', p_retention_key, p_table_name;
        RETURN;
    END IF;

    cutoff_date := CURRENT_DATE - retention_days;
    pattern := '^' || p_table_name || '_\d{4}_\d{2}_\d{2}$';

    FOR partition_record IN
        SELECT tablename
        FROM pg_tables
        WHERE schemaname = 'tensorzero' AND tablename ~ pattern
    LOOP
        IF to_date(substring(partition_record.tablename from '\d{4}_\d{2}_\d{2}$'), 'YYYY_MM_DD') < cutoff_date THEN
            EXECUTE format('DROP TABLE tensorzero.%I', partition_record.tablename);
        END IF;
    END LOOP;
END;
$$ LANGUAGE plpgsql;
