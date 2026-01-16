-- Extensions for observability features
CREATE EXTENSION IF NOT EXISTS pg_trgm;  -- Substring search (ILIKE with GIN)
-- Note: pgvector extension will be created in step-6 with DICL tables

-- Retention configuration table
-- Gateway writes retention_days on startup; pg_cron jobs read it
CREATE TABLE IF NOT EXISTS tensorzero_retention_config (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Partition management: create future partitions for a given table
-- Called by pg_cron daily or manually, or when creating new partitioned tables
CREATE OR REPLACE FUNCTION tensorzero_create_partitions(p_table_name TEXT)
RETURNS void AS $$
DECLARE
    partition_date DATE;
    partition_name TEXT;
BEGIN
    FOR i IN 0..7 LOOP
        partition_date := CURRENT_DATE + i;
        partition_name := p_table_name || '_' || to_char(partition_date, 'YYYY_MM_DD');
        IF NOT EXISTS (
            SELECT 1 FROM pg_tables WHERE tablename = partition_name
        ) THEN
            EXECUTE format(
                'CREATE TABLE IF NOT EXISTS %I PARTITION OF %I FOR VALUES FROM (%L) TO (%L)',
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
CREATE OR REPLACE FUNCTION tensorzero_drop_old_partitions(p_table_name TEXT)
RETURNS void AS $$
DECLARE
    retention_days INT;
    cutoff_date DATE;
    partition_record RECORD;
    pattern TEXT;
BEGIN
    SELECT value::INT INTO retention_days
    FROM tensorzero_retention_config
    WHERE key = 'retention_days';

    IF retention_days IS NULL THEN
        RAISE NOTICE 'retention_days not configured, skipping partition cleanup';
        RETURN;
    END IF;

    cutoff_date := CURRENT_DATE - retention_days;
    pattern := '^' || p_table_name || '_\d{4}_\d{2}_\d{2}$';

    FOR partition_record IN
        SELECT tablename
        FROM pg_tables
        WHERE tablename ~ pattern
    LOOP
        IF to_date(substring(partition_record.tablename from '\d{4}_\d{2}_\d{2}$'), 'YYYY_MM_DD') < cutoff_date THEN
            EXECUTE format('DROP TABLE %I', partition_record.tablename);
        END IF;
    END LOOP;
END;
$$ LANGUAGE plpgsql;
