-- Denormalize function_name/variant_name onto model_inferences.
-- These columns are needed so that ClickHouse materialized views on
-- ModelInference can be self-contained (no JOIN with InferenceById).
-- For Postgres, we also store them for query convenience.

ALTER TABLE tensorzero.model_inferences ADD COLUMN IF NOT EXISTS function_name TEXT NOT NULL DEFAULT '';
ALTER TABLE tensorzero.model_inferences ADD COLUMN IF NOT EXISTS variant_name TEXT NOT NULL DEFAULT '';

-- Backfill in batches to avoid holding row-level locks on the entire table
-- for the duration of a single large UPDATE. Each batch updates up to 10,000
-- rows at a time, which keeps lock duration short and avoids bloating WAL.
DO $$
DECLARE
    rows_updated INT;
BEGIN
    LOOP
        UPDATE tensorzero.model_inferences mi
        SET function_name = ci.function_name, variant_name = ci.variant_name
        FROM (
            SELECT id, function_name, variant_name FROM tensorzero.chat_inferences
            UNION ALL
            SELECT id, function_name, variant_name FROM tensorzero.json_inferences
        ) ci
        WHERE mi.inference_id = ci.id
            AND mi.function_name = ''
            AND mi.id IN (
                SELECT id FROM tensorzero.model_inferences
                WHERE function_name = ''
                LIMIT 10000
            );
        GET DIAGNOSTICS rows_updated = ROW_COUNT;
        EXIT WHEN rows_updated = 0;
        RAISE NOTICE 'Backfilled % model_inferences rows', rows_updated;
    END LOOP;
END;
$$;
