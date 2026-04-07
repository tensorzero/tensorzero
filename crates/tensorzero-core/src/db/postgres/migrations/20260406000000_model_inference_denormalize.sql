-- Denormalize function_name/variant_name onto model_inferences.
-- These columns are needed so that ClickHouse materialized views on
-- ModelInference can be self-contained (no JOIN with InferenceById).
-- For Postgres, we also store them for query convenience.
--
-- No backfill of historical rows — new inserts populate these columns
-- via application code. Old rows remain with the default empty string.

ALTER TABLE tensorzero.model_inferences ADD COLUMN IF NOT EXISTS function_name TEXT NOT NULL DEFAULT '';
ALTER TABLE tensorzero.model_inferences ADD COLUMN IF NOT EXISTS variant_name TEXT NOT NULL DEFAULT '';
