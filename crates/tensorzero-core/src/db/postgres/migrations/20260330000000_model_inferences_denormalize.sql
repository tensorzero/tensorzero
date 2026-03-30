-- Denormalize function_name and variant_name onto model_inferences.
-- This eliminates the need for JOINs when aggregating per-variant metrics.
-- Historical rows will have empty strings ('') as default values.

ALTER TABLE tensorzero.model_inferences
    ADD COLUMN IF NOT EXISTS function_name TEXT NOT NULL DEFAULT '',
    ADD COLUMN IF NOT EXISTS variant_name TEXT NOT NULL DEFAULT '';
