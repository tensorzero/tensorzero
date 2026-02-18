-- Add cost column to model_inferences.
-- Cost is stored in dollars as NUMERIC(18, 9) — 18 total digits, 9 fractional.
-- NULL means cost tracking was not configured when this inference was made.
ALTER TABLE tensorzero.model_inferences ADD COLUMN IF NOT EXISTS cost NUMERIC(18, 9);
