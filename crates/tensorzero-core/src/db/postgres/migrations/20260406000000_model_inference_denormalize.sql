ALTER TABLE tensorzero.model_inferences ADD COLUMN IF NOT EXISTS function_name TEXT NOT NULL DEFAULT '';
ALTER TABLE tensorzero.model_inferences ADD COLUMN IF NOT EXISTS variant_name TEXT NOT NULL DEFAULT '';

UPDATE tensorzero.model_inferences mi
SET function_name = ci.function_name, variant_name = ci.variant_name
FROM (
    SELECT id, function_name, variant_name FROM tensorzero.chat_inferences
    UNION ALL
    SELECT id, function_name, variant_name FROM tensorzero.json_inferences
) ci
WHERE mi.inference_id = ci.id AND mi.function_name = '';
