-- Adds OTel GenAI fields to model_inferences:
--   provider_response_id: provider-native response id (e.g. OpenAI chatcmpl-...)
--   response_model_name: actual model returned by the provider
--   operation: gen_ai.operation.name (chat, embeddings, etc.)

ALTER TABLE tensorzero.model_inferences ADD COLUMN IF NOT EXISTS provider_response_id TEXT;
ALTER TABLE tensorzero.model_inferences ADD COLUMN IF NOT EXISTS response_model_name TEXT;
ALTER TABLE tensorzero.model_inferences ADD COLUMN IF NOT EXISTS operation TEXT;
