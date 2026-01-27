-- Change tool_choice column from TEXT to JSONB
-- Converting existing text values to JSONB (e.g., '"auto"' or '{"specific":"tool_name"}')
ALTER TABLE tensorzero.chat_inferences
    ALTER COLUMN tool_choice TYPE JSONB USING tool_choice::jsonb;
