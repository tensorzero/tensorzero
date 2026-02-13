-- chat_inference_io: generated tsvector columns + GIN indexes
ALTER TABLE tensorzero.chat_inference_io
    ADD COLUMN input_tsvector tsvector
        GENERATED ALWAYS AS (jsonb_to_tsvector('simple', input, '["string"]'::jsonb)) STORED,
    ADD COLUMN output_tsvector tsvector
        GENERATED ALWAYS AS (jsonb_to_tsvector('simple', output, '["string"]'::jsonb)) STORED;

CREATE INDEX idx_chat_inference_io_input_tsvector
    ON tensorzero.chat_inference_io USING GIN (input_tsvector);
CREATE INDEX idx_chat_inference_io_output_tsvector
    ON tensorzero.chat_inference_io USING GIN (output_tsvector);

-- json_inference_io: generated tsvector columns + GIN indexes
ALTER TABLE tensorzero.json_inference_io
    ADD COLUMN input_tsvector tsvector
        GENERATED ALWAYS AS (jsonb_to_tsvector('simple', input, '["string"]'::jsonb)) STORED,
    ADD COLUMN output_tsvector tsvector
        GENERATED ALWAYS AS (jsonb_to_tsvector('simple', output, '["string"]'::jsonb)) STORED;

CREATE INDEX idx_json_inference_io_input_tsvector
    ON tensorzero.json_inference_io USING GIN (input_tsvector);
CREATE INDEX idx_json_inference_io_output_tsvector
    ON tensorzero.json_inference_io USING GIN (output_tsvector);

-- chat_datapoints: generated tsvector columns + GIN indexes
ALTER TABLE tensorzero.chat_datapoints
    ADD COLUMN input_tsvector tsvector
        GENERATED ALWAYS AS (jsonb_to_tsvector('simple', input, '["string"]'::jsonb)) STORED,
    ADD COLUMN output_tsvector tsvector
        GENERATED ALWAYS AS (jsonb_to_tsvector('simple', output, '["string"]'::jsonb)) STORED;

CREATE INDEX idx_chat_datapoints_input_tsvector
    ON tensorzero.chat_datapoints USING GIN (input_tsvector);
CREATE INDEX idx_chat_datapoints_output_tsvector
    ON tensorzero.chat_datapoints USING GIN (output_tsvector);

-- json_datapoints: generated tsvector columns + GIN indexes
ALTER TABLE tensorzero.json_datapoints
    ADD COLUMN input_tsvector tsvector
        GENERATED ALWAYS AS (jsonb_to_tsvector('simple', input, '["string"]'::jsonb)) STORED,
    ADD COLUMN output_tsvector tsvector
        GENERATED ALWAYS AS (jsonb_to_tsvector('simple', output, '["string"]'::jsonb)) STORED;

CREATE INDEX idx_json_datapoints_input_tsvector
    ON tensorzero.json_datapoints USING GIN (input_tsvector);
CREATE INDEX idx_json_datapoints_output_tsvector
    ON tensorzero.json_datapoints USING GIN (output_tsvector);
