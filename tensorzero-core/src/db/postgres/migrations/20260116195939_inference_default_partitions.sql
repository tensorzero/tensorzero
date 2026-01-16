-- Default partitions for backfilling historical data
-- These catch any rows that don't fit into the date-based partitions

CREATE TABLE chat_inferences_default PARTITION OF chat_inferences DEFAULT;
CREATE TABLE json_inferences_default PARTITION OF json_inferences DEFAULT;
