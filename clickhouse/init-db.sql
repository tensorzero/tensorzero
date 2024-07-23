-- Create the tensorzero database if it doesn't exist
CREATE DATABASE IF NOT EXISTS tensorzero;

-- Switch to the tensorzero database
USE tensorzero;

-- BooleanMetricFeedback table
CREATE TABLE IF NOT EXISTS BooleanMetricFeedback
(
    id UUID DEFAULT generateUUIDv7(),
    target_id UUID,
    metric_name LowCardinality(String),
    value Bool
) ENGINE = MergeTree()
ORDER BY (metric_name, target_id);

-- FloatMetricFeedback table
CREATE TABLE IF NOT EXISTS FloatMetricFeedback
(
    id UUID DEFAULT generateUUIDv7(),
    target_id UUID,
    metric_name LowCardinality(String),
    value Float32
) ENGINE = MergeTree()
ORDER BY (metric_name, target_id);

-- DemonstrationFeedback table
CREATE TABLE IF NOT EXISTS DemonstrationFeedback
(
    id UUID DEFAULT generateUUIDv7(),
    inference_id UUID,
    value String
) ENGINE = MergeTree()
ORDER BY inference_id;

-- CommentFeedback table
CREATE TABLE IF NOT EXISTS CommentFeedback
(
    id UUID DEFAULT generateUUIDv7(),
    target_id UUID,
    value String
) ENGINE = MergeTree()
ORDER BY target_id;

-- Inference table
CREATE TABLE IF NOT EXISTS Inference
(
    id UUID,
    function_name LowCardinality(String),
    variant_name LowCardinality(String),
    episode_id UUID,
    input String,
    output String
) ENGINE = MergeTree()
ORDER BY (function_name, variant_name, episode_id);

-- ModelInference table
CREATE TABLE IF NOT EXISTS ModelInference
(
    id UUID DEFAULT generateUUIDv7(),
    inference_id UUID,
    input String,
    output String,
    raw_response String,
    input_tokens UInt32,
    output_tokens UInt32
) ENGINE = MergeTree()
ORDER BY inference_id;
