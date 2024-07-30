-- Create the tensorzero database if it doesn't exist
CREATE DATABASE IF NOT EXISTS tensorzero;

-- Switch to the tensorzero database
USE tensorzero;

-- BooleanMetricFeedback table
CREATE TABLE IF NOT EXISTS BooleanMetricFeedback
(
    id UUID,
    target_id UUID,
    metric_name LowCardinality(String),
    value Bool
) ENGINE = MergeTree()
ORDER BY (metric_name, target_id);

-- FloatMetricFeedback table
CREATE TABLE IF NOT EXISTS FloatMetricFeedback
(
    id UUID,
    target_id UUID,
    metric_name LowCardinality(String),
    value Float32
) ENGINE = MergeTree()
ORDER BY (metric_name, target_id);

-- DemonstrationFeedback table
CREATE TABLE IF NOT EXISTS DemonstrationFeedback
(
    id UUID,
    inference_id UUID,
    value String
) ENGINE = MergeTree()
ORDER BY inference_id;

-- CommentFeedback table
CREATE TABLE IF NOT EXISTS CommentFeedback
(
    id UUID,
    target_id UUID,
    target_type Enum('inference' = 1, 'episode' = 2),
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
    output Optional(String),
    -- This is whatever string we got from the Inference, without output sanitization
    raw_output String,
) ENGINE = MergeTree()
ORDER BY (function_name, variant_name, episode_id);

-- ModelInference table
CREATE TABLE IF NOT EXISTS ModelInference
(
    id UUID,
    inference_id UUID,
    input String,
    output String,
    raw_response String,
    input_tokens UInt32,
    output_tokens UInt32
) ENGINE = MergeTree()
ORDER BY inference_id;
