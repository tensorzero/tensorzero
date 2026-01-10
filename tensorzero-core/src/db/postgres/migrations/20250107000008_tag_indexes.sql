-- =============================================================================
-- Migration: Additional Tag Indexes
-- Description: Creates optimized GIN indexes for tag queries and the episode_summary view.
-- =============================================================================

-- =============================================================================
-- OPTIMIZED TAG INDEXES
-- Using jsonb_path_ops for more efficient containment queries (@>)
-- =============================================================================

-- Inference tables
CREATE INDEX idx_chat_inference_tags_path ON chat_inference USING GIN(tags jsonb_path_ops);
CREATE INDEX idx_json_inference_tags_path ON json_inference USING GIN(tags jsonb_path_ops);

-- Feedback tables
CREATE INDEX idx_boolean_feedback_tags_path ON boolean_metric_feedback USING GIN(tags jsonb_path_ops);
CREATE INDEX idx_float_feedback_tags_path ON float_metric_feedback USING GIN(tags jsonb_path_ops);
CREATE INDEX idx_comment_feedback_tags_path ON comment_feedback USING GIN(tags jsonb_path_ops);
CREATE INDEX idx_demonstration_feedback_tags_path ON demonstration_feedback USING GIN(tags jsonb_path_ops);

-- =============================================================================
-- EPISODE SUMMARY VIEW
-- Provides aggregated episode statistics.
-- Replaces ClickHouse EpisodeById AggregatingMergeTree table.
-- =============================================================================

CREATE VIEW episode_summary AS
SELECT
    episode_id,
    array_agg(id ORDER BY timestamp) AS inference_ids,
    COUNT(*) AS inference_count,
    tensorzero_min(id) AS first_inference_id,
    tensorzero_max(id) AS last_inference_id,
    MIN(timestamp) AS first_timestamp,
    MAX(timestamp) AS last_timestamp
FROM (
    SELECT id, episode_id, timestamp FROM chat_inference
    UNION ALL
    SELECT id, episode_id, timestamp FROM json_inference
) all_inferences
GROUP BY episode_id;

-- =============================================================================
-- FEEDBACK BY VARIANT INDEX
-- For analytics queries that filter by function/variant/metric
-- =============================================================================

CREATE INDEX idx_boolean_feedback_variant ON boolean_metric_feedback(metric_name, target_id);
CREATE INDEX idx_float_feedback_variant ON float_metric_feedback(metric_name, target_id);
