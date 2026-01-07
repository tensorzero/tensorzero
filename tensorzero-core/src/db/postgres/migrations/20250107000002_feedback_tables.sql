-- =============================================================================
-- Migration: Feedback Tables
-- Description: Creates boolean_metric_feedback, float_metric_feedback,
--              comment_feedback, and demonstration_feedback tables.
-- =============================================================================

-- =============================================================================
-- ENUM TYPES
-- =============================================================================

CREATE TYPE target_type_enum AS ENUM ('inference', 'episode');

-- =============================================================================
-- BOOLEAN METRIC FEEDBACK
-- =============================================================================

CREATE TABLE boolean_metric_feedback (
    id UUID PRIMARY KEY,
    target_id UUID NOT NULL,
    metric_name TEXT NOT NULL,
    value BOOLEAN NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    tags JSONB NOT NULL DEFAULT '{}',
    snapshot_hash BYTEA
);

CREATE INDEX idx_boolean_metric_feedback_target ON boolean_metric_feedback(target_id);
CREATE INDEX idx_boolean_metric_feedback_metric ON boolean_metric_feedback(metric_name);
CREATE INDEX idx_boolean_metric_feedback_metric_target ON boolean_metric_feedback(metric_name, target_id);
CREATE INDEX idx_boolean_metric_feedback_timestamp ON boolean_metric_feedback(timestamp DESC);
CREATE INDEX idx_boolean_metric_feedback_tags ON boolean_metric_feedback USING GIN(tags);

-- =============================================================================
-- FLOAT METRIC FEEDBACK
-- =============================================================================

CREATE TABLE float_metric_feedback (
    id UUID PRIMARY KEY,
    target_id UUID NOT NULL,
    metric_name TEXT NOT NULL,
    value REAL NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    tags JSONB NOT NULL DEFAULT '{}',
    snapshot_hash BYTEA
);

CREATE INDEX idx_float_metric_feedback_target ON float_metric_feedback(target_id);
CREATE INDEX idx_float_metric_feedback_metric ON float_metric_feedback(metric_name);
CREATE INDEX idx_float_metric_feedback_metric_target ON float_metric_feedback(metric_name, target_id);
CREATE INDEX idx_float_metric_feedback_timestamp ON float_metric_feedback(timestamp DESC);
CREATE INDEX idx_float_metric_feedback_tags ON float_metric_feedback USING GIN(tags);

-- =============================================================================
-- COMMENT FEEDBACK
-- =============================================================================

CREATE TABLE comment_feedback (
    id UUID PRIMARY KEY,
    target_id UUID NOT NULL,
    target_type target_type_enum NOT NULL,
    value TEXT NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    tags JSONB NOT NULL DEFAULT '{}',
    snapshot_hash BYTEA
);

CREATE INDEX idx_comment_feedback_target ON comment_feedback(target_id);
CREATE INDEX idx_comment_feedback_timestamp ON comment_feedback(timestamp DESC);
CREATE INDEX idx_comment_feedback_tags ON comment_feedback USING GIN(tags);

-- =============================================================================
-- DEMONSTRATION FEEDBACK
-- =============================================================================

CREATE TABLE demonstration_feedback (
    id UUID PRIMARY KEY,
    inference_id UUID NOT NULL,
    value TEXT NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    tags JSONB NOT NULL DEFAULT '{}',
    snapshot_hash BYTEA
);

CREATE INDEX idx_demonstration_feedback_inference ON demonstration_feedback(inference_id);
CREATE INDEX idx_demonstration_feedback_timestamp ON demonstration_feedback(timestamp DESC);
CREATE INDEX idx_demonstration_feedback_tags ON demonstration_feedback USING GIN(tags);
