-- Feedback tables for the ClickHouse-to-Postgres migration (Step 2)
-- These tables store user feedback linked to inferences and episodes.

-- boolean_metric_feedback: Boolean feedback metrics linked to inferences/episodes
CREATE TABLE tensorzero.boolean_metric_feedback (
    id UUID PRIMARY KEY,
    target_id UUID NOT NULL,
    metric_name TEXT NOT NULL,
    value BOOLEAN NOT NULL,
    tags JSONB NOT NULL DEFAULT '{}',
    snapshot_hash BYTEA,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_boolean_feedback_metric_target ON tensorzero.boolean_metric_feedback(metric_name, target_id);
CREATE INDEX idx_boolean_feedback_target ON tensorzero.boolean_metric_feedback(target_id);
CREATE INDEX idx_boolean_feedback_created_at ON tensorzero.boolean_metric_feedback(created_at);
CREATE INDEX idx_boolean_feedback_tags ON tensorzero.boolean_metric_feedback USING GIN (tags);

-- float_metric_feedback: Float-valued metrics linked to inferences/episodes
CREATE TABLE tensorzero.float_metric_feedback (
    id UUID PRIMARY KEY,
    target_id UUID NOT NULL,
    metric_name TEXT NOT NULL,
    value DOUBLE PRECISION NOT NULL,
    tags JSONB NOT NULL DEFAULT '{}',
    snapshot_hash BYTEA,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_float_feedback_metric_target ON tensorzero.float_metric_feedback(metric_name, target_id);
CREATE INDEX idx_float_feedback_target ON tensorzero.float_metric_feedback(target_id);
CREATE INDEX idx_float_feedback_created_at ON tensorzero.float_metric_feedback(created_at);
CREATE INDEX idx_float_feedback_tags ON tensorzero.float_metric_feedback USING GIN (tags);

-- comment_feedback: Textual feedback on tensorzero.inferences or episodes
CREATE TABLE tensorzero.comment_feedback (
    id UUID PRIMARY KEY,
    target_id UUID NOT NULL,
    target_type TEXT NOT NULL CHECK (target_type IN ('inference', 'episode')),
    value TEXT NOT NULL,
    tags JSONB NOT NULL DEFAULT '{}',
    snapshot_hash BYTEA,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_comment_feedback_target ON tensorzero.comment_feedback(target_id);
CREATE INDEX idx_comment_feedback_created_at ON tensorzero.comment_feedback(created_at);
CREATE INDEX idx_comment_feedback_tags ON tensorzero.comment_feedback USING GIN (tags);

-- demonstration_feedback: Demonstrations linked to inferences
-- value is JSONB since it stores either JsonInferenceOutput or Vec<ContentBlockChatOutput>
CREATE TABLE tensorzero.demonstration_feedback (
    id UUID PRIMARY KEY,
    inference_id UUID NOT NULL,
    value JSONB NOT NULL,
    tags JSONB NOT NULL DEFAULT '{}',
    snapshot_hash BYTEA,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_demonstration_feedback_inference ON tensorzero.demonstration_feedback(inference_id);
CREATE INDEX idx_demonstration_feedback_created_at ON tensorzero.demonstration_feedback(created_at);
CREATE INDEX idx_demonstration_feedback_tags ON tensorzero.demonstration_feedback USING GIN (tags);

-- static_evaluation_human_feedback: Human feedback for inference evaluations
-- Note: "StaticEvaluation" is the legacy name, now called "Inference Evaluations"
CREATE TABLE tensorzero.inference_evaluation_human_feedback (
    feedback_id UUID PRIMARY KEY,
    metric_name TEXT NOT NULL,
    datapoint_id UUID NOT NULL,
    output TEXT NOT NULL,
    value TEXT NOT NULL,
    evaluator_inference_id UUID,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_static_eval_feedback_lookup
    ON tensorzero.inference_evaluation_human_feedback(metric_name, datapoint_id, output);
CREATE INDEX idx_static_eval_feedback_created_at ON tensorzero.inference_evaluation_human_feedback(created_at);

-- Custom aggregate functions for MIN/MAX on UUIDs
-- PostgreSQL before version 18 doesn't have built-in MIN/MAX for UUIDs, so we create our own.

-- Helper function for min_uuid aggregate
CREATE OR REPLACE FUNCTION tensorzero.uuid_smaller(uuid, uuid)
RETURNS uuid AS $$
    SELECT CASE
        WHEN $1 IS NULL THEN $2
        WHEN $2 IS NULL THEN $1
        WHEN $1 < $2 THEN $1
        ELSE $2
    END;
$$ LANGUAGE SQL IMMUTABLE;

-- Helper function for max_uuid aggregate
CREATE OR REPLACE FUNCTION tensorzero.uuid_larger(uuid, uuid)
RETURNS uuid AS $$
    SELECT CASE
        WHEN $1 IS NULL THEN $2
        WHEN $2 IS NULL THEN $1
        WHEN $1 > $2 THEN $1
        ELSE $2
    END;
$$ LANGUAGE SQL IMMUTABLE;

-- min_uuid aggregate
CREATE AGGREGATE tensorzero.min_uuid(uuid) (
    SFUNC = tensorzero.uuid_smaller,
    STYPE = uuid,
    PARALLEL = SAFE
);

-- max_uuid aggregate
CREATE AGGREGATE tensorzero.max_uuid(uuid) (
    SFUNC = tensorzero.uuid_larger,
    STYPE = uuid,
    PARALLEL = SAFE
);
