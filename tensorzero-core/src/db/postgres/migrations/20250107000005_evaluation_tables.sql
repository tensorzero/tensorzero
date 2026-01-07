-- =============================================================================
-- Migration: Evaluation Tables
-- Description: Creates workflow_evaluation_run, workflow_evaluation_run_episode,
--              and inference_evaluation_human_feedback tables.
-- =============================================================================

-- =============================================================================
-- WORKFLOW EVALUATION RUN (formerly DynamicEvaluationRun)
-- =============================================================================

CREATE TABLE workflow_evaluation_run (
    run_id UUID PRIMARY KEY,
    variant_pins JSONB NOT NULL DEFAULT '{}',
    tags JSONB NOT NULL DEFAULT '{}',
    project_name TEXT,
    run_display_name TEXT,
    is_deleted BOOLEAN NOT NULL DEFAULT FALSE,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    snapshot_hash BYTEA
);

-- Partial index for active runs by project
CREATE INDEX idx_workflow_run_project ON workflow_evaluation_run(project_name)
    WHERE project_name IS NOT NULL AND NOT is_deleted;
CREATE INDEX idx_workflow_run_updated ON workflow_evaluation_run(updated_at DESC);
CREATE INDEX idx_workflow_run_tags ON workflow_evaluation_run USING GIN(tags);

-- Trigger for auto-updating updated_at
CREATE TRIGGER update_workflow_run_updated_at
    BEFORE UPDATE ON workflow_evaluation_run
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- =============================================================================
-- WORKFLOW EVALUATION RUN EPISODE
-- =============================================================================

CREATE TABLE workflow_evaluation_run_episode (
    run_id UUID NOT NULL,
    episode_id UUID NOT NULL,
    variant_pins JSONB NOT NULL DEFAULT '{}',
    datapoint_name TEXT,
    tags JSONB NOT NULL DEFAULT '{}',
    is_deleted BOOLEAN NOT NULL DEFAULT FALSE,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    snapshot_hash BYTEA,
    PRIMARY KEY (episode_id)
);

-- Index for looking up episodes by run
CREATE INDEX idx_workflow_episode_run ON workflow_evaluation_run_episode(run_id)
    WHERE NOT is_deleted;
CREATE INDEX idx_workflow_episode_updated ON workflow_evaluation_run_episode(updated_at DESC);
CREATE INDEX idx_workflow_episode_tags ON workflow_evaluation_run_episode USING GIN(tags);

-- Trigger for auto-updating updated_at
CREATE TRIGGER update_workflow_episode_updated_at
    BEFORE UPDATE ON workflow_evaluation_run_episode
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- =============================================================================
-- INFERENCE EVALUATION HUMAN FEEDBACK (formerly StaticEvaluationHumanFeedback)
-- =============================================================================

CREATE TABLE inference_evaluation_human_feedback (
    feedback_id UUID PRIMARY KEY,
    metric_name TEXT NOT NULL,
    datapoint_id UUID NOT NULL,
    output TEXT NOT NULL,
    value JSONB NOT NULL,
    evaluator_inference_id UUID,
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_eval_feedback_metric_datapoint ON inference_evaluation_human_feedback(metric_name, datapoint_id);
CREATE INDEX idx_eval_feedback_datapoint ON inference_evaluation_human_feedback(datapoint_id);
CREATE INDEX idx_eval_feedback_timestamp ON inference_evaluation_human_feedback(timestamp DESC);
