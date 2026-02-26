use async_trait::async_trait;
use sqlx::Row;

use super::PostgresConnectionInfo;
use crate::db::{HowdyFeedbackCounts, HowdyInferenceCounts, HowdyQueries, HowdyTokenUsage};
use crate::error::Error;

#[async_trait]
impl HowdyQueries for PostgresConnectionInfo {
    async fn count_inferences_for_howdy(&self) -> Result<HowdyInferenceCounts, Error> {
        let pool = self.get_pool_result()?;
        // Use pg_class.reltuples for fast approximate counts (sufficient for telemetry).
        // For partitioned tables, sum across child partitions.
        let row = sqlx::query(
            r"
            SELECT
                COALESCE((
                    SELECT SUM(GREATEST(c.reltuples, 0))::BIGINT
                    FROM pg_class c
                    JOIN pg_namespace n ON n.oid = c.relnamespace
                    JOIN pg_inherits i ON i.inhrelid = c.oid
                    JOIN pg_class parent ON parent.oid = i.inhparent
                    JOIN pg_namespace pn ON pn.oid = parent.relnamespace
                    WHERE pn.nspname = 'tensorzero' AND parent.relname = 'chat_inferences'
                ), 0) AS chat_inference_count,
                COALESCE((
                    SELECT SUM(GREATEST(c.reltuples, 0))::BIGINT
                    FROM pg_class c
                    JOIN pg_namespace n ON n.oid = c.relnamespace
                    JOIN pg_inherits i ON i.inhrelid = c.oid
                    JOIN pg_class parent ON parent.oid = i.inhparent
                    JOIN pg_namespace pn ON pn.oid = parent.relnamespace
                    WHERE pn.nspname = 'tensorzero' AND parent.relname = 'json_inferences'
                ), 0) AS json_inference_count
            ",
        )
        .fetch_one(pool)
        .await?;
        Ok(HowdyInferenceCounts {
            chat_inference_count: row.try_get::<i64, _>("chat_inference_count")? as u64,
            json_inference_count: row.try_get::<i64, _>("json_inference_count")? as u64,
        })
    }

    async fn count_feedbacks_for_howdy(&self) -> Result<HowdyFeedbackCounts, Error> {
        let pool = self.get_pool_result()?;
        // Use pg_class.reltuples for fast approximate counts (sufficient for telemetry).
        // Feedback tables are non-partitioned, so read reltuples directly.
        let row = sqlx::query(
            r"
            SELECT
                COALESCE((
                    SELECT GREATEST(c.reltuples, 0)::BIGINT
                    FROM pg_class c
                    JOIN pg_namespace n ON n.oid = c.relnamespace
                    WHERE n.nspname = 'tensorzero' AND c.relname = 'boolean_metric_feedback'
                ), 0) AS boolean_metric_feedback_count,
                COALESCE((
                    SELECT GREATEST(c.reltuples, 0)::BIGINT
                    FROM pg_class c
                    JOIN pg_namespace n ON n.oid = c.relnamespace
                    WHERE n.nspname = 'tensorzero' AND c.relname = 'float_metric_feedback'
                ), 0) AS float_metric_feedback_count,
                COALESCE((
                    SELECT GREATEST(c.reltuples, 0)::BIGINT
                    FROM pg_class c
                    JOIN pg_namespace n ON n.oid = c.relnamespace
                    WHERE n.nspname = 'tensorzero' AND c.relname = 'comment_feedback'
                ), 0) AS comment_feedback_count,
                COALESCE((
                    SELECT GREATEST(c.reltuples, 0)::BIGINT
                    FROM pg_class c
                    JOIN pg_namespace n ON n.oid = c.relnamespace
                    WHERE n.nspname = 'tensorzero' AND c.relname = 'demonstration_feedback'
                ), 0) AS demonstration_feedback_count
            ",
        )
        .fetch_one(pool)
        .await?;
        Ok(HowdyFeedbackCounts {
            boolean_metric_feedback_count: row.try_get::<i64, _>("boolean_metric_feedback_count")?
                as u64,
            float_metric_feedback_count: row.try_get::<i64, _>("float_metric_feedback_count")?
                as u64,
            comment_feedback_count: row.try_get::<i64, _>("comment_feedback_count")? as u64,
            demonstration_feedback_count: row.try_get::<i64, _>("demonstration_feedback_count")?
                as u64,
        })
    }

    async fn get_token_totals_for_howdy(&self) -> Result<HowdyTokenUsage, Error> {
        let pool = self.get_pool_result()?;
        let row = sqlx::query(
            r"
            SELECT
                SUM(total_input_tokens)::BIGINT AS input_tokens,
                SUM(total_output_tokens)::BIGINT AS output_tokens
            FROM tensorzero.model_provider_statistics
            ",
        )
        .fetch_one(pool)
        .await?;
        Ok(HowdyTokenUsage {
            input_tokens: row
                .try_get::<Option<i64>, _>("input_tokens")?
                .map(|v| v as u64),
            output_tokens: row
                .try_get::<Option<i64>, _>("output_tokens")?
                .map(|v| v as u64),
        })
    }
}
