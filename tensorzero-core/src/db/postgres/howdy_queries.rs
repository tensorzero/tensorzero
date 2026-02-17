use async_trait::async_trait;
use sqlx::Row;

use super::PostgresConnectionInfo;
use crate::db::{HowdyFeedbackCounts, HowdyInferenceCounts, HowdyQueries, HowdyTokenUsage};
use crate::error::Error;

#[async_trait]
impl HowdyQueries for PostgresConnectionInfo {
    async fn count_inferences_for_howdy(&self) -> Result<HowdyInferenceCounts, Error> {
        let pool = self.get_pool_result()?;
        let row = sqlx::query(
            r"
            SELECT
                (SELECT COUNT(*)::BIGINT FROM tensorzero.chat_inferences) AS chat_inference_count,
                (SELECT COUNT(*)::BIGINT FROM tensorzero.json_inferences) AS json_inference_count
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
        let row = sqlx::query(
            r"
            SELECT
                (SELECT COUNT(*)::BIGINT FROM tensorzero.boolean_metric_feedback) AS boolean_metric_feedback_count,
                (SELECT COUNT(*)::BIGINT FROM tensorzero.float_metric_feedback) AS float_metric_feedback_count,
                (SELECT COUNT(*)::BIGINT FROM tensorzero.comment_feedback) AS comment_feedback_count,
                (SELECT COUNT(*)::BIGINT FROM tensorzero.demonstration_feedback) AS demonstration_feedback_count
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
                SUM(input_tokens)::BIGINT AS input_tokens,
                SUM(output_tokens)::BIGINT AS output_tokens
            FROM tensorzero.model_inferences
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
