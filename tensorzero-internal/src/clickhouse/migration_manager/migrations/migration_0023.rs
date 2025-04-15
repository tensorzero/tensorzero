use crate::clickhouse::migration_manager::migration_trait::Migration;
use crate::clickhouse::ClickHouseConnectionInfo;
use crate::error::Error;
use async_trait::async_trait;

use super::check_table_exists;

/// This migration adds a table StaticEvaluationHumanFeedback that stores human feedback in an easy-to-reference format.
/// This is technically an auxiliary table as the primary store is still the various feedback tables.
/// The gateway should write to this table when a feedback is tagged with key "tensorzero::human_feedback"
pub struct Migration0023<'a> {
    pub clickhouse: &'a ClickHouseConnectionInfo,
}

#[async_trait]
impl Migration for Migration0023<'_> {
    async fn can_apply(&self) -> Result<(), Error> {
        Ok(())
    }

    async fn should_apply(&self) -> Result<bool, Error> {
        let human_feedback_table_exists =
            check_table_exists(self.clickhouse, "StaticEvaluationHumanFeedback", "0023").await?;

        Ok(!human_feedback_table_exists)
    }

    async fn apply(&self) -> Result<(), Error> {
        self.clickhouse
            .run_query_synchronous(
                r#"CREATE TABLE IF NOT EXISTS StaticEvaluationHumanFeedback (
                    metric_name LowCardinality(String),
                    datapoint_id UUID,
                    output String,
                    value String,  -- JSON encoded value of the feedback
                    feedback_id UUID,
                    timestamp DateTime MATERIALIZED UUIDv7ToDateTime(feedback_id)
                ) ENGINE = MergeTree()
                ORDER BY (metric_name, datapoint_id, output)
                SETTINGS index_granularity = 256 -- We use a small index granularity to improve lookup performance
            "#.to_string(),
                None,
            )
            .await?;

        Ok(())
    }

    fn rollback_instructions(&self) -> String {
        "DROP TABLE IF EXISTS StaticEvaluationHumanFeedback".to_string()
    }

    async fn has_succeeded(&self) -> Result<bool, Error> {
        let should_apply = self.should_apply().await?;
        Ok(!should_apply)
    }
}
