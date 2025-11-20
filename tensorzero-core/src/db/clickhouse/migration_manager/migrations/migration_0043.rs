use crate::db::clickhouse::migration_manager::migration_trait::Migration;
use crate::db::clickhouse::migration_manager::migrations::{
    check_column_exists, check_table_exists,
};
use crate::db::clickhouse::ClickHouseConnectionInfo;
use crate::error::Error;
use async_trait::async_trait;

const MIGRATION_ID: &str = "0043";

/// TODO
pub struct Migration0043<'a> {
    pub clickhouse: &'a ClickHouseConnectionInfo,
}

#[async_trait]
impl<'a> Migration for Migration0043<'a> {
    async fn can_apply(&self) -> Result<(), Error> {
        Ok(())
    }

    async fn should_apply(&self) -> Result<bool, Error> {
        // Check if ConfigSnapshot table doesn't exist
        if !check_table_exists(self.clickhouse, "ConfigSnapshot", MIGRATION_ID).await? {
            return Ok(true);
        }

        // Check if any of the tables is missing the snapshot_hash column
        let tables = [
            "BatchModelInference",
            "BatchRequest",
            "BooleanMetricFeedback",
            "BooleanMetricFeedbackByTargetId",
            "BooleanMetricFeedbackByVariant",
            "ChatInference",
            "ChatInferenceDatapoint",
            "CommentFeedback",
            "CommentFeedbackByTargetId",
            "DemonstrationFeedback",
            "DemonstrationFeedbackByInferenceId",
            "DynamicEvaluationRun",
            "DynamicEvaluationRunByProjectName",
            "DynamicEvaluationRunEpisode",
            "FeedbackTag",
            "FloatMetricFeedback",
            "FloatMetricFeedbackByTargetId",
            "InferenceByEpisodeId",
            "InferenceById",
            "InferenceTag",
            "JsonInference",
            "JsonInferenceDatapoint",
            "ModelInference",
            "TagInference",
        ];

        for table in tables {
            if !check_column_exists(self.clickhouse, table, "snapshot_hash", MIGRATION_ID).await? {
                return Ok(true);
            }
        }
        Ok(false)
    }

    async fn apply(&self, _clean_start: bool) -> Result<(), Error> {
        // Create ConfigSnapshot table
        let create_table_query = "
            CREATE TABLE IF NOT EXISTS ConfigSnapshot (
                config String,
                extra_templates Map(String, String),
                version_hash UInt256,
                tensorzero_version String,
                created_at DateTime64(6) DEFAULT now(),
                last_used DateTime64(6) DEFAULT now()
            ) ENGINE = ReplacingMergeTree(last_used)
            ORDER BY version_hash
            SETTINGS index_granularity = 256
        ";
        self.clickhouse
            .run_query_synchronous_no_params(create_table_query.to_string())
            .await?;

        // Add snapshot_hash column to existing tables
        let tables = [
            "BatchModelInference",
            "BatchRequest",
            "BooleanMetricFeedback",
            "BooleanMetricFeedbackByTargetId",
            "BooleanMetricFeedbackByVariant",
            "ChatInference",
            "ChatInferenceDatapoint",
            "CommentFeedback",
            "CommentFeedbackByTargetId",
            "DemonstrationFeedback",
            "DemonstrationFeedbackByInferenceId",
            "DynamicEvaluationRun",
            "DynamicEvaluationRunByProjectName",
            "DynamicEvaluationRunEpisode",
            "FeedbackTag",
            "FloatMetricFeedback",
            "FloatMetricFeedbackByTargetId",
            "InferenceByEpisodeId",
            "InferenceById",
            "InferenceTag",
            "JsonInference",
            "JsonInferenceDatapoint",
            "ModelInference",
            "TagInference",
        ];

        for table in tables {
            let query = format!(
                "ALTER TABLE {table} ADD COLUMN IF NOT EXISTS snapshot_hash Nullable(UInt256)"
            );
            self.clickhouse
                .run_query_synchronous_no_params(query)
                .await?;
        }
        Ok(())
    }

    fn rollback_instructions(&self) -> String {
        let mut instructions = String::from("DROP TABLE ConfigSnapshot;\n");

        let tables = [
            "BatchModelInference",
            "BatchRequest",
            "BooleanMetricFeedback",
            "BooleanMetricFeedbackByTargetId",
            "BooleanMetricFeedbackByVariant",
            "ChatInference",
            "ChatInferenceDatapoint",
            "CommentFeedback",
            "CommentFeedbackByTargetId",
            "DemonstrationFeedback",
            "DemonstrationFeedbackByInferenceId",
            "DynamicEvaluationRun",
            "DynamicEvaluationRunByProjectName",
            "DynamicEvaluationRunEpisode",
            "FeedbackTag",
            "FloatMetricFeedback",
            "FloatMetricFeedbackByTargetId",
            "InferenceByEpisodeId",
            "InferenceById",
            "InferenceTag",
            "JsonInference",
            "JsonInferenceDatapoint",
            "ModelInference",
            "TagInference",
        ];

        for table in tables {
            instructions.push_str(&format!("ALTER TABLE {table} DROP COLUMN snapshot_hash;\n"));
        }
        instructions
    }

    async fn has_succeeded(&self) -> Result<bool, Error> {
        Ok(!self.should_apply().await?)
    }
}
