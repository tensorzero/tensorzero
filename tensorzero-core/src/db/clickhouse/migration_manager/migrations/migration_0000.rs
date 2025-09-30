use crate::db::clickhouse::migration_manager::migration_trait::Migration;
use crate::db::clickhouse::{ClickHouseConnectionInfo, GetMaybeReplicatedTableEngineNameArgs};
use crate::error::Error;
use async_trait::async_trait;

use super::check_table_exists;

/// This migration is used to create the initial tables in the ClickHouse database.
///
/// It is used to create the following tables:
/// - BooleanMetricFeedback
/// - CommentFeedback
/// - DemonstrationFeedback
/// - FloatMetricFeedback
/// - ChatInference
/// - JsonInference
/// - ModelInference
pub struct Migration0000<'a> {
    pub clickhouse: &'a ClickHouseConnectionInfo,
}

#[async_trait]
impl Migration for Migration0000<'_> {
    async fn can_apply(&self) -> Result<(), Error> {
        Ok(())
    }

    /// Check if the tables exist
    async fn should_apply(&self) -> Result<bool, Error> {
        let tables = vec![
            "BooleanMetricFeedback",
            "CommentFeedback",
            "DemonstrationFeedback",
            "FloatMetricFeedback",
            "ChatInference",
            "JsonInference",
            "ModelInference",
        ];
        for table in tables {
            match check_table_exists(self.clickhouse, table, "0000").await {
                Ok(exists) => {
                    if !exists {
                        return Ok(true);
                    }
                }
                // If `can_apply` succeeds but this fails, it likely means the database does not exist
                Err(_) => return Ok(true),
            }
        }

        Ok(false)
    }

    async fn apply(&self, _clean_start: bool) -> Result<(), Error> {
        // Create the `BooleanMetricFeedback` table
        let boolean_feedback_schema = r"
            (
                id UUID, -- must be a UUIDv7
                target_id UUID, -- must be a UUIDv7
                metric_name LowCardinality(String),
                value Bool,
                timestamp DateTime MATERIALIZED UUIDv7ToDateTime(id)
            )";

        self.clickhouse.get_create_table_statements(
            "BooleanMetricFeedback",
            boolean_feedback_schema,
            &GetMaybeReplicatedTableEngineNameArgs {
                table_engine_name: "MergeTree",
                table_name: "BooleanMetricFeedback",
                engine_args: &[],
            },
            Some("ORDER BY (metric_name, target_id)"),
            None,
        ).await?;

        // Create the `CommentFeedback` table
        let comment_feedback_schema = r"
            (
                id UUID, -- must be a UUIDv7
                target_id UUID, -- must be a UUIDv7
                target_type Enum('inference' = 1, 'episode' = 2),
                value String,
                timestamp DateTime MATERIALIZED UUIDv7ToDateTime(id)
            )";

        self.clickhouse.get_create_table_statements(
            "CommentFeedback",
            comment_feedback_schema,
            &GetMaybeReplicatedTableEngineNameArgs {
                table_engine_name: "MergeTree",
                table_name: "CommentFeedback",
                engine_args: &[],
            },
            Some("ORDER BY target_id"),
            None,
        ).await?;

        // Create the `DemonstrationFeedback` table
        let demo_feedback_schema = r"
            (
                id UUID, -- must be a UUIDv7
                inference_id UUID, -- must be a UUIDv7
                value String,
                timestamp DateTime MATERIALIZED UUIDv7ToDateTime(id)
            )";

        self.clickhouse.get_create_table_statements(
            "DemonstrationFeedback",
            demo_feedback_schema,
            &GetMaybeReplicatedTableEngineNameArgs {
                table_engine_name: "MergeTree",
                table_name: "DemonstrationFeedback",
                engine_args: &[],
            },
            Some("ORDER BY inference_id"),
            None,
        ).await?;

        // Create the `FloatMetricFeedback` table
        let float_feedback_schema = r"
            (
                id UUID, -- must be a UUIDv7
                target_id UUID, -- must be a UUIDv7
                metric_name LowCardinality(String),
                value Float32,
                timestamp DateTime MATERIALIZED UUIDv7ToDateTime(id)
            )";

        self.clickhouse.get_create_table_statements(
            "FloatMetricFeedback",
            float_feedback_schema,
            &GetMaybeReplicatedTableEngineNameArgs {
                table_engine_name: "MergeTree",
                table_name: "FloatMetricFeedback",
                engine_args: &[],
            },
            Some("ORDER BY (metric_name, target_id)"),
            None,
        ).await?;

        // Create the `ChatInference` table
        let chat_inference_schema = r"
            (
                id UUID, -- must be a UUIDv7
                function_name LowCardinality(String),
                variant_name LowCardinality(String),
                episode_id UUID, -- must be a UUIDv7
                input String,
                output String,
                tool_params String,
                inference_params String,
                processing_time_ms UInt32,
                timestamp DateTime MATERIALIZED UUIDv7ToDateTime(id)
            )";

        self.clickhouse.get_create_table_statements(
            "ChatInference",
            chat_inference_schema,
            &GetMaybeReplicatedTableEngineNameArgs {
                table_engine_name: "MergeTree",
                table_name: "ChatInference",
                engine_args: &[],
            },
            Some("ORDER BY (function_name, variant_name, episode_id)"),
            Some("cityHash64(episode_id)"),
        ).await?;

        // Create the `JsonInference` table
        let json_inference_schema = r"
            (
                id UUID, -- must be a UUIDv7
                function_name LowCardinality(String),
                variant_name LowCardinality(String),
                episode_id UUID, -- must be a UUIDv7
                input String,
                output String,
                output_schema String,
                inference_params String,
                processing_time_ms UInt32,
                timestamp DateTime MATERIALIZED UUIDv7ToDateTime(id)
            )";

        self.clickhouse.get_create_table_statements(
            "JsonInference",
            json_inference_schema,
            &GetMaybeReplicatedTableEngineNameArgs {
                table_engine_name: "MergeTree",
                table_name: "JsonInference",
                engine_args: &[],
            },
            Some("ORDER BY (function_name, variant_name, episode_id)"),
            Some("cityHash64(episode_id)"),
        ).await?;

        // Create the `ModelInference` table
        let model_inference_schema = r"
            (
                id UUID, -- must be a UUIDv7
                inference_id UUID, -- must be a UUIDv7
                raw_request String,
                raw_response String,
                model_name LowCardinality(String),
                model_provider_name LowCardinality(String),
                input_tokens UInt32,
                output_tokens UInt32,
                response_time_ms UInt32,
                ttft_ms Nullable(UInt32),
                timestamp DateTime MATERIALIZED UUIDv7ToDateTime(id)
            )";

        self.clickhouse.get_create_table_statements(
            "ModelInference",
            model_inference_schema,
            &GetMaybeReplicatedTableEngineNameArgs {
                table_engine_name: "MergeTree",
                table_name: "ModelInference",
                engine_args: &[],
            },
            Some("ORDER BY inference_id"),
            None,
        ).await?;

        Ok(())
    }

    fn rollback_instructions(&self) -> String {
        let database = self.clickhouse.database();
        let on_cluster_name = self.clickhouse.get_on_cluster_name();

        format!(
            "/* **CAREFUL: THIS WILL DELETE ALL DATA** */\
            /* Drop each table */\
            {}\
            {}\
            {}\
            {}\
            {}\
            {}\
            {}\
            {}\
            /* Drop the database */\
            DROP DATABASE IF EXISTS {database}{on_cluster_name};\
            /* **CAREFUL: THIS WILL DELETE ALL DATA** */",
            self.clickhouse.get_drop_table_rollback_statements("BooleanMetricFeedback"),
            self.clickhouse.get_drop_table_rollback_statements("CommentFeedback"),
            self.clickhouse.get_drop_table_rollback_statements("DemonstrationFeedback"),
            self.clickhouse.get_drop_table_rollback_statements("FloatMetricFeedback"),
            self.clickhouse.get_drop_table_rollback_statements("ChatInference"),
            self.clickhouse.get_drop_table_rollback_statements("JsonInference"),
            self.clickhouse.get_drop_table_rollback_statements("ModelInference"),
            self.clickhouse.get_drop_table_rollback_statements("TensorZeroMigration")
        )
    }

    /// Check if the migration has succeeded (i.e. it should not be applied again)
    async fn has_succeeded(&self) -> Result<bool, Error> {
        let should_apply = self.should_apply().await?;
        Ok(!should_apply)
    }
}
