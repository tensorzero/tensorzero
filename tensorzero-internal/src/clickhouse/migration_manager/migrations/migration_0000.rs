use crate::clickhouse::migration_manager::migration_trait::Migration;
use crate::clickhouse::ClickHouseConnectionInfo;
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

    async fn apply(&self) -> Result<(), Error> {
        // Create the `BooleanMetricFeedback` table
        let query = r#"
            CREATE TABLE IF NOT EXISTS BooleanMetricFeedback
            (
                id UUID, -- must be a UUIDv7
                target_id UUID, -- must be a UUIDv7
                metric_name LowCardinality(String),
                value Bool,
                timestamp DateTime MATERIALIZED UUIDv7ToDateTime(id)
            ) ENGINE = MergeTree()
            ORDER BY (metric_name, target_id);
        "#;
        let _ = self.clickhouse.run_query(query.to_string(), None).await?;

        // Create the `CommentFeedback` table
        let query = r#"
            CREATE TABLE IF NOT EXISTS CommentFeedback
            (
                id UUID, -- must be a UUIDv7
                target_id UUID, -- must be a UUIDv7
                target_type Enum('inference' = 1, 'episode' = 2),
                value String,
                timestamp DateTime MATERIALIZED UUIDv7ToDateTime(id)
            ) ENGINE = MergeTree()
            ORDER BY target_id;
        "#;
        let _ = self.clickhouse.run_query(query.to_string(), None).await?;

        // Create the `DemonstrationFeedback` table
        let query = r#"
           CREATE TABLE IF NOT EXISTS DemonstrationFeedback
            (
                id UUID, -- must be a UUIDv7
                inference_id UUID, -- must be a UUIDv7
                value String,
                timestamp DateTime MATERIALIZED UUIDv7ToDateTime(id)
            ) ENGINE = MergeTree()
            ORDER BY inference_id;
        "#;
        let _ = self.clickhouse.run_query(query.to_string(), None).await?;

        // Create the `FloatMetricFeedback` table
        let query = r#"
            CREATE TABLE IF NOT EXISTS FloatMetricFeedback
            (
                id UUID, -- must be a UUIDv7
                target_id UUID, -- must be a UUIDv7
                metric_name LowCardinality(String),
                value Float32,
                timestamp DateTime MATERIALIZED UUIDv7ToDateTime(id)
            ) ENGINE = MergeTree()
            ORDER BY (metric_name, target_id);
        "#;
        let _ = self.clickhouse.run_query(query.to_string(), None).await?;

        // Create the `ChatInference` table
        let query = r#"
            CREATE TABLE IF NOT EXISTS ChatInference
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
            ) ENGINE = MergeTree()
            ORDER BY (function_name, variant_name, episode_id);
        "#;
        let _ = self.clickhouse.run_query(query.to_string(), None).await?;

        // Create the `JsonInference` table
        let query = r#"
            CREATE TABLE IF NOT EXISTS JsonInference
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
            ) ENGINE = MergeTree()
            ORDER BY (function_name, variant_name, episode_id);
        "#;
        let _ = self.clickhouse.run_query(query.to_string(), None).await?;

        // Create the `ModelInference` table
        let query = r#"
            CREATE TABLE IF NOT EXISTS ModelInference
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
            ) ENGINE = MergeTree()
            ORDER BY inference_id;
        "#;
        let _ = self.clickhouse.run_query(query.to_string(), None).await?;

        Ok(())
    }

    fn rollback_instructions(&self) -> String {
        let database = self.clickhouse.database();

        format!(
            "\
            **CAREFUL: THIS WILL DELETE ALL DATA**\n\
            \n\
            -- Drop the database\n\
            DROP DATABASE IF EXISTS {database};\n\
            \n\
            **CAREFUL: THIS WILL DELETE ALL DATA**\n\
            "
        )
    }

    /// Check if the migration has succeeded (i.e. it should not be applied again)
    async fn has_succeeded(&self) -> Result<bool, Error> {
        let should_apply = self.should_apply().await?;
        Ok(!should_apply)
    }
}
