use crate::clickhouse::migration_manager::migration_trait::Migration;
use crate::clickhouse::ClickHouseConnectionInfo;
use crate::error::Error;
use async_trait::async_trait;

use super::check_table_exists;

/// This migration is used to set up the ClickHouse database for batch inference
/// We will add two main tables: `BatchModelInference` and `BatchRequest` as well as a
/// materialized view `BatchIdByInferenceId` that maps inference ids to batch ids.
///
/// `BatchModelInference` contains each actual inference being made in a batch request.
/// It should contain enough information to create the eventual insertions into
/// JsonInference, ChatInference, and ModelInference once the batch has been completed.
///
/// `BatchRequest` contains metadata about a batch request.
/// Each time the batch is polled by either inference_id or batch_id, a row will be written to this table.
/// This allows us to know and also to know the history of actions which have been taken here.
pub struct Migration0006<'a> {
    pub clickhouse: &'a ClickHouseConnectionInfo,
}

#[async_trait]
impl Migration for Migration0006<'_> {
    async fn can_apply(&self) -> Result<(), Error> {
        Ok(())
    }

    /// Check if the migration has already been applied by checking if the new tables exist or the new view exists
    async fn should_apply(&self) -> Result<bool, Error> {
        let tables = vec![
            "BatchModelInference",
            "BatchRequest",
            "BatchIdByInferenceId",
            "BatchIdByInferenceIdView",
        ];
        for table in tables {
            if !check_table_exists(self.clickhouse, table, "0006").await? {
                return Ok(true);
            }
        }

        // Everything is in place, so we should not apply the migration
        Ok(false)
    }

    async fn apply(&self) -> Result<(), Error> {
        // Create the `BatchModelInference` table
        let query = r#"
            CREATE TABLE IF NOT EXISTS BatchModelInference
            (
                inference_id UUID,
                batch_id UUID,
                function_name LowCardinality(String),
                variant_name LowCardinality(String),
                episode_id UUID,
                input String,
                input_messages String,
                system Nullable(String),
                tool_params Nullable(String),
                inference_params String,
                raw_request String,
                model_name LowCardinality(String),
                model_provider_name LowCardinality(String),
                output_schema Nullable(String),
                tags Map(String, String) DEFAULT map(),
                timestamp DateTime MATERIALIZED UUIDv7ToDateTime(inference_id),
            ) ENGINE = MergeTree()
            ORDER BY (batch_id, inference_id)
        "#;
        let _ = self.clickhouse.run_query(query.to_string(), None).await?;

        // Create the `BatchRequest` table
        let query = r#"
            CREATE TABLE IF NOT EXISTS BatchRequest
            (
                batch_id UUID,
                id UUID,
                batch_params String,
                model_name LowCardinality(String),
                model_provider_name LowCardinality(String),
                status Enum('pending' = 1, 'completed' = 2, 'failed' = 3),
                errors Map(UUID, String),
                timestamp DateTime MATERIALIZED UUIDv7ToDateTime(id),
            ) ENGINE = MergeTree()
            ORDER BY (batch_id, id)
        "#;
        let _ = self.clickhouse.run_query(query.to_string(), None).await?;

        // Create the BatchIdByInferenceId table
        let query = r#"
            CREATE TABLE IF NOT EXISTS BatchIdByInferenceId
            (
                inference_id UUID,
                batch_id UUID,
            ) ENGINE = MergeTree()
            ORDER BY (inference_id)
        "#;
        let _ = self.clickhouse.run_query(query.to_string(), None).await?;

        // Create the materialized view for the BatchIdByInferenceId table
        let query = r#"
            CREATE MATERIALIZED VIEW IF NOT EXISTS BatchIdByInferenceIdView
            TO BatchIdByInferenceId
            AS
                SELECT
                    inference_id,
                    batch_id
                FROM BatchModelInference
            "#;
        let _ = self.clickhouse.run_query(query.to_string(), None).await?;
        Ok(())
    }

    /// Check if the migration has succeeded (i.e. it should not be applied again)
    async fn has_succeeded(&self) -> Result<bool, Error> {
        let should_apply = self.should_apply().await?;
        Ok(!should_apply)
    }

    fn rollback_instructions(&self) -> String {
        "\
            -- Drop the materialized views\n\
            DROP MATERIALIZED VIEW IF EXISTS BatchIdByInferenceIdView;\n\
            \n\
            -- Drop the tables\n\
            DROP TABLE IF EXISTS BatchIdByInferenceId;\n\
            DROP TABLE IF EXISTS BatchRequest;\n\
            DROP TABLE IF EXISTS BatchModelInference;\n\
        "
        .to_string()
    }
}
