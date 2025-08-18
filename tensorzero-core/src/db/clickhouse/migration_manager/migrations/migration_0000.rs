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
        let table_engine_name = self.clickhouse.get_maybe_replicated_table_engine_name(
            GetMaybeReplicatedTableEngineNameArgs {
                table_engine_name: "MergeTree",
                table_name: "BooleanMetricFeedback",
                engine_args: &[],
            },
        );
        let on_cluster_name = self.clickhouse.get_on_cluster_name();
        let query = format!(
            r"
            CREATE TABLE IF NOT EXISTS BooleanMetricFeedback{on_cluster_name}
            (
                id UUID, -- must be a UUIDv7
                target_id UUID, -- must be a UUIDv7
                metric_name LowCardinality(String),
                value Bool,
                timestamp DateTime MATERIALIZED UUIDv7ToDateTime(id)
            ) ENGINE = {table_engine_name}
            ORDER BY (metric_name, target_id);
        ",
        );
        let _ = self
            .clickhouse
            .run_query_synchronous_no_params(query.to_string())
            .await?;

        // Create the `CommentFeedback` table
        let table_engine_name = self.clickhouse.get_maybe_replicated_table_engine_name(
            GetMaybeReplicatedTableEngineNameArgs {
                table_engine_name: "MergeTree",
                table_name: "CommentFeedback",
                engine_args: &[],
            },
        );
        let query = format!(
            r"
            CREATE TABLE IF NOT EXISTS CommentFeedback{on_cluster_name}
            (
                id UUID, -- must be a UUIDv7
                target_id UUID, -- must be a UUIDv7
                target_type Enum('inference' = 1, 'episode' = 2),
                value String,
                timestamp DateTime MATERIALIZED UUIDv7ToDateTime(id)
            ) ENGINE = {table_engine_name}
            ORDER BY target_id;
        ",
        );
        let _ = self
            .clickhouse
            .run_query_synchronous_no_params(query.to_string())
            .await?;

        // Create the `DemonstrationFeedback` table
        let table_engine_name = self.clickhouse.get_maybe_replicated_table_engine_name(
            GetMaybeReplicatedTableEngineNameArgs {
                table_engine_name: "MergeTree",
                table_name: "DemonstrationFeedback",
                engine_args: &[],
            },
        );
        let query = format!(
            r"
            CREATE TABLE IF NOT EXISTS DemonstrationFeedback{on_cluster_name}
            (
                id UUID, -- must be a UUIDv7
                inference_id UUID, -- must be a UUIDv7
                value String,
                timestamp DateTime MATERIALIZED UUIDv7ToDateTime(id)
            ) ENGINE = {table_engine_name}
            ORDER BY inference_id;
        ",
        );
        let _ = self
            .clickhouse
            .run_query_synchronous_no_params(query.to_string())
            .await?;

        // Create the `FloatMetricFeedback` table
        let table_engine_name = self.clickhouse.get_maybe_replicated_table_engine_name(
            GetMaybeReplicatedTableEngineNameArgs {
                table_engine_name: "MergeTree",
                table_name: "FloatMetricFeedback",
                engine_args: &[],
            },
        );
        let query = format!(
            r"
            CREATE TABLE IF NOT EXISTS FloatMetricFeedback{on_cluster_name}
            (
                id UUID, -- must be a UUIDv7
                target_id UUID, -- must be a UUIDv7
                metric_name LowCardinality(String),
                value Float32,
                timestamp DateTime MATERIALIZED UUIDv7ToDateTime(id)
            ) ENGINE = {table_engine_name}
            ORDER BY (metric_name, target_id);
        ",
        );
        let _ = self
            .clickhouse
            .run_query_synchronous_no_params(query.to_string())
            .await?;

        // Create the `ChatInference` table
        let table_engine_name = self.clickhouse.get_maybe_replicated_table_engine_name(
            GetMaybeReplicatedTableEngineNameArgs {
                table_engine_name: "MergeTree",
                table_name: "ChatInference",
                engine_args: &[],
            },
        );
        let query = format!(
            r"
            CREATE TABLE IF NOT EXISTS ChatInference{on_cluster_name}
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
            ) ENGINE = {table_engine_name}
            ORDER BY (function_name, variant_name, episode_id);
        ",
        );
        let _ = self
            .clickhouse
            .run_query_synchronous_no_params(query.to_string())
            .await?;

        // Create the `JsonInference` table
        let table_engine_name = self.clickhouse.get_maybe_replicated_table_engine_name(
            GetMaybeReplicatedTableEngineNameArgs {
                table_engine_name: "MergeTree",
                table_name: "JsonInference",
                engine_args: &[],
            },
        );
        let query = format!(
            r"
            CREATE TABLE IF NOT EXISTS JsonInference{on_cluster_name}
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
            ) ENGINE = {table_engine_name}
            ORDER BY (function_name, variant_name, episode_id);
        ",
        );
        let _ = self
            .clickhouse
            .run_query_synchronous_no_params(query.to_string())
            .await?;

        // Create the `ModelInference` table
        let table_engine_name = self.clickhouse.get_maybe_replicated_table_engine_name(
            GetMaybeReplicatedTableEngineNameArgs {
                table_engine_name: "MergeTree",
                table_name: "ModelInference",
                engine_args: &[],
            },
        );
        let query = format!(
            r"
            CREATE TABLE IF NOT EXISTS ModelInference{on_cluster_name}
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
            ) ENGINE = {table_engine_name}
            ORDER BY inference_id;
        ",
        );
        let _ = self
            .clickhouse
            .run_query_synchronous_no_params(query.to_string())
            .await?;

        Ok(())
    }

    fn rollback_instructions(&self) -> String {
        let database = self.clickhouse.database();
        let on_cluster_name = self.clickhouse.get_on_cluster_name();

        format!(
            "/* **CAREFUL: THIS WILL DELETE ALL DATA** */\
            /* Drop each table first (this seems to be required for replicated setups) */\
            DROP TABLE IF EXISTS BooleanMetricFeedback{on_cluster_name} SYNC;
            DROP TABLE IF EXISTS CommentFeedback{on_cluster_name} SYNC;
            DROP TABLE IF EXISTS DemonstrationFeedback{on_cluster_name} SYNC;
            DROP TABLE IF EXISTS FloatMetricFeedback{on_cluster_name} SYNC;
            DROP TABLE IF EXISTS ChatInference{on_cluster_name} SYNC;
            DROP TABLE IF EXISTS JsonInference{on_cluster_name} SYNC;
            DROP TABLE IF EXISTS ModelInference{on_cluster_name} SYNC;
            /* Even though this is created elsewhere we should */\
            /* drop it here to ensure a truly clean start.*/\
            DROP TABLE IF EXISTS TensorZeroMigration{on_cluster_name} SYNC;
            /* Drop the database */\
            DROP DATABASE IF EXISTS {database}{on_cluster_name};\
            /* **CAREFUL: THIS WILL DELETE ALL DATA** */"
        )
    }

    /// Check if the migration has succeeded (i.e. it should not be applied again)
    async fn has_succeeded(&self) -> Result<bool, Error> {
        let should_apply = self.should_apply().await?;
        Ok(!should_apply)
    }
}
