use crate::clickhouse::migration_manager::migration_trait::Migration;
use crate::clickhouse::migration_manager::migrations::{
    check_table_exists, create_table_engine, create_cluster_clause
};
use crate::clickhouse::ClickHouseConnectionInfo;
use crate::config_parser::Config;
use crate::error::Error;
use async_trait::async_trait;

/// This migration is used to create the initial tables in the ClickHouse database.
/// 
/// As of the replication-aware migration system, this migration creates tables with
/// the appropriate engine (replicated vs non-replicated) based on configuration.
/// This ensures new installations get the right table types from the start.
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
    pub config: &'a Config,
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
        let cluster_clause = create_cluster_clause(
            self.config.clickhouse.replication_enabled, 
            &self.config.clickhouse.cluster_name
        );
        
        // Create the `BooleanMetricFeedback` table
        let engine = create_table_engine(
            self.config.clickhouse.replication_enabled,
            &self.config.clickhouse.cluster_name,
            "BooleanMetricFeedback"
        );
        let query = format!(
            r"CREATE TABLE IF NOT EXISTS BooleanMetricFeedback {cluster_clause}
            (
                id UUID, -- must be a UUIDv7
                target_id UUID, -- must be a UUIDv7
                metric_name LowCardinality(String),
                value Bool,
                timestamp DateTime MATERIALIZED UUIDv7ToDateTime(id)
            ) ENGINE = {engine}
            ORDER BY (metric_name, target_id);"
        );
        let _ = self
            .clickhouse
            .run_query_synchronous_no_params(query)
            .await?;

        // Create the `CommentFeedback` table
        let engine = create_table_engine(
            self.config.clickhouse.replication_enabled,
            &self.config.clickhouse.cluster_name,
            "CommentFeedback"
        );
        let query = format!(
            r"CREATE TABLE IF NOT EXISTS CommentFeedback {cluster_clause}
            (
                id UUID, -- must be a UUIDv7
                target_id UUID, -- must be a UUIDv7
                target_type Enum('inference' = 1, 'episode' = 2),
                value String,
                timestamp DateTime MATERIALIZED UUIDv7ToDateTime(id)
            ) ENGINE = {engine}
            ORDER BY target_id;"
        );
        let _ = self
            .clickhouse
            .run_query_synchronous_no_params(query)
            .await?;

        // Create the `DemonstrationFeedback` table
        let engine = create_table_engine(
            self.config.clickhouse.replication_enabled,
            &self.config.clickhouse.cluster_name,
            "DemonstrationFeedback"
        );
        let query = format!(
            r"CREATE TABLE IF NOT EXISTS DemonstrationFeedback {cluster_clause}
            (
                id UUID, -- must be a UUIDv7
                inference_id UUID, -- must be a UUIDv7
                value String,
                timestamp DateTime MATERIALIZED UUIDv7ToDateTime(id)
            ) ENGINE = {engine}
            ORDER BY inference_id;"
        );
        let _ = self
            .clickhouse
            .run_query_synchronous_no_params(query)
            .await?;

        // Create the `FloatMetricFeedback` table
        let engine = create_table_engine(
            self.config.clickhouse.replication_enabled,
            &self.config.clickhouse.cluster_name,
            "FloatMetricFeedback"
        );
        let query = format!(
            r"CREATE TABLE IF NOT EXISTS FloatMetricFeedback {cluster_clause}
            (
                id UUID, -- must be a UUIDv7
                target_id UUID, -- must be a UUIDv7
                metric_name LowCardinality(String),
                value Float32,
                timestamp DateTime MATERIALIZED UUIDv7ToDateTime(id)
            ) ENGINE = {engine}
            ORDER BY (metric_name, target_id);"
        );
        let _ = self
            .clickhouse
            .run_query_synchronous_no_params(query)
            .await?;

        // Create the `ChatInference` table
        let engine = create_table_engine(
            self.config.clickhouse.replication_enabled,
            &self.config.clickhouse.cluster_name,
            "ChatInference"
        );
        let query = format!(
            r"CREATE TABLE IF NOT EXISTS ChatInference {cluster_clause}
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
            ) ENGINE = {engine}
            ORDER BY (function_name, variant_name, episode_id);"
        );
        let _ = self
            .clickhouse
            .run_query_synchronous_no_params(query)
            .await?;

        // Create the `JsonInference` table
        let engine = create_table_engine(
            self.config.clickhouse.replication_enabled,
            &self.config.clickhouse.cluster_name,
            "JsonInference"
        );
        let query = format!(
            r"CREATE TABLE IF NOT EXISTS JsonInference {cluster_clause}
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
            ) ENGINE = {engine}
            ORDER BY (function_name, variant_name, episode_id);"
        );
        let _ = self
            .clickhouse
            .run_query_synchronous_no_params(query)
            .await?;

        // Create the `ModelInference` table
        let engine = create_table_engine(
            self.config.clickhouse.replication_enabled,
            &self.config.clickhouse.cluster_name,
            "ModelInference"
        );
        let query = format!(
            r"CREATE TABLE IF NOT EXISTS ModelInference {cluster_clause}
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
            ) ENGINE = {engine}
            ORDER BY inference_id;"
        );
        let _ = self
            .clickhouse
            .run_query_synchronous_no_params(query)
            .await?;

        Ok(())
    }

    fn rollback_instructions(&self) -> String {
        let database = self.clickhouse.database();

        format!(
            "/* **CAREFUL: THIS WILL DELETE ALL DATA** */\
            /* Drop the database */\
            DROP DATABASE IF EXISTS {database};\
            /* **CAREFUL: THIS WILL DELETE ALL DATA** */"
        )
    }

    /// Check if the migration has succeeded (i.e. it should not be applied again)
    async fn has_succeeded(&self) -> Result<bool, Error> {
        let should_apply = self.should_apply().await?;
        Ok(!should_apply)
    }
}
