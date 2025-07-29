use async_trait::async_trait;

use super::{check_table_exists, create_replacing_table_engine, create_cluster_clause};
use crate::clickhouse::migration_manager::migration_trait::Migration;
use crate::clickhouse::ClickHouseConnectionInfo;
use crate::config_parser::Config;
use crate::error::Error;

/// This migration adds the `DynamicEvaluationRun` and `DynamicEvaluationRunEpisode` tables.
/// These support TensorZero's dynamic evaluations.
/// A `DynamicEvaluationRun` is a related set of `DynamicEvaluationRunEpisode`s with a common
/// set of variant pins and experiment tags.
/// A `DynamicEvaluationRunEpisode` is a single evaluation of a model variant under a given set of
/// variant pins and experiment tags.
/// 
/// As of the replication-aware migration system, this migration creates tables with
/// the appropriate engine (replicated vs non-replicated) based on configuration.
pub struct Migration0025<'a> {
    pub clickhouse: &'a ClickHouseConnectionInfo,
    pub config: &'a Config,
}

#[async_trait]
impl Migration for Migration0025<'_> {
    async fn can_apply(&self) -> Result<(), Error> {
        Ok(())
    }

    async fn should_apply(&self) -> Result<bool, Error> {
        let dynamic_evaluation_run_table_exists =
            check_table_exists(self.clickhouse, "DynamicEvaluationRun", "0025").await?;
        let dynamic_evaluation_run_episode_table_exists =
            check_table_exists(self.clickhouse, "DynamicEvaluationRunEpisode", "0025").await?;
        Ok(!dynamic_evaluation_run_table_exists || !dynamic_evaluation_run_episode_table_exists)
    }

    async fn apply(&self, _clean_start: bool) -> Result<(), Error> {
        let cluster_clause = create_cluster_clause(
            self.config.clickhouse.replication_enabled, 
            &self.config.clickhouse.cluster_name
        );
        
        let engine = create_replacing_table_engine(
            self.config.clickhouse.replication_enabled,
            &self.config.clickhouse.cluster_name,
            "DynamicEvaluationRun",
            Some("updated_at, is_deleted")
        );
        
        let query = format!(
            r"CREATE TABLE IF NOT EXISTS DynamicEvaluationRun {cluster_clause}
                (
                    run_id_uint UInt128, -- UUID encoded as a UInt128
                    variant_pins Map(String, String),
                    tags Map(String, String),
                    project_name Nullable(String),
                    run_display_name Nullable(String),
                    is_deleted Bool DEFAULT false,
                    updated_at DateTime64(6, 'UTC') DEFAULT now()
                ) ENGINE = {engine}
                ORDER BY run_id_uint;"
        );
        let _ = self
            .clickhouse
            .run_query_synchronous_no_params(query)
            .await?;

        let engine = create_replacing_table_engine(
            self.config.clickhouse.replication_enabled,
            &self.config.clickhouse.cluster_name,
            "DynamicEvaluationRunEpisode",
            Some("updated_at, is_deleted")
        );
        let query = format!(
            r"CREATE TABLE IF NOT EXISTS DynamicEvaluationRunEpisode {cluster_clause}
            (
                run_id UUID,
                episode_id_uint UInt128, -- UUID encoded as a UInt128
                -- this is duplicated so that we can look it up without joining at inference time
                variant_pins Map(String, String),
                datapoint_name Nullable(String),
                tags Map(String, String),
                is_deleted Bool DEFAULT false,
                updated_at DateTime64(6, 'UTC') DEFAULT now()
            ) ENGINE = {engine}
                ORDER BY episode_id_uint;"
        );
        let _ = self
            .clickhouse
            .run_query_synchronous_no_params(query)
            .await?;

        Ok(())
    }

    fn rollback_instructions(&self) -> String {
        "DROP TABLE IF EXISTS DynamicEvaluationRun\nDROP TABLE IF EXISTS DynamicEvaluationRunEpisode"
            .to_string()
    }

    async fn has_succeeded(&self) -> Result<bool, Error> {
        let should_apply = self.should_apply().await?;
        Ok(!should_apply)
    }
}
