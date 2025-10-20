use async_trait::async_trait;

use super::check_table_exists;
use crate::db::clickhouse::migration_manager::migration_trait::Migration;
use crate::db::clickhouse::{ClickHouseConnectionInfo, GetMaybeReplicatedTableEngineNameArgs};
use crate::error::Error;

/// This migration adds the `DynamicEvaluationRun` and `DynamicEvaluationRunEpisode` tables.
/// These support TensorZero's workflow evaluations (formerly called "dynamic evaluations").
/// A `DynamicEvaluationRun` is a related set of `DynamicEvaluationRunEpisode`s with a common
/// set of variant pins and experiment tags.
/// A `DynamicEvaluationRunEpisode` is a single evaluation of a model variant under a given set of
/// variant pins and experiment tags.
///
/// IMPORTANT: These tables use "DynamicEvaluation" in their names for historical reasons.
/// Externally, this feature is now called "Workflow Evaluations" (renamed from "Dynamic Evaluations").
/// The table names remain unchanged to avoid complex data migrations.
/// All internal code has been updated to use "workflow_evaluation" terminology,
/// but continues to read/write to these "DynamicEvaluation" tables.
pub struct Migration0025<'a> {
    pub clickhouse: &'a ClickHouseConnectionInfo,
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
        let on_cluster_name = self.clickhouse.get_on_cluster_name();
        let table_engine_name = self.clickhouse.get_maybe_replicated_table_engine_name(
            GetMaybeReplicatedTableEngineNameArgs {
                table_engine_name: "ReplacingMergeTree",
                table_name: "DynamicEvaluationRun",
                engine_args: &["updated_at", "is_deleted"],
            },
        );
        let query = format!(
            r"
            CREATE TABLE IF NOT EXISTS DynamicEvaluationRun{on_cluster_name}
                (
                    run_id_uint UInt128, -- UUID encoded as a UInt128
                    variant_pins Map(String, String),
                    tags Map(String, String),
                    project_name Nullable(String),
                    run_display_name Nullable(String),
                    is_deleted Bool DEFAULT false,
                    updated_at DateTime64(6, 'UTC') DEFAULT now()
                ) ENGINE = {table_engine_name}
                ORDER BY run_id_uint;
        ",
        );
        let _ = self
            .clickhouse
            .run_query_synchronous_no_params(query.to_string())
            .await?;

        let table_engine_name = self.clickhouse.get_maybe_replicated_table_engine_name(
            GetMaybeReplicatedTableEngineNameArgs {
                table_engine_name: "ReplacingMergeTree",
                table_name: "DynamicEvaluationRunEpisode",
                engine_args: &["updated_at", "is_deleted"],
            },
        );
        let query = format!(
            r"
            CREATE TABLE IF NOT EXISTS DynamicEvaluationRunEpisode{on_cluster_name}
            (
                run_id UUID,
                episode_id_uint UInt128, -- UUID encoded as a UInt128
                -- this is duplicated so that we can look it up without joining at inference time
                variant_pins Map(String, String),
                datapoint_name Nullable(String), -- externally: task_name (TODO: rename in a future migration)
                tags Map(String, String),
                is_deleted Bool DEFAULT false,
                updated_at DateTime64(6, 'UTC') DEFAULT now()
            ) ENGINE = {table_engine_name}
                ORDER BY episode_id_uint;
        ",
        );
        let _ = self
            .clickhouse
            .run_query_synchronous_no_params(query.to_string())
            .await?;

        Ok(())
    }

    fn rollback_instructions(&self) -> String {
        let on_cluster_name = self.clickhouse.get_on_cluster_name();
        format!(
            r"
            DROP TABLE IF EXISTS DynamicEvaluationRunEpisode{on_cluster_name} SYNC;
            DROP TABLE IF EXISTS DynamicEvaluationRun{on_cluster_name} SYNC;
        "
        )
    }

    async fn has_succeeded(&self) -> Result<bool, Error> {
        let should_apply = self.should_apply().await?;
        Ok(!should_apply)
    }
}
