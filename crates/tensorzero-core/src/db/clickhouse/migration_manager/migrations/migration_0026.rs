use async_trait::async_trait;

use super::check_table_exists;
use crate::db::clickhouse::migration_manager::migration_trait::Migration;
use crate::db::clickhouse::{ClickHouseConnectionInfo, GetMaybeReplicatedTableEngineNameArgs};
use crate::error::{Error, ErrorDetails};

/// This migration adds the `DynamicRunEpisodeByRunId` table and the
/// `DynamicRunEpisodeByRunIdView` materialized view.
/// It also adds the `DynamicEvaluationRunByProjectName` table and the
/// `DynamicEvaluationRunByProjectNameView` materialized view.
/// These support consumption of workflow evaluations (formerly "dynamic evaluations") indexed by run id and project name.
/// The `DynamicRunEpisodeByRunId` table contains the same data as the
/// `DynamicEvaluationRunEpisode` table with different indexing.
/// The `DynamicEvaluationRunByProjectName` table contains the same data as the
/// `DynamicEvaluationRun` table with different indexing.
///
/// IMPORTANT: These tables use "DynamicEvaluation" in their names for historical reasons.
/// Externally, this feature is now called "Workflow Evaluations" (renamed from "Dynamic Evaluations").
/// The table names remain unchanged to avoid complex data migrations.
pub struct Migration0026<'a> {
    pub clickhouse: &'a ClickHouseConnectionInfo,
}

#[async_trait]
impl Migration for Migration0026<'_> {
    async fn can_apply(&self) -> Result<(), Error> {
        let dynamic_evaluation_run_table_exists =
            check_table_exists(self.clickhouse, "DynamicEvaluationRun", "0027").await?;
        if !dynamic_evaluation_run_table_exists {
            return Err(ErrorDetails::ClickHouseMigration {
                id: "0027".to_string(),
                message: "DynamicEvaluationRun table does not exist".to_string(),
            }
            .into());
        }
        let dynamic_evaluation_run_episode_table_exists =
            check_table_exists(self.clickhouse, "DynamicEvaluationRunEpisode", "0026").await?;
        if !dynamic_evaluation_run_episode_table_exists {
            return Err(ErrorDetails::ClickHouseMigration {
                id: "0026".to_string(),
                message: "DynamicEvaluationRunEpisode table does not exist".to_string(),
            }
            .into());
        }
        Ok(())
    }

    async fn should_apply(&self) -> Result<bool, Error> {
        let dynamic_evaluation_run_episode_by_run_id_table_exists = check_table_exists(
            self.clickhouse,
            "DynamicEvaluationRunEpisodeByRunId",
            "0026",
        )
        .await?;
        let dynamic_evaluation_run_episode_by_run_id_view_exists = check_table_exists(
            self.clickhouse,
            "DynamicEvaluationRunEpisodeByRunIdView",
            "0026",
        )
        .await?;
        let dynamic_evaluation_run_by_project_name_table_exists =
            check_table_exists(self.clickhouse, "DynamicEvaluationRunByProjectName", "0027")
                .await?;
        let dynamic_evaluation_run_by_project_name_view_exists = check_table_exists(
            self.clickhouse,
            "DynamicEvaluationRunByProjectNameView",
            "0027",
        )
        .await?;

        Ok(!dynamic_evaluation_run_episode_by_run_id_table_exists
            || !dynamic_evaluation_run_episode_by_run_id_view_exists
            || !dynamic_evaluation_run_by_project_name_table_exists
            || !dynamic_evaluation_run_by_project_name_view_exists)
    }

    async fn apply(&self, _clean_start: bool) -> Result<(), Error> {
        let on_cluster_name = self.clickhouse.get_on_cluster_name();
        let table_engine_name = self.clickhouse.get_maybe_replicated_table_engine_name(
            GetMaybeReplicatedTableEngineNameArgs {
                table_engine_name: "ReplacingMergeTree",
                table_name: "DynamicEvaluationRunEpisodeByRunId",
                engine_args: &["updated_at", "is_deleted"],
            },
        );
        let query = format!(
            r"
            CREATE TABLE IF NOT EXISTS DynamicEvaluationRunEpisodeByRunId{on_cluster_name}
                (
                    run_id_uint UInt128, -- UUID encoded as a UInt128
                    episode_id_uint UInt128, -- UUID encoded as a UInt128
                    variant_pins Map(String, String),
                    tags Map(String, String),
                    datapoint_name Nullable(String), -- externally: task_name (TODO: rename in a future migration)
                    is_deleted Bool DEFAULT false,
                    updated_at DateTime64(6, 'UTC') DEFAULT now()
                ) ENGINE = {table_engine_name}
                ORDER BY (run_id_uint, episode_id_uint);
        "
        );
        let _ = self
            .clickhouse
            .run_query_synchronous_no_params(query.to_string())
            .await?;

        let query = format!(
            r"
            CREATE MATERIALIZED VIEW IF NOT EXISTS DynamicEvaluationRunEpisodeByRunIdView{on_cluster_name}
                TO DynamicEvaluationRunEpisodeByRunId
                AS
                SELECT * EXCEPT run_id, toUInt128(run_id) AS run_id_uint FROM DynamicEvaluationRunEpisode
                ORDER BY run_id_uint, episode_id_uint;
        "
        );
        let _ = self
            .clickhouse
            .run_query_synchronous_no_params(query.to_string())
            .await?;

        let table_engine_name = self.clickhouse.get_maybe_replicated_table_engine_name(
            GetMaybeReplicatedTableEngineNameArgs {
                table_engine_name: "ReplacingMergeTree",
                table_name: "DynamicEvaluationRunByProjectName",
                engine_args: &["updated_at", "is_deleted"],
            },
        );
        let query = format!(
            r"
            CREATE TABLE IF NOT EXISTS DynamicEvaluationRunByProjectName{on_cluster_name}
                (
                    run_id_uint UInt128, -- UUID encoded as a UInt128
                    variant_pins Map(String, String),
                    tags Map(String, String),
                    project_name String,
                    run_display_name Nullable(String),
                    is_deleted Bool DEFAULT false,
                    updated_at DateTime64(6, 'UTC') DEFAULT now()
                ) ENGINE = {table_engine_name}
                ORDER BY (project_name, run_id_uint);
        ",
        );
        let _ = self
            .clickhouse
            .run_query_synchronous_no_params(query.to_string())
            .await?;

        let query = format!(
            r"
            CREATE MATERIALIZED VIEW IF NOT EXISTS DynamicEvaluationRunByProjectNameView{on_cluster_name}
                TO DynamicEvaluationRunByProjectName
                AS
                SELECT * FROM DynamicEvaluationRun
                WHERE project_name IS NOT NULL
                ORDER BY project_name, run_id_uint;
        "
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
            "/* Drop the materialized views */\
            DROP VIEW IF EXISTS DynamicEvaluationRunEpisodeByRunIdView{on_cluster_name};
            DROP VIEW IF EXISTS DynamicEvaluationRunByProjectNameView{on_cluster_name};
            /* Drop the tables */\
            DROP TABLE IF EXISTS DynamicEvaluationRunEpisodeByRunId{on_cluster_name} SYNC;
            DROP TABLE IF EXISTS DynamicEvaluationRunByProjectName{on_cluster_name} SYNC;
            "
        )
    }

    async fn has_succeeded(&self) -> Result<bool, Error> {
        let should_apply = self.should_apply().await?;
        Ok(!should_apply)
    }
}
