use async_trait::async_trait;

use super::check_table_exists;
use crate::db::clickhouse::migration_manager::migration_trait::Migration;
use crate::db::clickhouse::{ClickHouseConnectionInfo, GetMaybeReplicatedTableEngineNameArgs};
use crate::error::Error;

/// This migration adds the `DynamicEvaluationRun` and `DynamicEvaluationRunEpisode` tables.
/// These support TensorZero's dynamic evaluations.
/// A `DynamicEvaluationRun` is a related set of `DynamicEvaluationRunEpisode`s with a common
/// set of variant pins and experiment tags.
/// A `DynamicEvaluationRunEpisode` is a single evaluation of a model variant under a given set of
/// variant pins and experiment tags.
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
        self.clickhouse
            .get_create_table_statements(
                "DynamicEvaluationRun",
                r"(
                    run_id_uint UInt128, -- UUID encoded as a UInt128
                    variant_pins Map(String, String),
                    tags Map(String, String),
                    project_name Nullable(String),
                    run_display_name Nullable(String),
                    is_deleted Bool DEFAULT false,
                    updated_at DateTime64(6, 'UTC') DEFAULT now()
                )",
                &GetMaybeReplicatedTableEngineNameArgs {
                    table_engine_name: "ReplacingMergeTree",
                    table_name: "DynamicEvaluationRun",
                    engine_args: &["updated_at", "is_deleted"],
                },
                Some("ORDER BY run_id_uint"),
                None,
            )
            .await?;

        self.clickhouse
            .get_create_table_statements(
                "DynamicEvaluationRunEpisode",
                r"(
                    run_id UUID,
                    episode_id_uint UInt128, -- UUID encoded as a UInt128
                    -- this is duplicated so that we can look it up without joining at inference time
                    variant_pins Map(String, String),
                    datapoint_name Nullable(String),
                    tags Map(String, String),
                    is_deleted Bool DEFAULT false,
                    updated_at DateTime64(6, 'UTC') DEFAULT now()
                )",
                &GetMaybeReplicatedTableEngineNameArgs {
                    table_engine_name: "ReplacingMergeTree",
                    table_name: "DynamicEvaluationRunEpisode",
                    engine_args: &["updated_at", "is_deleted"],
                },
                Some("ORDER BY episode_id_uint"),
                Some("cityHash64(toString(episode_id_uint))"),
            )
            .await?;

        Ok(())
    }

    fn rollback_instructions(&self) -> String {
        format!(
            "{}\
            {}",
            self.clickhouse.get_drop_table_rollback_statements("DynamicEvaluationRunEpisode"),
            self.clickhouse.get_drop_table_rollback_statements("DynamicEvaluationRun")
        )
    }

    async fn has_succeeded(&self) -> Result<bool, Error> {
        let should_apply = self.should_apply().await?;
        Ok(!should_apply)
    }
}
