use async_trait::async_trait;

use super::check_table_exists;
use crate::clickhouse::migration_manager::migration_trait::Migration;
use crate::clickhouse::ClickHouseConnectionInfo;
use crate::error::{Error, ErrorDetails};

/// This migration adds the `DynamicRunEpisodeByRunId` table and the
/// `DynamicRunEpisodeByRunIdView` materialized view.
/// These support consumption of dynamic evaluations indexed by run id.
/// The `DynamicRunEpisodeByRunId` table contains the same data as the
/// `DynamicEvaluationRunEpisode` table with different indexing.
pub struct Migration0026<'a> {
    pub clickhouse: &'a ClickHouseConnectionInfo,
}

#[async_trait]
impl Migration for Migration0026<'_> {
    async fn can_apply(&self) -> Result<(), Error> {
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
        Ok(!dynamic_evaluation_run_episode_by_run_id_table_exists
            || !dynamic_evaluation_run_episode_by_run_id_view_exists)
    }

    async fn apply(&self) -> Result<(), Error> {
        let query = r#"
            CREATE TABLE IF NOT EXISTS DynamicEvaluationRunEpisodeByRunId
                (
                    run_id_uint UInt128, -- UUID encoded as a UInt128
                    episode_id_uint UInt128, -- UUID encoded as a UInt128
                    variant_pins Map(String, String),
                    tags Map(String, String),
                    datapoint_name Nullable(String),
                    is_deleted Bool DEFAULT false,
                    updated_at DateTime64(6, 'UTC') DEFAULT now()
                ) ENGINE = ReplacingMergeTree(updated_at, is_deleted)
                ORDER BY (run_id_uint, episode_id_uint);
        "#;
        let _ = self
            .clickhouse
            .run_query_synchronous(query.to_string(), None)
            .await?;

        let query = r#"
            CREATE MATERIALIZED VIEW IF NOT EXISTS DynamicEvaluationRunEpisodeByRunIdView
                TO DynamicEvaluationRunEpisodeByRunId
                AS
                SELECT * EXCEPT run_id, toUInt128(run_id) AS run_id_uint FROM DynamicEvaluationRunEpisode
                ORDER BY run_id_uint, episode_id_uint;
        "#;
        let _ = self
            .clickhouse
            .run_query_synchronous(query.to_string(), None)
            .await?;

        Ok(())
    }

    fn rollback_instructions(&self) -> String {
        "\
        -- Drop the materialized view\n\
        DROP MATERIALIZED VIEW IF EXISTS DynamicEvaluationRunEpisodeByRunIdView;\n\
        -- Drop the `DynamicEvaluationRunEpisodeByRunId` table\n\
        DROP TABLE IF EXISTS DynamicEvaluationRunEpisodeByRunId;
        "
        .to_string()
    }

    async fn has_succeeded(&self) -> Result<bool, Error> {
        let should_apply = self.should_apply().await?;
        Ok(!should_apply)
    }
}
