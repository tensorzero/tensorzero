use async_trait::async_trait;

use super::check_table_exists;
use crate::clickhouse::migration_manager::migration_trait::Migration;
use crate::clickhouse::ClickHouseConnectionInfo;
use crate::error::Error;

/// This migration adds the `DynamicEvaluationRun` and `DynamicEvaluationRunTrial` tables.
/// These support TensorZero's dynamic evaluations.
/// A `DynamicEvaluationRun` is a related set of `DynamicEvaluationRunTrials` with a common
/// set of variant pins and experiment tags.
/// A `DynamicEvaluationRunTrial` is a single evaluation of a model variant under a given set of
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
        let dynamic_evaluation_run_trial_table_exists =
            check_table_exists(self.clickhouse, "DynamicEvaluationRunTrial", "0025").await?;
        Ok(!dynamic_evaluation_run_table_exists || !dynamic_evaluation_run_trial_table_exists)
    }

    async fn apply(&self) -> Result<(), Error> {
        let query = r#"
            CREATE TABLE IF NOT EXISTS DynamicEvaluationRun
                (
                    run_id UInt128, -- UUID encoded as a UInt128
                    variant_pins Map(String, String),
                    tags Map(String, String),
                    project_name Nullable(String),
                    run_display_name Nullable(String),
                    is_deleted Bool DEFAULT false,
                    updated_at DateTime64(6, 'UTC') DEFAULT now()
                ) ENGINE = ReplacingMergeTree(updated_at, is_deleted)
                ORDER BY run_id;
        "#;
        let _ = self.clickhouse.run_query_synchronous(query.to_string(), None).await?;

        let query = r#"
            CREATE TABLE IF NOT EXISTS DynamicEvaluationRunTrial
            (
                run_id UInt128, -- UUID encoded as a UInt128
                episode_id UInt128, -- UUID encoded as a UInt128
                
        "
    }
}
