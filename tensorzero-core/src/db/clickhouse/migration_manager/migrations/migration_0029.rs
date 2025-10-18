use crate::db::clickhouse::migration_manager::migration_trait::Migration;
use crate::db::clickhouse::ClickHouseConnectionInfo;
use crate::error::Error;
use async_trait::async_trait;

use super::check_table_exists;

/// ============================================================================
/// TERMINOLOGY NOTE:
/// "Static Evaluations" are now called "Inference Evaluations" in the project,
/// configuration, and user-facing documentation. However, the database table
/// names and view names referenced by this migration still use the "StaticEvaluation"
/// prefix (e.g., StaticEvaluationHumanFeedbackFloatView, StaticEvaluationHumanFeedbackBooleanView)
/// for backwards compatibility. This naming mismatch is intentional and should
/// not be changed in existing database objects to avoid breaking deployments.
/// ============================================================================
///
/// This migration completes the process started by migration_0028 of migrating away from 0023. We drop the old view
/// StaticEvaluationHumanFeedbackFloatView and StaticEvaluationHumanFeedbackBooleanView, which were subsumed in 0028.
/// Note: StaticEvaluation prefix retained for backwards compatibility (now called "Inference Evaluations")
pub struct Migration0029<'a> {
    pub clickhouse: &'a ClickHouseConnectionInfo,
}

const MIGRATION_ID: &str = "0029";

#[async_trait]
impl Migration for Migration0029<'_> {
    async fn can_apply(&self) -> Result<(), Error> {
        Ok(())
    }

    async fn should_apply(&self) -> Result<bool, Error> {
        // Note: This migration is special in that in its original form it wouldn't run in a
        // clean start setting because migration 0023 was banned.
        // We want to write at least once that this migration was run to the TensorZeroMigration table
        // so that we can skip the migrations if the table is full.
        // This migration "cheats" by checking if the migration manager has written that this migration has
        // already been run successfully.
        // If not, we run this once, the migration manager will write the row, and we will skip it every subsequent time.
        let response = self
            .clickhouse
            .run_query_synchronous_no_params(
                "SELECT 1 FROM TensorZeroMigration WHERE migration_id = 29 LIMIT 1".to_string(),
            )
            .await?;
        return Ok(response.response.trim() != "1");
    }

    async fn apply(&self, _clean_start: bool) -> Result<(), Error> {
        // Note: StaticEvaluation prefix retained for backwards compatibility (now called "Inference Evaluations")
        self.clickhouse
            .run_query_synchronous_no_params(
                r"DROP VIEW IF EXISTS StaticEvaluationHumanFeedbackFloatView;".to_string(),
            )
            .await?;
        self.clickhouse
            .run_query_synchronous_no_params(
                r"DROP VIEW IF EXISTS StaticEvaluationHumanFeedbackBooleanView;".to_string(),
            )
            .await?;

        Ok(())
    }

    fn rollback_instructions(&self) -> String {
        // We include 'SELECT 1' so that our test code can run these rollback instructions
        r"/* no action required */ SELECT 1;".to_string()
    }

    async fn has_succeeded(&self) -> Result<bool, Error> {
        // Note: StaticEvaluation prefix retained for backwards compatibility (now called "Inference Evaluations")
        let float_materialized_view_exists = check_table_exists(
            self.clickhouse,
            "StaticEvaluationHumanFeedbackFloatView",
            MIGRATION_ID,
        )
        .await?;
        let boolean_materialized_view_exists = check_table_exists(
            self.clickhouse,
            "StaticEvaluationHumanFeedbackBooleanView",
            MIGRATION_ID,
        )
        .await?;
        Ok(!float_materialized_view_exists && !boolean_materialized_view_exists)
    }
}
