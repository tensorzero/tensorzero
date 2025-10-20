use crate::db::clickhouse::migration_manager::migration_trait::Migration;
use crate::db::clickhouse::ClickHouseConnectionInfo;
use crate::error::Error;
use async_trait::async_trait;

use super::check_detached_table_exists;

/// ============================================================================
/// TERMINOLOGY NOTE:
/// "Static Evaluations" are now called "Inference Evaluations" in the project,
/// configuration, and user-facing documentation. However, the database table
/// names and view names referenced by this migration still use the "StaticEvaluation"
/// prefix (e.g., StaticEvaluationHumanFeedback, StaticEvaluationFloatHumanFeedbackView,
/// StaticEvaluationBooleanHumanFeedbackView) for backwards compatibility.
/// This naming mismatch is intentional and should not be changed in existing
/// database objects to avoid breaking deployments.
/// ============================================================================
///
/// Migration 0028 set up a materialized view for our StaticEvaluationHumanFeedback table.
/// Note: StaticEvaluation prefix retained for backwards compatibility (now called "Inference Evaluations")
/// This view used some prefiltered joins to get the data needed.
/// However, this was causing memory issues for high-throughput feedback ingestion.
/// So we've decided to handle this logic client-side in the feedback handler rather than
/// delegate this to ClickHouse.
/// This migration therefore disables the
/// StaticEvaluationFloatHumanFeedback and StaticEvaluationBooleanHumanFeedback views.
/// Note: StaticEvaluation prefix retained for backwards compatibility (now called "Inference Evaluations")
/// We leave them around in disabled form so as to avoid breaking the 0028 should_apply check.
pub struct Migration0036<'a> {
    pub clickhouse: &'a ClickHouseConnectionInfo,
}

const MIGRATION_ID: &str = "0036";

#[async_trait]
impl Migration for Migration0036<'_> {
    async fn can_apply(&self) -> Result<(), Error> {
        Ok(())
    }

    async fn should_apply(&self) -> Result<bool, Error> {
        // Note: StaticEvaluation prefix retained for backwards compatibility (now called "Inference Evaluations")
        if !check_detached_table_exists(
            self.clickhouse,
            "StaticEvaluationFloatHumanFeedbackView",
            MIGRATION_ID,
        )
        .await?
        {
            return Ok(true);
        }
        if !check_detached_table_exists(
            self.clickhouse,
            "StaticEvaluationBooleanHumanFeedbackView",
            MIGRATION_ID,
        )
        .await?
        {
            return Ok(true);
        }
        Ok(false)
    }

    async fn apply(&self, _clean_start: bool) -> Result<(), Error> {
        let on_cluster_name = self.clickhouse.get_on_cluster_name();
        // Note: StaticEvaluation prefix retained for backwards compatibility (now called "Inference Evaluations")
        self.clickhouse
            .run_query_synchronous_no_params(format!(
                r"DETACH TABLE IF EXISTS StaticEvaluationBooleanHumanFeedbackView{on_cluster_name} PERMANENTLY;"
            ))
            .await?;
        self.clickhouse
            .run_query_synchronous_no_params(format!(
                r"DETACH TABLE IF EXISTS StaticEvaluationFloatHumanFeedbackView{on_cluster_name} PERMANENTLY;"
            ))
            .await?;

        Ok(())
    }

    fn rollback_instructions(&self) -> String {
        // No reason to ever roll back to a buggy previous migration.
        // 0028 can drop
        "SELECT 1".to_string()
    }

    async fn has_succeeded(&self) -> Result<bool, Error> {
        let should_apply = self.should_apply().await?;
        Ok(!should_apply)
    }
}
