use crate::clickhouse::migration_manager::migration_trait::Migration;
use crate::clickhouse::ClickHouseConnectionInfo;
use crate::error::Error;
use async_trait::async_trait;

use super::check_table_exists;

/// This migration completes the process started by migration_0028 of migrating away from 0023. We drop the old view
/// StaticEvaluationHumanFeedbackFloatView and StaticEvaluationHumanFeedbackBooleanView, which were subsumed in 0028.
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
        // We need to run this migration if either the float or boolean materialized views exist.
        Ok(float_materialized_view_exists || boolean_materialized_view_exists)
    }

    async fn apply(&self, _clean_start: bool) -> Result<(), Error> {
        self.clickhouse
            .run_query_synchronous(
                r#"DROP VIEW IF EXISTS StaticEvaluationHumanFeedbackFloatView;"#.to_string(),
                None,
            )
            .await?;
        self.clickhouse
            .run_query_synchronous(
                r#"DROP VIEW IF EXISTS StaticEvaluationHumanFeedbackBooleanView;"#.to_string(),
                None,
            )
            .await?;

        Ok(())
    }

    fn rollback_instructions(&self) -> String {
        // We include 'SELECT 1' so that our test code can run these rollback instructions
        r#"/* no action required */ SELECT 1;"#.to_string()
    }

    async fn has_succeeded(&self) -> Result<bool, Error> {
        let should_apply = self.should_apply().await?;
        Ok(!should_apply)
    }
}
