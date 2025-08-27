use super::check_table_exists;
use crate::db::clickhouse::migration_manager::migration_trait::Migration;
use crate::db::clickhouse::ClickHouseConnectionInfo;
use crate::error::{Error, ErrorDetails};
use async_trait::async_trait;

/// Migration 0034 contained a slightly incorrect implementation of the CumulativeUsageView that assumed
/// the input tokens would be null iff the output tokens were null.
/// This caused some extra clickhouse errors to occur when output_tokens were null but the input tokens were not.
/// We want to make sure that the columns are handled independently.
/// This PR tears alters the MV to a new one that handles the columns independently
/// with ifNull(.., 0).
pub struct Migration0035<'a> {
    pub clickhouse: &'a ClickHouseConnectionInfo,
}

const MIGRATION_ID: &str = "0035";

#[async_trait]
impl Migration for Migration0035<'_> {
    async fn can_apply(&self) -> Result<(), Error> {
        if !check_table_exists(self.clickhouse, "ModelInference", MIGRATION_ID).await? {
            return Err(Error::new(ErrorDetails::ClickHouseMigration {
                id: MIGRATION_ID.to_string(),
                message: "ModelInference table does not exist".to_string(),
            }));
        }
        let cumulative_usage_table_exists =
            check_table_exists(self.clickhouse, "CumulativeUsage", MIGRATION_ID).await?;
        if !cumulative_usage_table_exists {
            return Err(Error::new(ErrorDetails::ClickHouseMigration {
                id: MIGRATION_ID.to_string(),
                message: "CumulativeUsage table does not exist".to_string(),
            }));
        }
        let cumulative_usage_view_exists =
            check_table_exists(self.clickhouse, "CumulativeUsageView", MIGRATION_ID).await?;
        if !cumulative_usage_view_exists {
            return Err(Error::new(ErrorDetails::ClickHouseMigration {
                id: MIGRATION_ID.to_string(),
                message: "CumulativeUsageView table does not exist".to_string(),
            }));
        }
        Ok(())
    }

    async fn should_apply(&self) -> Result<bool, Error> {
        let create_view = self
            .clickhouse
            .run_query_synchronous_no_params("SHOW CREATE TABLE CumulativeUsageView".to_string())
            .await?
            .response;

        if create_view.contains("ifNull") {
            return Ok(false);
        }
        Ok(true)
    }

    async fn apply(&self, _clean_start: bool) -> Result<(), Error> {
        let on_cluster_name = self.clickhouse.get_on_cluster_name();
        self.clickhouse
            .run_query_synchronous_no_params(format!(
                r"ALTER TABLE CumulativeUsageView{on_cluster_name} MODIFY QUERY
                  SELECT
                    tupleElement(t, 1) AS type,
                    tupleElement(t, 2) AS count
                  FROM (
                    SELECT
                        arrayJoin([
                            tuple('input_tokens', ifNull(input_tokens, 0)),
                            tuple('output_tokens', ifNull(output_tokens, 0)),
                            tuple('model_inferences', 1)
                        ]) AS t
                  FROM ModelInference
                  WHERE input_tokens IS NOT NULL
                )
                "
            ))
            .await?;

        Ok(())
    }

    fn rollback_instructions(&self) -> String {
        // No reason to ever roll back to a buggy previous migration.
        // 0034 can drop
        "SELECT 1".to_string()
    }

    async fn has_succeeded(&self) -> Result<bool, Error> {
        let should_apply = self.should_apply().await?;
        Ok(!should_apply)
    }
}
