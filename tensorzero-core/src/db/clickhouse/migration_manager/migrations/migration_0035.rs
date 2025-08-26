use super::check_table_exists;
use crate::db::clickhouse::migration_manager::migration_trait::Migration;
use crate::db::clickhouse::ClickHouseConnectionInfo;
use crate::error::{Error, ErrorDetails};
use async_trait::async_trait;

/// Migration 0034 contained a slightly incorrect implementation of the CumulativeUsageView that assumed
/// the input tokens would be null iff the output tokens were null.
/// This caused some extra clickhouse errors to occur when output_tokens were null but the input tokens were not.
/// We want to make sure that the columns are handled independently.
/// This PR tears down the old MV and in it's stead initializes a new one that handles the columns independently
/// with ifNull(.., 0).
/// We don't need a cutoff since we already backfilled the data here. However, we may lose a small amount of data
/// for cumulative usage in between the drop and the recreation.
/// This is acceptable since the CumulativeUsage table is not used by the frontend.
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
        if !check_table_exists(self.clickhouse, "CumulativeUsage", MIGRATION_ID).await? {
            return Err(Error::new(ErrorDetails::ClickHouseMigration {
                id: MIGRATION_ID.to_string(),
                message: "CumulativeUsage table does not exist".to_string(),
            }));
        }
        Ok(())
    }

    async fn should_apply(&self) -> Result<bool, Error> {
        let cumulative_usage_view_exists =
            check_table_exists(self.clickhouse, "CumulativeUsageView", MIGRATION_ID).await?;
        if cumulative_usage_view_exists {
            return Ok(true);
        }
        let cumulative_usage_view_v2_exists =
            check_table_exists(self.clickhouse, "CumulativeUsageViewV2", MIGRATION_ID).await?;
        return Ok(!cumulative_usage_view_v2_exists);
    }

    async fn apply(&self, _clean_start: bool) -> Result<(), Error> {
        let on_cluster_name = self.clickhouse.get_on_cluster_name();

        self.clickhouse
            .run_query_synchronous_no_params(format!(
                r"DROP TABLE IF EXISTS CumulativeUsageView{on_cluster_name} SYNC"
            ))
            .await?;

        let query = format!(
            r"
            CREATE MATERIALIZED VIEW IF NOT EXISTS CumulativeUsageViewV2{on_cluster_name}
            TO CumulativeUsage
            AS
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
            )
            "
        );
        let _ = self
            .clickhouse
            .run_query_synchronous_no_params(query)
            .await?;

        Ok(())
    }

    fn rollback_instructions(&self) -> String {
        let on_cluster_name = self.clickhouse.get_on_cluster_name();
        format!("DROP TABLE IF EXISTS CumulativeUsageViewV2{on_cluster_name} SYNC;")
    }

    async fn has_succeeded(&self) -> Result<bool, Error> {
        let should_apply = self.should_apply().await?;
        Ok(!should_apply)
    }
}
