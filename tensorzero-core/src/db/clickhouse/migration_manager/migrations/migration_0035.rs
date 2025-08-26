use super::check_table_exists;
use crate::db::clickhouse::migration_manager::migration_trait::Migration;
use crate::db::clickhouse::{ClickHouseConnectionInfo, GetMaybeReplicatedTableEngineNameArgs};
use crate::error::{Error, ErrorDetails};
use crate::serde_util::deserialize_u64;
use async_trait::async_trait;
use serde::Deserialize;
use tokio::time::Duration;

/// Migration 0034 contained a slightly incorrect implementation of the CumulativeUsageView that assumed
/// the input tokens would be null iff the output tokens were null.
/// This caused some extra clickhouse errors to occur when output_tokens were null but the input tokens were not.
/// We want to make sure that the columns are handled independently.
/// This PR tears down the old MV and in its stead initializes a new one that handles the columns independently
/// with ifNull(.., 0).
/// We also drop the underlying table and backfill it to ensure the data is consistent.
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
        Ok(())
    }

    async fn should_apply(&self) -> Result<bool, Error> {
        let cumulative_usage_table_exists =
            check_table_exists(self.clickhouse, "CumulativeUsage", MIGRATION_ID).await?;
        if cumulative_usage_table_exists {
            return Ok(true);
        }
        let cumulative_usage_view_exists =
            check_table_exists(self.clickhouse, "CumulativeUsageView", MIGRATION_ID).await?;
        if cumulative_usage_view_exists {
            return Ok(true);
        }
        let cumulative_usage_table_v2_exists =
            check_table_exists(self.clickhouse, "CumulativeUsageV2", MIGRATION_ID).await?;
        let cumulative_usage_view_v2_exists =
            check_table_exists(self.clickhouse, "CumulativeUsageViewV2", MIGRATION_ID).await?;
        // We should run the migration if either the table or the view for v2 is missing
        return Ok(!(cumulative_usage_view_v2_exists && cumulative_usage_table_v2_exists));
    }

    async fn apply(&self, clean_start: bool) -> Result<(), Error> {
        let on_cluster_name = self.clickhouse.get_on_cluster_name();
        self.clickhouse
            .run_query_synchronous_no_params(format!(
                r"DROP TABLE IF EXISTS CumulativeUsageView{on_cluster_name} SYNC"
            ))
            .await?;
        self.clickhouse
            .run_query_synchronous_no_params(format!(
                r"DROP TABLE IF EXISTS CumulativeUsage{on_cluster_name} SYNC"
            ))
            .await?;
        let view_offset = Duration::from_secs(15);
        let view_timestamp_nanos = (std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_err(|e| {
                Error::new(ErrorDetails::ClickHouseMigration {
                    id: MIGRATION_ID.to_string(),
                    message: e.to_string(),
                })
            })?
            + view_offset)
            .as_nanos();
        let table_engine_name = self.clickhouse.get_maybe_replicated_table_engine_name(
            GetMaybeReplicatedTableEngineNameArgs {
                table_name: "CumulativeUsageV2",
                table_engine_name: "SummingMergeTree",
                engine_args: &[],
            },
        );
        self.clickhouse
            .run_query_synchronous_no_params(format!(
                r"CREATE TABLE IF NOT EXISTS CumulativeUsageV2{on_cluster_name} (
                        type LowCardinality(String),
                        count UInt64,
                    )
                    ENGINE = {table_engine_name}
                    ORDER BY type;"
            ))
            .await?;
        // Create the materialized view for the CumulativeUsage table from ModelInference
        // If we are not doing a clean start, we need to add a where clause ot the view to only include rows that have been created
        // after the view_timestamp
        let view_where_clause = if clean_start {
            String::new()
        } else {
            format!("WHERE UUIDv7ToDateTime(id) >= fromUnixTimestamp64Nano({view_timestamp_nanos})")
        };
        let query = format!(
            r"
            CREATE MATERIALIZED VIEW IF NOT EXISTS CumulativeUsageViewV2{on_cluster_name}
            TO CumulativeUsageV2
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
                {view_where_clause}
            )
            "
        );
        let _ = self
            .clickhouse
            .run_query_synchronous_no_params(query)
            .await?;
        if !clean_start {
            tokio::time::sleep(view_offset).await;
            // Check if the materialized view we wrote is still in the table.
            // If this is the case, we should compute the backfilled sums and add them to the table.
            // Otherwise, we should warn that our view was not written (probably because a concurrent client did this first)
            // and conclude the migration.
            let create_table = self
                .clickhouse
                .run_query_synchronous_no_params(
                    "SHOW CREATE TABLE CumulativeUsageViewV2".to_string(),
                )
                .await?
                .response;
            let view_timestamp_nanos_string = view_timestamp_nanos.to_string();
            if !create_table.contains(&view_timestamp_nanos_string) {
                tracing::warn!("Materialized view `CumulativeUsageViewV2` was not written because it was recently created. This is likely due to a concurrent migration. Unless the other migration failed, no action is required.");
                return Ok(());
            }

            tracing::info!("Running backfill of CumulativeUsageV2");
            let query = format!(
                r"
                SELECT
                    sum(ifNull(input_tokens, 0)) as total_input_tokens,
                    sum(ifNull(output_tokens, 0)) as total_output_tokens,
                    COUNT(input_tokens) as total_count
                FROM ModelInference
                WHERE UUIDv7ToDateTime(id) < fromUnixTimestamp64Nano({view_timestamp_nanos})
                FORMAT JsonEachRow;
                "
            );
            let response = self
                .clickhouse
                .run_query_synchronous_no_params(query)
                .await?;
            let trimmed_response = response.response.trim();
            let parsed_response =
                serde_json::from_str::<CountResponse>(trimmed_response).map_err(|e| {
                    Error::new(ErrorDetails::ClickHouseDeserialization {
                        message: format!("Failed to deserialize count query: {e}"),
                    })
                })?;
            let CountResponse {
                total_input_tokens,
                total_output_tokens,
                total_count,
            } = parsed_response;

            let write_query = format!(
                r"
                INSERT INTO CumulativeUsageV2 (type, count) VALUES
                ('input_tokens', {total_input_tokens}),
                ('output_tokens', {total_output_tokens}),
                ('model_inferences', {total_count})
                "
            );
            self.clickhouse
                .run_query_synchronous_no_params(write_query)
                .await?;
        }

        Ok(())
    }

    fn rollback_instructions(&self) -> String {
        let on_cluster_name = self.clickhouse.get_on_cluster_name();
        format!("DROP TABLE IF EXISTS CumulativeUsageViewV2{on_cluster_name} SYNC;\nDROP TABLE IF EXISTS CumulativeUsageV2{on_cluster_name} SYNC;")
    }

    async fn has_succeeded(&self) -> Result<bool, Error> {
        let should_apply = self.should_apply().await?;
        Ok(!should_apply)
    }
}

#[derive(Debug, Deserialize)]
struct CountResponse {
    #[serde(deserialize_with = "deserialize_u64")]
    total_input_tokens: u64,
    #[serde(deserialize_with = "deserialize_u64")]
    total_output_tokens: u64,
    #[serde(deserialize_with = "deserialize_u64")]
    total_count: u64,
}
