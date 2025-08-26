use super::check_table_exists;
use crate::db::clickhouse::migration_manager::migration_trait::Migration;
use crate::db::clickhouse::{ClickHouseConnectionInfo, GetMaybeReplicatedTableEngineNameArgs};
use crate::error::{Error, ErrorDetails};
use crate::serde_util::deserialize_u64;
use async_trait::async_trait;
use serde::Deserialize;
use std::time::Duration;

/// This migration adds a `CumulativeUsage` table and `CumulativeUsageView` materialized view
/// This will allow the sum of tokens in the ModelInference table to be amortized and
/// looked up as needed.
/// NOTE: this migration is subsumed by migration_0035.rs
/// We will not BAN it but rather rely on migration manager handling.
pub struct Migration0034<'a> {
    pub clickhouse: &'a ClickHouseConnectionInfo,
}

const MIGRATION_ID: &str = "0034";

#[async_trait]
impl Migration for Migration0034<'_> {
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
        // If either the CumulativeUsage table or CumulativeUsageView view doesn't exist, we need to create it
        if !check_table_exists(self.clickhouse, "CumulativeUsage", MIGRATION_ID).await? {
            return Ok(true);
        }
        if !check_table_exists(self.clickhouse, "CumulativeUsageView", MIGRATION_ID).await? {
            return Ok(true);
        }
        Ok(false)
    }

    async fn apply(&self, clean_start: bool) -> Result<(), Error> {
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
        let on_cluster_name = self.clickhouse.get_on_cluster_name();
        let table_engine_name = self.clickhouse.get_maybe_replicated_table_engine_name(
            GetMaybeReplicatedTableEngineNameArgs {
                table_name: "CumulativeUsage",
                table_engine_name: "SummingMergeTree",
                engine_args: &[],
            },
        );
        self.clickhouse
            .run_query_synchronous_no_params(format!(
                r"CREATE TABLE IF NOT EXISTS CumulativeUsage{on_cluster_name} (
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
            format!("AND UUIDv7ToDateTime(id) >= fromUnixTimestamp64Nano({view_timestamp_nanos})")
        };
        let query = format!(
            r"
            CREATE MATERIALIZED VIEW IF NOT EXISTS CumulativeUsageView{on_cluster_name}
            TO CumulativeUsage
            AS
            SELECT
                tupleElement(t, 1) AS type,
                tupleElement(t, 2) AS count
            FROM (
                SELECT
                    arrayJoin([
                        tuple('input_tokens', input_tokens),
                        tuple('output_tokens', output_tokens),
                        tuple('model_inferences', 1)
                    ]) AS t
                FROM ModelInference
                WHERE input_tokens IS NOT NULL
                {view_where_clause}
            )
            "
        );
        let _ = self
            .clickhouse
            .run_query_synchronous_no_params(query)
            .await?;

        // NOTE: this migration is subsumed by 0035 so we do not need to run the backfill for this table any more
        // If we are not clean starting, we must backfill this table
        if !clean_start {
            tokio::time::sleep(view_offset).await;
            // Check if the materialized view we wrote is still in the table.
            // If this is the case, we should compute the backfilled sums and add them to the table.
            // Otherwise, we should warn that our view was not written (probably because a concurrent client did this first)
            // and conclude the migration.
            let create_table = self
                .clickhouse
                .run_query_synchronous_no_params(
                    "SHOW CREATE TABLE CumulativeUsageView".to_string(),
                )
                .await?
                .response;
            let view_timestamp_nanos_string = view_timestamp_nanos.to_string();
            if !create_table.contains(&view_timestamp_nanos_string) {
                tracing::warn!("Materialized view `CumulativeUsageView` was not written because it was recently created. This is likely due to a concurrent migration. Unless the other migration failed, no action is required.");
                return Ok(());
            }

            tracing::info!("Running backfill of CumulativeUsage");
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
                INSERT INTO CumulativeUsage (type, count) VALUES
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
        format!(
            r"
        DROP TABLE IF EXISTS CumulativeUsageView{on_cluster_name} SYNC;
        DROP TABLE IF EXISTS CumulativeUsage{on_cluster_name} SYNC;"
        )
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
