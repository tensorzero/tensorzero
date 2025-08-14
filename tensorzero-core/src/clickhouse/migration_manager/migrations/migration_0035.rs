use super::check_table_exists;
use crate::clickhouse::migration_manager::migration_trait::Migration;
use crate::clickhouse::{ClickHouseConnectionInfo, GetMaybeReplicatedTableEngineNameArgs};
use crate::error::{Error, ErrorDetails};
use async_trait::async_trait;
use std::time::Duration;

/// This migration adds materialized view for latency quantiles and aggregated usage information for
/// models and model providers.
/// In particular, this creates the ModelProviderStatistics table.
/// and ModelProviderStatisticsView.
/// This stores quantiles of latency and TTFT for each model provider as well as
/// aggregated usage information for each model provider.
/// We aggregate by minute. (TODO: should we do hourly instead?)
pub struct Migration0035<'a> {
    pub clickhouse: &'a ClickHouseConnectionInfo,
}

const MIGRATION_ID: &str = "0035";
// TODO: are these the right quantiles? too many?
const QUANTILES: &str = "0.001, 0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10, 0.12, 0.14, 0.16, 0.18, 0.20, 0.22, 0.24, 0.26, 0.28, 0.30, 0.32, 0.34, 0.36, 0.38, 0.40, 0.42, 0.44, 0.46, 0.48, 0.50, 0.52, 0.54, 0.56, 0.58, 0.60, 0.62, 0.64, 0.66, 0.68, 0.70, 0.72, 0.74, 0.76, 0.78, 0.80, 0.82, 0.84, 0.86, 0.88, 0.90, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99, 0.991, 0.992, 0.993, 0.994, 0.995, 0.996, 0.997, 0.998, 0.999";

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
        if !check_table_exists(self.clickhouse, "ModelProviderStatistics", MIGRATION_ID).await? {
            return Ok(true);
        }
        if !check_table_exists(self.clickhouse, "ModelProviderStatisticsView", MIGRATION_ID).await?
        {
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
                table_name: "ModelProviderStatistics",
                table_engine_name: "AggregatingMergeTree",
                engine_args: &[],
            },
        );
        // TODO: handle cached inferences somehow
        self.clickhouse
            .run_query_synchronous_no_params(format!(
                r"CREATE TABLE IF NOT EXISTS ModelProviderStatistics{on_cluster_name} (
                        model_name LowCardinality(String),
                        model_provider_name LowCardinality(String),
                        minute DateTime,
                        response_time_ms_quantiles AggregateFunction(quantilesTDigest({QUANTILES}), Nullable(UInt32)),
                        ttft_ms_quantiles AggregateFunction(quantilesTDigest({QUANTILES}), Nullable(UInt32)),
                        total_input_tokens AggregateFunction(sum, Nullable(UInt32)),
                        total_output_tokens AggregateFunction(sum, Nullable(UInt32)),
                        count AggregateFunction(count, UInt32)
                    )
                    ENGINE = {table_engine_name}
                    ORDER BY (model_name, model_provider_name, minute)"
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
            CREATE MATERIALIZED VIEW IF NOT EXISTS ModelProviderStatisticsView{on_cluster_name}
            TO ModelProviderStatistics
            AS
            SELECT
                model_name,
                model_provider_name,
                toStartOfMinute(timestamp) as minute,

                quantilesTDigestState({QUANTILES})(response_time_ms) as response_time_ms_quantiles,
                quantilesTDigestState({QUANTILES})(ttft_ms) as ttft_ms_quantiles,
                sumState(input_tokens) as total_input_tokens,
                sumState(output_tokens) as total_output_tokens,
                countState() as count
            FROM ModelInference
            {view_where_clause}
            GROUP BY (model_name, model_provider_name, minute)
            "
        );
        let _ = self
            .clickhouse
            .run_query_synchronous_no_params(query)
            .await?;

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
                    "SHOW CREATE TABLE ModelProviderStatisticsView".to_string(),
                )
                .await?
                .response;
            let view_timestamp_nanos_string = view_timestamp_nanos.to_string();
            if !create_table.contains(&view_timestamp_nanos_string) {
                tracing::warn!("Materialized view `ModelProviderStatisticsView` was not written because it was recently created. This is likely due to a concurrent migration. Unless the other migration failed, no action is required.");
                return Ok(());
            }

            tracing::info!("Running backfill of ModelProviderStatistics");
            let query = format!(
                r"
                INSERT INTO ModelProviderStatistics
                SELECT
                    model_name,
                    model_provider_name,
                    toStartOfMinute(timestamp) as minute,

                    quantilesTDigestState({QUANTILES})(response_time_ms) as response_time_ms_quantiles,
                    quantilesTDigestState({QUANTILES})(ttft_ms) as ttft_ms_quantiles,
                    sumState(input_tokens) as total_input_tokens,
                    sumState(output_tokens) as total_output_tokens,
                    countState() as count
                FROM ModelInference
                WHERE UUIDv7ToDateTime(id) < fromUnixTimestamp64Nano({view_timestamp_nanos})
                GROUP BY model_name, model_provider_name, minute
                "
            );
            self.clickhouse
                .run_query_synchronous_no_params(query)
                .await?;
        }

        Ok(())
    }

    fn rollback_instructions(&self) -> String {
        let on_cluster_name = self.clickhouse.get_on_cluster_name();
        format!(
            r"
        DROP TABLE IF EXISTS ModelProviderStatisticsView{on_cluster_name} SYNC;
        DROP TABLE IF EXISTS ModelProviderView{on_cluster_name} SYNC;"
        )
    }

    async fn has_succeeded(&self) -> Result<bool, Error> {
        let should_apply = self.should_apply().await?;
        Ok(!should_apply)
    }
}

/*
*
* Example query:
*
* SELECT
    model_name,
    model_provider_name,
    minute,
    quantilesTDigestMerge(0.001, 0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.12, 0.14, 0.16, 0.18, 0.2, 0.22, 0.24, 0.26, 0.28, 0.3, 0.32, 0.34, 0.36, 0.38, 0.4, 0.42, 0.44, 0.46, 0.48, 0.5, 0.52, 0.54, 0.56, 0.58, 0.6, 0.62, 0.64, 0.66, 0.68, 0.7, 0.72, 0.74, 0.76, 0.78, 0.8, 0.82, 0.84, 0.86, 0.88, 0.9, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99, 0.991, 0.992, 0.993, 0.994, 0.995, 0.996, 0.997, 0.998, 0.999)(response_time_ms_quantiles) AS response_time_quantiles,
    sumMerge(total_input_tokens) AS total_input_tokens,
    sumMerge(total_output_tokens) AS total_output_tokens,
    countMerge(count) AS count
FROM ModelProviderStatistics
GROUP BY model_name, model_provider_name, minute
LIMIT 1;
*/
