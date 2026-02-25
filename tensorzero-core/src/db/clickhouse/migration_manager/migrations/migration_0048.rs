use super::check_column_exists;
use super::check_table_exists;
use crate::db::clickhouse::ClickHouseConnectionInfo;
use crate::db::clickhouse::migration_manager::migration_trait::Migration;
use crate::error::{Error, ErrorDetails};
use async_trait::async_trait;
use std::time::Duration;

use super::migration_0037::quantiles_sql_args;

pub struct Migration0048<'a> {
    pub clickhouse: &'a ClickHouseConnectionInfo,
}

const MIGRATION_ID: &str = "0048";

#[async_trait]
impl Migration for Migration0048<'_> {
    async fn can_apply(&self) -> Result<(), Error> {
        if !check_table_exists(self.clickhouse, "ModelProviderStatistics", MIGRATION_ID).await? {
            return Err(Error::new(ErrorDetails::ClickHouseMigration {
                id: MIGRATION_ID.to_string(),
                message: "ModelProviderStatistics table does not exist".to_string(),
            }));
        }
        Ok(())
    }

    async fn should_apply(&self) -> Result<bool, Error> {
        Ok(!check_column_exists(
            self.clickhouse,
            "ModelProviderStatistics",
            "total_cost",
            MIGRATION_ID,
        )
        .await?)
    }

    async fn apply(&self, clean_start: bool) -> Result<(), Error> {
        let qs = quantiles_sql_args();
        let on_cluster_name = self.clickhouse.get_on_cluster_name();

        // 1. Add the total_cost column
        self.clickhouse
            .run_query_synchronous_no_params(format!(
                "ALTER TABLE ModelProviderStatistics{on_cluster_name} ADD COLUMN IF NOT EXISTS total_cost AggregateFunction(sum, Nullable(Decimal(18, 9)))"
            ))
            .await?;

        // 2. Record timestamp T (now + 15s offset)
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

        // 3. Drop the existing MV
        self.clickhouse
            .run_query_synchronous_no_params(format!(
                "DROP TABLE IF EXISTS ModelProviderStatisticsView{on_cluster_name} SYNC"
            ))
            .await?;

        // 4. Recreate the MV with the new total_cost column
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
                quantilesTDigestState({qs})(response_time_ms) as response_time_ms_quantiles,
                quantilesTDigestState({qs})(ttft_ms) as ttft_ms_quantiles,
                sumState(input_tokens) as total_input_tokens,
                sumState(output_tokens) as total_output_tokens,
                countState() as count,
                sumState(cost) as total_cost
            FROM ModelInference
            {view_where_clause}
            GROUP BY model_name, model_provider_name, minute
            "
        );
        self.clickhouse
            .run_query_synchronous_no_params(query)
            .await?;

        // 5. Backfill if needed
        if !clean_start {
            tokio::time::sleep(view_offset).await;

            let create_table = self
                .clickhouse
                .run_query_synchronous_no_params(
                    "SHOW CREATE TABLE ModelProviderStatisticsView".to_string(),
                )
                .await?
                .response;

            let view_timestamp_nanos_string = view_timestamp_nanos.to_string();
            if !create_table.contains(&view_timestamp_nanos_string) {
                tracing::warn!(
                    "Materialized view `ModelProviderStatisticsView` was not written because it was recently created. This is likely due to a concurrent migration. Unless the other migration failed, no action is required."
                );
                return Ok(());
            }

            // Only backfill the new `total_cost` column. The other columns (quantiles,
            // tokens, count) were already aggregated by migration_0037. Inserting only
            // `total_cost` lets AggregatingMergeTree merge the new partial row with
            // the existing one, adding cost without double-counting other metrics.
            tracing::info!("Running backfill of `ModelProviderStatistics` for `total_cost`");
            let query = format!(
                r"
                INSERT INTO ModelProviderStatistics
                    (model_name, model_provider_name, minute, total_cost)
                SELECT
                    model_name,
                    model_provider_name,
                    toStartOfMinute(timestamp) as minute,
                    sumState(cost) as total_cost
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
        let qs = quantiles_sql_args();
        let on_cluster_name = self.clickhouse.get_on_cluster_name();
        format!(
            r"
            DROP TABLE IF EXISTS ModelProviderStatisticsView{on_cluster_name} SYNC;
            ALTER TABLE ModelProviderStatistics{on_cluster_name} DROP COLUMN total_cost;
            CREATE MATERIALIZED VIEW IF NOT EXISTS ModelProviderStatisticsView{on_cluster_name} TO ModelProviderStatistics AS SELECT model_name, model_provider_name, toStartOfMinute(timestamp) as minute, quantilesTDigestState({qs})(response_time_ms) as response_time_ms_quantiles, quantilesTDigestState({qs})(ttft_ms) as ttft_ms_quantiles, sumState(input_tokens) as total_input_tokens, sumState(output_tokens) as total_output_tokens, countState() as count FROM ModelInference GROUP BY model_name, model_provider_name, minute;"
        )
    }

    async fn has_succeeded(&self) -> Result<bool, Error> {
        Ok(check_column_exists(
            self.clickhouse,
            "ModelProviderStatistics",
            "total_cost",
            MIGRATION_ID,
        )
        .await?)
    }
}
