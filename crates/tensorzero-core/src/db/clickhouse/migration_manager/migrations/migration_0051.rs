use super::check_column_exists;
use super::check_table_exists;
use crate::db::clickhouse::ClickHouseConnectionInfo;
use crate::db::clickhouse::migration_manager::migration_trait::Migration;
use crate::error::{Error, ErrorDetails};
use async_trait::async_trait;
use std::time::Duration;

use super::migration_0037::quantiles_sql_args;

/// This migration adds `provider_cache_read_input_tokens` and `provider_cache_write_input_tokens` columns
/// to `ModelInference`, and corresponding aggregate columns to `ModelProviderStatistics`.
/// This brings ClickHouse in parity with the Postgres migration
/// `20260313000000_cache_token_columns.sql`.
pub struct Migration0051<'a> {
    pub clickhouse: &'a ClickHouseConnectionInfo,
}

const MIGRATION_ID: &str = "0051";

#[async_trait]
impl Migration for Migration0051<'_> {
    async fn can_apply(&self) -> Result<(), Error> {
        if !check_table_exists(self.clickhouse, "ModelInference", MIGRATION_ID).await? {
            return Err(Error::new(ErrorDetails::ClickHouseMigration {
                id: MIGRATION_ID.to_string(),
                message: "ModelInference table does not exist".to_string(),
            }));
        }
        if !check_table_exists(self.clickhouse, "ModelProviderStatistics", MIGRATION_ID).await? {
            return Err(Error::new(ErrorDetails::ClickHouseMigration {
                id: MIGRATION_ID.to_string(),
                message: "ModelProviderStatistics table does not exist".to_string(),
            }));
        }
        Ok(())
    }

    async fn should_apply(&self) -> Result<bool, Error> {
        // Check all four columns so a partially applied migration is re-attempted.
        let mi_read = check_column_exists(
            self.clickhouse,
            "ModelInference",
            "provider_cache_read_input_tokens",
            MIGRATION_ID,
        )
        .await?;
        let mi_write = check_column_exists(
            self.clickhouse,
            "ModelInference",
            "provider_cache_write_input_tokens",
            MIGRATION_ID,
        )
        .await?;
        let stats_read = check_column_exists(
            self.clickhouse,
            "ModelProviderStatistics",
            "total_provider_cache_read_input_tokens",
            MIGRATION_ID,
        )
        .await?;
        let stats_write = check_column_exists(
            self.clickhouse,
            "ModelProviderStatistics",
            "total_provider_cache_write_input_tokens",
            MIGRATION_ID,
        )
        .await?;
        Ok(!(mi_read && mi_write && stats_read && stats_write))
    }

    async fn apply(&self, clean_start: bool) -> Result<(), Error> {
        let qs = quantiles_sql_args();
        let on_cluster_name = self.clickhouse.get_on_cluster_name();

        // 1. Add columns to ModelInference
        self.clickhouse
            .run_query_synchronous_no_params(format!(
                "ALTER TABLE ModelInference{on_cluster_name} ADD COLUMN IF NOT EXISTS provider_cache_read_input_tokens Nullable(UInt32)"
            ))
            .await?;
        self.clickhouse
            .run_query_synchronous_no_params(format!(
                "ALTER TABLE ModelInference{on_cluster_name} ADD COLUMN IF NOT EXISTS provider_cache_write_input_tokens Nullable(UInt32)"
            ))
            .await?;

        // 2. Add aggregate columns to ModelProviderStatistics
        self.clickhouse
            .run_query_synchronous_no_params(format!(
                "ALTER TABLE ModelProviderStatistics{on_cluster_name} ADD COLUMN IF NOT EXISTS total_provider_cache_read_input_tokens AggregateFunction(sum, Nullable(UInt32))"
            ))
            .await?;
        self.clickhouse
            .run_query_synchronous_no_params(format!(
                "ALTER TABLE ModelProviderStatistics{on_cluster_name} ADD COLUMN IF NOT EXISTS total_provider_cache_write_input_tokens AggregateFunction(sum, Nullable(UInt32))"
            ))
            .await?;

        // 3. Record timestamp T (now + 15s offset)
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

        // 4. Drop the existing MV
        self.clickhouse
            .run_query_synchronous_no_params(format!(
                "DROP TABLE IF EXISTS ModelProviderStatisticsView{on_cluster_name} SYNC"
            ))
            .await?;

        // 5. Recreate the MV with the new cache token columns
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
                sumState(cost) as total_cost,
                sumState(provider_cache_read_input_tokens) as total_provider_cache_read_input_tokens,
                sumState(provider_cache_write_input_tokens) as total_provider_cache_write_input_tokens
            FROM ModelInference
            {view_where_clause}
            GROUP BY model_name, model_provider_name, minute
            "
        );
        self.clickhouse
            .run_query_synchronous_no_params(query)
            .await?;

        // 6. Backfill if needed
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

            // Only backfill the new cache token columns. The other columns were already
            // aggregated by previous migrations. Inserting only cache token columns lets
            // AggregatingMergeTree merge the new partial row with the existing one.
            tracing::info!("Running backfill of `ModelProviderStatistics` for cache token columns");
            let query = format!(
                r"
                INSERT INTO ModelProviderStatistics
                    (model_name, model_provider_name, minute, total_provider_cache_read_input_tokens, total_provider_cache_write_input_tokens)
                SELECT
                    model_name,
                    model_provider_name,
                    toStartOfMinute(timestamp) as minute,
                    sumState(provider_cache_read_input_tokens) as total_provider_cache_read_input_tokens,
                    sumState(provider_cache_write_input_tokens) as total_provider_cache_write_input_tokens
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
            ALTER TABLE ModelProviderStatistics{on_cluster_name} DROP COLUMN total_provider_cache_read_input_tokens;
            ALTER TABLE ModelProviderStatistics{on_cluster_name} DROP COLUMN total_provider_cache_write_input_tokens;
            ALTER TABLE ModelInference{on_cluster_name} DROP COLUMN provider_cache_read_input_tokens;
            ALTER TABLE ModelInference{on_cluster_name} DROP COLUMN provider_cache_write_input_tokens;
            CREATE MATERIALIZED VIEW IF NOT EXISTS ModelProviderStatisticsView{on_cluster_name} TO ModelProviderStatistics AS SELECT model_name, model_provider_name, toStartOfMinute(timestamp) as minute, quantilesTDigestState({qs})(response_time_ms) as response_time_ms_quantiles, quantilesTDigestState({qs})(ttft_ms) as ttft_ms_quantiles, sumState(input_tokens) as total_input_tokens, sumState(output_tokens) as total_output_tokens, countState() as count, sumState(cost) as total_cost FROM ModelInference GROUP BY model_name, model_provider_name, minute;"
        )
    }

    async fn has_succeeded(&self) -> Result<bool, Error> {
        Ok(check_column_exists(
            self.clickhouse,
            "ModelInference",
            "provider_cache_read_input_tokens",
            MIGRATION_ID,
        )
        .await?
            && check_column_exists(
                self.clickhouse,
                "ModelInference",
                "provider_cache_write_input_tokens",
                MIGRATION_ID,
            )
            .await?
            && check_column_exists(
                self.clickhouse,
                "ModelProviderStatistics",
                "total_provider_cache_read_input_tokens",
                MIGRATION_ID,
            )
            .await?
            && check_column_exists(
                self.clickhouse,
                "ModelProviderStatistics",
                "total_provider_cache_write_input_tokens",
                MIGRATION_ID,
            )
            .await?)
    }
}
