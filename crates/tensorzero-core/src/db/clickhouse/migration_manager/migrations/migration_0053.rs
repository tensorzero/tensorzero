use super::check_column_exists;
use super::check_table_exists;
use super::migration_0037::quantiles_sql_args;
use crate::db::clickhouse::migration_manager::migration_trait::Migration;
use crate::db::clickhouse::{ClickHouseConnectionInfo, GetMaybeReplicatedTableEngineNameArgs};
use crate::error::{ErrorDetails, delayed_error::DelayedError};
use async_trait::async_trait;

/// Denormalizes `function_name` and `variant_name` onto the `ModelInference` table,
/// then creates the `VariantStatistics` AggregatingMergeTree table and three
/// materialized views that feed it:
/// - `VariantStatisticsModelView`: triggers on `ModelInference`, reads denormalized
///   `function_name` and `variant_name` directly, aggregates token/cost metrics.
/// - `VariantStatisticsChatView`: triggers on `ChatInference`, aggregates latency and count.
/// - `VariantStatisticsJsonView`: triggers on `JsonInference`, aggregates latency and count.
///
/// The `AggregatingMergeTree` engine merges partial rows from all three MVs by the
/// (function_name, variant_name, minute) ORDER BY key.
///
/// See #6980.
pub struct Migration0053<'a> {
    pub clickhouse: &'a ClickHouseConnectionInfo,
}

const MIGRATION_ID: &str = "0053";

#[async_trait]
impl Migration for Migration0053<'_> {
    async fn can_apply(&self) -> Result<(), DelayedError> {
        for table in ["ModelInference", "ChatInference", "JsonInference"] {
            if !check_table_exists(self.clickhouse, table, MIGRATION_ID).await? {
                return Err(DelayedError::new(ErrorDetails::ClickHouseMigration {
                    id: MIGRATION_ID.to_string(),
                    message: format!("`{table}` table does not exist"),
                }));
            }
        }
        Ok(())
    }

    async fn should_apply(&self) -> Result<bool, DelayedError> {
        // Check denormalization columns on ModelInference
        let has_fn = check_column_exists(
            self.clickhouse,
            "ModelInference",
            "function_name",
            MIGRATION_ID,
        )
        .await?;
        let has_vn = check_column_exists(
            self.clickhouse,
            "ModelInference",
            "variant_name",
            MIGRATION_ID,
        )
        .await?;
        if !(has_fn && has_vn) {
            return Ok(true);
        }

        // Check VariantStatistics tables
        for table in [
            "VariantStatistics",
            "VariantStatisticsModelView",
            "VariantStatisticsChatView",
            "VariantStatisticsJsonView",
        ] {
            if !check_table_exists(self.clickhouse, table, MIGRATION_ID).await? {
                return Ok(true);
            }
        }
        Ok(false)
    }

    async fn apply(&self, clean_start: bool) -> Result<(), DelayedError> {
        let qs = quantiles_sql_args();
        let on_cluster_name = self.clickhouse.get_on_cluster_name();

        // ── Phase 1: Denormalize function_name/variant_name onto ModelInference ──

        self.clickhouse
            .run_query_synchronous_no_params_delayed_err(format!(
                "ALTER TABLE ModelInference{on_cluster_name} ADD COLUMN IF NOT EXISTS function_name LowCardinality(String) DEFAULT ''"
            ))
            .await?;
        self.clickhouse
            .run_query_synchronous_no_params_delayed_err(format!(
                "ALTER TABLE ModelInference{on_cluster_name} ADD COLUMN IF NOT EXISTS variant_name LowCardinality(String) DEFAULT ''"
            ))
            .await?;

        // Note: we do NOT backfill existing ModelInference rows because ClickHouse
        // does not support correlated subqueries in ALTER TABLE UPDATE. Instead:
        // - New rows: application code populates function_name/variant_name on insert.
        // - VariantStatistics backfill (Phase 3): JOINs ChatInference/JsonInference
        //   for historical data, so denormalized columns aren't needed for old rows.
        // - The MV's `function_name != ''` filter correctly excludes unbackfilled rows.

        // ── Phase 2: Create VariantStatistics table and materialized views ──

        let table_engine_name = self.clickhouse.get_maybe_replicated_table_engine_name(
            GetMaybeReplicatedTableEngineNameArgs {
                table_name: "VariantStatistics",
                table_engine_name: "AggregatingMergeTree",
                engine_args: &[],
            },
        );

        self.clickhouse
            .run_query_synchronous_no_params_delayed_err(format!(
                r"CREATE TABLE IF NOT EXISTS VariantStatistics{on_cluster_name} (
                    function_name LowCardinality(String),
                    variant_name LowCardinality(String),
                    minute DateTime,
                    processing_time_ms_quantiles AggregateFunction(quantilesTDigest({qs}), Nullable(UInt32)),
                    ttft_ms_quantiles AggregateFunction(quantilesTDigest({qs}), Nullable(UInt32)),
                    total_input_tokens AggregateFunction(sum, Nullable(UInt32)),
                    total_output_tokens AggregateFunction(sum, Nullable(UInt32)),
                    count AggregateFunction(count, UInt32),
                    total_cost AggregateFunction(sum, Nullable(Decimal(18, 9))),
                    total_provider_cache_read_input_tokens AggregateFunction(sum, Nullable(UInt32)),
                    total_provider_cache_write_input_tokens AggregateFunction(sum, Nullable(UInt32)),
                    count_with_cost AggregateFunction(count, Nullable(Decimal(18, 9)))
                )
                ENGINE = {table_engine_name}
                ORDER BY (function_name, variant_name, minute)"
            ))
            .await?;

        // Record timestamp T (now + small offset) for MV WHERE clauses.
        // On non-clean-start, MVs only capture inserts after this timestamp
        // to avoid double-counting with any concurrent writes.
        let view_timestamp_nanos = (std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_err(|e| {
                DelayedError::new(ErrorDetails::ClickHouseMigration {
                    id: MIGRATION_ID.to_string(),
                    message: e.to_string(),
                })
            })?
            + std::time::Duration::from_secs(5))
        .as_nanos();

        let model_view_where = if clean_start {
            "mi.function_name != ''".to_string()
        } else {
            format!(
                "mi.function_name != '' AND UUIDv7ToDateTime(mi.id) >= fromUnixTimestamp64Nano({view_timestamp_nanos})"
            )
        };

        let inference_view_where = if clean_start {
            String::new()
        } else {
            format!("WHERE UUIDv7ToDateTime(id) >= fromUnixTimestamp64Nano({view_timestamp_nanos})")
        };

        // MV 1: VariantStatisticsModelView (triggers on ModelInference inserts)
        // Reads denormalized function_name/variant_name directly from ModelInference.
        // The `function_name != ''` filter excludes rows that predate the denormalization
        // and haven't been backfilled yet.
        self.clickhouse
            .run_query_synchronous_no_params_delayed_err(format!(
                r"
                CREATE MATERIALIZED VIEW IF NOT EXISTS VariantStatisticsModelView{on_cluster_name}
                TO VariantStatistics
                AS
                SELECT
                    mi.function_name AS function_name,
                    mi.variant_name AS variant_name,
                    toStartOfMinute(mi.timestamp) AS minute,
                    sumState(mi.input_tokens) AS total_input_tokens,
                    sumState(mi.output_tokens) AS total_output_tokens,
                    sumState(mi.cost) AS total_cost,
                    sumState(mi.provider_cache_read_input_tokens) AS total_provider_cache_read_input_tokens,
                    sumState(mi.provider_cache_write_input_tokens) AS total_provider_cache_write_input_tokens,
                    countState(mi.cost) AS count_with_cost
                FROM ModelInference mi
                WHERE {model_view_where}
                GROUP BY mi.function_name, mi.variant_name, minute
                "
            ))
            .await?;

        // MV 2: VariantStatisticsChatView (triggers on ChatInference inserts)
        self.clickhouse
            .run_query_synchronous_no_params_delayed_err(format!(
                r"
                CREATE MATERIALIZED VIEW IF NOT EXISTS VariantStatisticsChatView{on_cluster_name}
                TO VariantStatistics
                AS
                SELECT
                    function_name,
                    variant_name,
                    toStartOfMinute(timestamp) AS minute,
                    quantilesTDigestState({qs})(processing_time_ms) AS processing_time_ms_quantiles,
                    quantilesTDigestState({qs})(ttft_ms) AS ttft_ms_quantiles,
                    countState() AS count
                FROM ChatInference
                {inference_view_where}
                GROUP BY function_name, variant_name, minute
                "
            ))
            .await?;

        // MV 3: VariantStatisticsJsonView (triggers on JsonInference inserts)
        self.clickhouse
            .run_query_synchronous_no_params_delayed_err(format!(
                r"
                CREATE MATERIALIZED VIEW IF NOT EXISTS VariantStatisticsJsonView{on_cluster_name}
                TO VariantStatistics
                AS
                SELECT
                    function_name,
                    variant_name,
                    toStartOfMinute(timestamp) AS minute,
                    quantilesTDigestState({qs})(processing_time_ms) AS processing_time_ms_quantiles,
                    quantilesTDigestState({qs})(ttft_ms) AS ttft_ms_quantiles,
                    countState() AS count
                FROM JsonInference
                {inference_view_where}
                GROUP BY function_name, variant_name, minute
                "
            ))
            .await?;

        // No backfill of historical data into VariantStatistics. The MVs will
        // capture all new inserts going forward. Historical data can be backfilled
        // manually if needed via INSERT ... SELECT queries.

        Ok(())
    }

    fn rollback_instructions(&self) -> String {
        let on_cluster_name = self.clickhouse.get_on_cluster_name();
        format!(
            r"
            DROP TABLE IF EXISTS VariantStatisticsModelView{on_cluster_name} SYNC;
            DROP TABLE IF EXISTS VariantStatisticsChatView{on_cluster_name} SYNC;
            DROP TABLE IF EXISTS VariantStatisticsJsonView{on_cluster_name} SYNC;
            DROP TABLE IF EXISTS VariantStatistics{on_cluster_name} SYNC;
            ALTER TABLE ModelInference{on_cluster_name} DROP COLUMN IF EXISTS function_name;
            ALTER TABLE ModelInference{on_cluster_name} DROP COLUMN IF EXISTS variant_name;"
        )
    }

    async fn has_succeeded(&self) -> Result<bool, DelayedError> {
        let should_apply = self.should_apply().await?;
        Ok(!should_apply)
    }
}
