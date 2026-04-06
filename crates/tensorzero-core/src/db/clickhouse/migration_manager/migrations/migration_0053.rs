use super::ViewOffsetDeadline;
use super::check_column_exists;
use super::check_table_exists;
use super::migration_0037::quantiles_sql_args;
use crate::db::clickhouse::migration_manager::migration_trait::Migration;
use crate::db::clickhouse::{ClickHouseConnectionInfo, GetMaybeReplicatedTableEngineNameArgs};
use crate::error::{Error, ErrorDetails};
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
    async fn can_apply(&self) -> Result<(), Error> {
        for table in ["ModelInference", "ChatInference", "JsonInference"] {
            if !check_table_exists(self.clickhouse, table, MIGRATION_ID).await? {
                return Err(Error::new(ErrorDetails::ClickHouseMigration {
                    id: MIGRATION_ID.to_string(),
                    message: format!("`{table}` table does not exist"),
                }));
            }
        }
        Ok(())
    }

    async fn should_apply(&self) -> Result<bool, Error> {
        // Check denormalization columns
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

    async fn apply(&self, clean_start: bool) -> Result<(), Error> {
        let qs = quantiles_sql_args();
        let on_cluster_name = self.clickhouse.get_on_cluster_name();

        // ── Phase 1: Denormalize function_name/variant_name onto ModelInference ──

        self.clickhouse
            .run_query_synchronous_no_params(format!(
                "ALTER TABLE ModelInference{on_cluster_name} ADD COLUMN IF NOT EXISTS function_name LowCardinality(String) DEFAULT ''"
            ))
            .await?;
        self.clickhouse
            .run_query_synchronous_no_params(format!(
                "ALTER TABLE ModelInference{on_cluster_name} ADD COLUMN IF NOT EXISTS variant_name LowCardinality(String) DEFAULT ''"
            ))
            .await?;

        if !clean_start {
            for inference_table in ["ChatInference", "JsonInference"] {
                tracing::info!(
                    "Backfilling `ModelInference.function_name`/`variant_name` from `{inference_table}`"
                );
                self.clickhouse
                    .run_query_synchronous_no_params(format!(
                        r"ALTER TABLE ModelInference UPDATE
                        function_name = (SELECT {inference_table}.function_name FROM {inference_table} WHERE {inference_table}.id = ModelInference.inference_id LIMIT 1),
                        variant_name = (SELECT {inference_table}.variant_name FROM {inference_table} WHERE {inference_table}.id = ModelInference.inference_id LIMIT 1)
                    WHERE function_name = '' AND inference_id IN (SELECT id FROM {inference_table})"
                    ))
                    .await?;
            }
        }

        // ── Phase 2: Create VariantStatistics table and materialized views ──

        let table_engine_name = self.clickhouse.get_maybe_replicated_table_engine_name(
            GetMaybeReplicatedTableEngineNameArgs {
                table_name: "VariantStatistics",
                table_engine_name: "AggregatingMergeTree",
                engine_args: &[],
            },
        );

        self.clickhouse
            .run_query_synchronous_no_params(format!(
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

        // Record timestamp T (now + offset) for MV WHERE clauses
        let deadline = ViewOffsetDeadline::new();
        let view_timestamp_nanos = (std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_err(|e| {
                Error::new(ErrorDetails::ClickHouseMigration {
                    id: MIGRATION_ID.to_string(),
                    message: e.to_string(),
                })
            })?
            + ViewOffsetDeadline::offset())
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
            .run_query_synchronous_no_params(format!(
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
            .run_query_synchronous_no_params(format!(
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
            .run_query_synchronous_no_params(format!(
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

        // ── Phase 3: Backfill historical data ──

        if !clean_start {
            deadline.wait().await;

            let view_timestamp_nanos_string = view_timestamp_nanos.to_string();

            // Verify the model view was created with our timestamp
            let create_table = self
                .clickhouse
                .run_query_synchronous_no_params(
                    "SHOW CREATE TABLE VariantStatisticsModelView".to_string(),
                )
                .await?
                .response;

            if !create_table.contains(&view_timestamp_nanos_string) {
                tracing::warn!(
                    "Materialized view `VariantStatisticsModelView` was not written because it was recently created. This is likely due to a concurrent migration. Unless the other migration failed, no action is required."
                );
                return Ok(());
            }

            // Backfill from ModelInference (token/cost metrics)
            // Reads denormalized function_name/variant_name directly — no JOIN needed.
            tracing::info!("Running backfill of `VariantStatistics` from `ModelInference`");
            self.clickhouse
                .run_query_synchronous_no_params(format!(
                    r"
                    INSERT INTO VariantStatistics
                        (function_name, variant_name, minute,
                         total_input_tokens, total_output_tokens, total_cost,
                         total_provider_cache_read_input_tokens, total_provider_cache_write_input_tokens,
                         count_with_cost)
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
                    WHERE mi.function_name != ''
                        AND UUIDv7ToDateTime(mi.id) < fromUnixTimestamp64Nano({view_timestamp_nanos})
                    GROUP BY mi.function_name, mi.variant_name, minute
                    "
                ))
                .await?;

            // Backfill from ChatInference (latency/count metrics)
            tracing::info!("Running backfill of `VariantStatistics` from `ChatInference`");
            self.clickhouse
                .run_query_synchronous_no_params(format!(
                    r"
                    INSERT INTO VariantStatistics
                        (function_name, variant_name, minute,
                         processing_time_ms_quantiles, ttft_ms_quantiles, count)
                    SELECT
                        function_name,
                        variant_name,
                        toStartOfMinute(timestamp) AS minute,
                        quantilesTDigestState({qs})(processing_time_ms) AS processing_time_ms_quantiles,
                        quantilesTDigestState({qs})(ttft_ms) AS ttft_ms_quantiles,
                        countState() AS count
                    FROM ChatInference
                    WHERE UUIDv7ToDateTime(id) < fromUnixTimestamp64Nano({view_timestamp_nanos})
                    GROUP BY function_name, variant_name, minute
                    "
                ))
                .await?;

            // Backfill from JsonInference (latency/count metrics)
            tracing::info!("Running backfill of `VariantStatistics` from `JsonInference`");
            self.clickhouse
                .run_query_synchronous_no_params(format!(
                    r"
                    INSERT INTO VariantStatistics
                        (function_name, variant_name, minute,
                         processing_time_ms_quantiles, ttft_ms_quantiles, count)
                    SELECT
                        function_name,
                        variant_name,
                        toStartOfMinute(timestamp) AS minute,
                        quantilesTDigestState({qs})(processing_time_ms) AS processing_time_ms_quantiles,
                        quantilesTDigestState({qs})(ttft_ms) AS ttft_ms_quantiles,
                        countState() AS count
                    FROM JsonInference
                    WHERE UUIDv7ToDateTime(id) < fromUnixTimestamp64Nano({view_timestamp_nanos})
                    GROUP BY function_name, variant_name, minute
                    "
                ))
                .await?;
        }

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

    async fn has_succeeded(&self) -> Result<bool, Error> {
        let should_apply = self.should_apply().await?;
        Ok(!should_apply)
    }
}
