use super::check_table_exists;
use crate::db::clickhouse::migration_manager::migration_trait::Migration;
use crate::db::clickhouse::{ClickHouseConnectionInfo, GetMaybeReplicatedTableEngineNameArgs};
use crate::error::{Error, ErrorDetails};
use async_trait::async_trait;
use std::time::Duration;

/*
 * Introduces a table FeedbackByVariantStatistics which stores aggregated summary statistics
 * about feedback by function, variant, and time.
 * These pre-aggregated statistics will be useful for planned experimentation features.
 *
 * The views flow as follows:
 * When feedback is sent to either of FloatMetricFeedback or BooleanMetricFeedback,
 * the associated view writes FloatMetricFeedbackByVariant or BooleanMetricFeedbackByVariant if
 * there is an associated inference in InferenceById or InferenceByEpisodeId.
 * This view also checks if the feedback is episode-level and if so that a single variant was used for that
 * episode-function pair.
 * NOTE: This migration does not handle the case where an episode uses only one variant for a function and  episode, feedback is
 * sent, and then a new inference is made in the episode for the same function with a different variant.
 * NOTE: This migration does not support "overwriting" feedback so there could be multiple entries for the
 * same function / variant / metric triple.
 *
 * After this process a second materialized view is triggered to send the joined / denormalized data to
 * the FeedbackByVariantStatistics table, which aggregates the data by function, variant, and time and
 * computes running means and variances.
 */

pub struct Migration0039<'a> {
    pub clickhouse: &'a ClickHouseConnectionInfo,
}

const MIGRATION_ID: &str = "0039";

#[async_trait]
impl Migration for Migration0039<'_> {
    async fn can_apply(&self) -> Result<(), Error> {
        let tables_to_check = [
            "InferenceById",
            "InferenceByEpisodeId",
            "FloatMetricFeedback",
            "BooleanMetricFeedback",
        ];

        for table_name in tables_to_check {
            if !check_table_exists(self.clickhouse, table_name, MIGRATION_ID).await? {
                return Err(Error::new(ErrorDetails::ClickHouseMigration {
                    id: MIGRATION_ID.to_string(),
                    message: format!("{table_name} table does not exist"),
                }));
            }
        }
        Ok(())
    }

    async fn should_apply(&self) -> Result<bool, Error> {
        let tables_to_check = [
            "FeedbackByVariantStatistics",
            "FloatMetricFeedbackByVariant",
            "BooleanMetricFeedbackByVariant",
            "FloatMetricFeedbackByVariantView",
            "BooleanMetricFeedbackByVariantView",
            "FloatMetricFeedbackByVariantStatisticsView",
            "BooleanMetricFeedbackByVariantStatisticsView",
        ];

        for table_name in tables_to_check {
            if !check_table_exists(self.clickhouse, table_name, MIGRATION_ID).await? {
                // If any table is missing, we should apply the migration.
                return Ok(true);
            }
        }

        // If all tables exist, no need to apply.
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
        let float_table_engine_name = self.clickhouse.get_maybe_replicated_table_engine_name(
            GetMaybeReplicatedTableEngineNameArgs {
                table_name: "FloatMetricFeedbackByVariant",
                table_engine_name: "MergeTree",
                engine_args: &[],
            },
        );

        // We order by function_name, metric_name, variant_name, minute
        // because it is more likely that we'll need to aggregate all variants for a given function and metric
        // than to do metrics for a given variant and function

        self.clickhouse
            .run_query_synchronous_no_params(format!(
                r"CREATE TABLE IF NOT EXISTS FloatMetricFeedbackByVariant{on_cluster_name} (
                        function_name LowCardinality(String),
                        variant_name LowCardinality(String),
                        metric_name LowCardinality(String),
                        id_uint UInt128,
                        target_id_uint UInt128,
                        value Float32,
                        feedback_tags Map(String, String)
                    )
                    ENGINE = {float_table_engine_name}
                    ORDER BY (function_name, metric_name, variant_name, id_uint);
                    "
            ))
            .await?;

        let boolean_table_engine_name = self.clickhouse.get_maybe_replicated_table_engine_name(
            GetMaybeReplicatedTableEngineNameArgs {
                table_name: "BooleanMetricFeedbackByVariant",
                table_engine_name: "MergeTree",
                engine_args: &[],
            },
        );

        self.clickhouse
            .run_query_synchronous_no_params(format!(
                r"CREATE TABLE IF NOT EXISTS BooleanMetricFeedbackByVariant{on_cluster_name} (
                        function_name LowCardinality(String),
                        variant_name LowCardinality(String),
                        metric_name LowCardinality(String),
                        id_uint UInt128,
                        target_id_uint UInt128,
                        value Bool,
                        feedback_tags Map(String, String)
                    )
                    ENGINE = {boolean_table_engine_name}
                    ORDER BY (function_name, metric_name, variant_name, id_uint);
                    "
            ))
            .await?;

        let aggregating_table_engine_name = self.clickhouse.get_maybe_replicated_table_engine_name(
            GetMaybeReplicatedTableEngineNameArgs {
                table_name: "FeedbackByVariantStatistics",
                table_engine_name: "AggregatingMergeTree",
                engine_args: &[],
            },
        );

        self.clickhouse
            .run_query_synchronous_no_params(format!(
                r"CREATE TABLE IF NOT EXISTS FeedbackByVariantStatistics{on_cluster_name} (
                    function_name LowCardinality(String),
                    variant_name LowCardinality(String),
                    metric_name LowCardinality(String),
                    minute DateTime,
                    feedback_mean AggregateFunction(avg, Float32),
                    feedback_variance AggregateFunction(varSampStable, Float32),
                    count SimpleAggregateFunction(sum, UInt64)
                )
                Engine = {aggregating_table_engine_name}
                ORDER BY (function_name, metric_name, variant_name, minute);
                    "
            ))
            .await?;

        // If not a clean start, restrict MV ingestion to rows >= view timestamp.
        let view_timestamp_where_clause = if clean_start {
            "1=1".to_string()
        } else {
            format!("UUIDv7ToDateTime(id) >= fromUnixTimestamp64Nano({view_timestamp_nanos})")
        };
        let statistics_view_timestamp_where_clause = if clean_start {
            "1=1".to_string()
        } else {
            format!("UUIDv7ToDateTime(uint_to_uuid(id_uint)) >= fromUnixTimestamp64Nano({view_timestamp_nanos})")
        };

        // Build MV for FloatMetricFeedbackByVariant table
        let query = format!(
            r"
            CREATE MATERIALIZED VIEW IF NOT EXISTS FloatMetricFeedbackByVariantView{on_cluster_name}
            TO FloatMetricFeedbackByVariant
            AS
            WITH
                float_feedback AS (
                    SELECT
                        toUInt128(id) as id_uint,
                        metric_name,
                        target_id,
                        toUInt128(target_id) as target_id_uint,
                        value,
                        tags
                    FROM FloatMetricFeedback
                    WHERE {view_timestamp_where_clause}
                ),
                targets AS (
                    SELECT
                        uint_to_uuid(id_uint) as target_id,
                        function_name,
                        variant_name
                    FROM InferenceById
                    WHERE id_uint IN (SELECT target_id_uint FROM float_feedback)
                    UNION ALL
                    SELECT
                        uint_to_uuid(episode_id_uint) as target_id,
                        function_name,
                        unique_variants[1] as variant_name
                    FROM (
                    -- NOTE: this is conservative and only takes episodes with a single
                    -- unique variant for the function
                    -- We may consider generalizing this in the future
                    SELECT
                            episode_id_uint,
                            function_name,
                            groupUniqArray(variant_name) as unique_variants
                        FROM InferenceByEpisodeId
                        WHERE episode_id_uint IN (SELECT target_id_uint FROM float_feedback)
                        GROUP BY (episode_id_uint, function_name)
                    )
                    WHERE length(unique_variants) = 1
                )

            SELECT
                t.function_name as function_name,
                t.variant_name as variant_name,
                f.metric_name as metric_name,
                f.id_uint as id_uint,
                f.target_id_uint as target_id_uint,
                f.value as value,
                f.tags as feedback_tags
            FROM float_feedback f
            JOIN targets t ON f.target_id = t.target_id
            "
        );
        self.clickhouse
            .run_query_synchronous_no_params(query)
            .await?;

        // Build MV for BooleanMetricFeedbackByVariant table
        let query = format!(
            r"
            CREATE MATERIALIZED VIEW IF NOT EXISTS BooleanMetricFeedbackByVariantView{on_cluster_name}
            TO BooleanMetricFeedbackByVariant
            AS
            WITH
                boolean_feedback AS (
                    SELECT
                        toUInt128(id) as id_uint,
                        metric_name,
                        target_id,
                        toUInt128(target_id) as target_id_uint,
                        value,
                        tags
                    FROM BooleanMetricFeedback
                    WHERE {view_timestamp_where_clause}
                ),
                targets AS (
                    SELECT
                        uint_to_uuid(id_uint) as target_id,
                        function_name,
                        variant_name
                    FROM InferenceById
                    WHERE id_uint IN (SELECT target_id_uint FROM boolean_feedback)
                    UNION ALL
                    SELECT
                        uint_to_uuid(episode_id_uint) as target_id,
                        function_name,
                        unique_variants[1] as variant_name
                    FROM (
                    -- NOTE: this is conservative and only takes episodes with a single
                    -- unique variant for the function
                    -- We may consider generalizing this in the future
                        SELECT
                            episode_id_uint,
                            function_name,
                            groupUniqArray(variant_name) as unique_variants
                        FROM InferenceByEpisodeId
                        WHERE episode_id_uint IN (SELECT target_id_uint FROM boolean_feedback)
                        GROUP BY (episode_id_uint, function_name)
                    )
                    WHERE length(unique_variants) = 1
                )

            SELECT
                t.function_name as function_name,
                t.variant_name as variant_name,
                f.metric_name as metric_name,
                f.id_uint as id_uint,
                f.target_id_uint as target_id_uint,
                f.value as value,
                f.tags as feedback_tags
            FROM boolean_feedback f
            JOIN targets t ON f.target_id = t.target_id
            "
        );
        self.clickhouse
            .run_query_synchronous_no_params(query)
            .await?;

        // Build MV for FloatMetricFeedbackByVariantStatistics to FeedbackByVariantStatistics table
        let query = format!(
            r"
            CREATE MATERIALIZED VIEW IF NOT EXISTS FloatMetricFeedbackByVariantStatisticsView{on_cluster_name}
            TO FeedbackByVariantStatistics AS
            SELECT
                function_name,
                variant_name,
                metric_name,
                toStartOfMinute(UUIDv7ToDateTime(uint_to_uuid(id_uint))) as minute,
                avgState(value) as feedback_mean,
                varSampStableState(value) as feedback_variance,
                count() as count
            FROM FloatMetricFeedbackByVariant
            WHERE {statistics_view_timestamp_where_clause}
            GROUP BY function_name, metric_name, variant_name, minute;
            "
        );
        self.clickhouse
            .run_query_synchronous_no_params(query)
            .await?;

        // Build MV for BooleanMetricFeedbackByVariantStatistics to FeedbackByVariantStatistics table
        let query = format!(
            r"
            CREATE MATERIALIZED VIEW IF NOT EXISTS BooleanMetricFeedbackByVariantStatisticsView{on_cluster_name}
            TO FeedbackByVariantStatistics AS
            SELECT
                function_name,
                variant_name,
                metric_name,
                toStartOfMinute(UUIDv7ToDateTime(uint_to_uuid(id_uint))) as minute,
                avgState(toFloat32(value)) as feedback_mean,
                varSampStableState(toFloat32(value)) as feedback_variance,
                count() as count
            FROM BooleanMetricFeedbackByVariant
            WHERE {statistics_view_timestamp_where_clause}
            GROUP BY function_name, metric_name, variant_name, minute;
            "
        );
        self.clickhouse
            .run_query_synchronous_no_params(query)
            .await?;

        // Backfill if needed
        if !clean_start {
            tokio::time::sleep(view_offset).await;

            let create_float_feedback_by_variant_view = self
                .clickhouse
                .run_query_synchronous_no_params(
                    "SHOW CREATE TABLE FloatMetricFeedbackByVariantView".to_string(),
                )
                .await?
                .response;

            let view_timestamp_nanos_string = view_timestamp_nanos.to_string();
            if create_float_feedback_by_variant_view.contains(&view_timestamp_nanos_string) {
                // Run backfill for EpisodeByIdChatView if the chat timestamps match
                let query = format!(
                    r"
                    INSERT INTO FloatMetricFeedbackByVariant
                    WITH
                        float_feedback AS (
                            SELECT
                                toUInt128(id) as id_uint,
                                metric_name,
                                target_id,
                                toUInt128(target_id) as target_id_uint,
                                value,
                                tags
                            FROM FloatMetricFeedback
                            WHERE UUIDv7ToDateTime(id) < fromUnixTimestamp64Nano({view_timestamp_nanos})
                        ),
                        targets AS (
                            SELECT
                                uint_to_uuid(id_uint) as target_id,
                                function_name,
                                variant_name
                            FROM InferenceById
                            WHERE id_uint IN (SELECT target_id_uint FROM float_feedback)
                            UNION ALL
                            SELECT
                                uint_to_uuid(episode_id_uint) as target_id,
                                function_name,
                                unique_variants[1] as variant_name
                            FROM (
                                SELECT
                                    episode_id_uint,
                                    function_name,
                                    groupUniqArray(variant_name) as unique_variants
                                FROM InferenceByEpisodeId
                                WHERE episode_id_uint IN (SELECT target_id_uint FROM float_feedback)
                                GROUP BY (episode_id_uint, function_name)
                            )
                            WHERE length(unique_variants) = 1
                        )


                    SELECT
                        t.function_name as function_name,
                        t.variant_name as variant_name,
                        f.metric_name as metric_name,
                        f.id_uint as id_uint,
                        f.target_id_uint as target_id_uint,
                        f.value as value,
                        f.tags as feedback_tags
                    FROM float_feedback f
                    JOIN targets t ON f.target_id = t.target_id
                    "
                );
                self.clickhouse
                    .run_query_synchronous_no_params(query)
                    .await?;
            } else {
                tracing::warn!("Materialized view `FloatMetricFeedbackByVariantView` was not written because it was recently created. This is likely due to a concurrent migration. Unless the other migration failed, no action is required.");
            }

            let create_boolean_feedback_by_variant_view = self
                .clickhouse
                .run_query_synchronous_no_params(
                    "SHOW CREATE TABLE BooleanMetricFeedbackByVariantView".to_string(),
                )
                .await?
                .response;

            if create_boolean_feedback_by_variant_view.contains(&view_timestamp_nanos_string) {
                // Run backfill for EpisodeByIdJsonView if the json timestamps match
                let query = format!(
                    r"
                    INSERT INTO BooleanMetricFeedbackByVariant
                    WITH
                        boolean_feedback AS (
                            SELECT
                                toUInt128(id) as id_uint,
                                metric_name,
                                target_id,
                                toUInt128(target_id) as target_id_uint,
                                value,
                                tags
                            FROM BooleanMetricFeedback
                            WHERE UUIDv7ToDateTime(id) < fromUnixTimestamp64Nano({view_timestamp_nanos})
                        ),
                        targets AS (
                            SELECT
                                uint_to_uuid(id_uint) as target_id,
                                function_name,
                                variant_name
                            FROM InferenceById
                            WHERE id_uint IN (SELECT target_id_uint FROM boolean_feedback)
                            UNION ALL
                            SELECT
                                uint_to_uuid(episode_id_uint) as target_id,
                                function_name,
                                unique_variants[1] as variant_name
                            FROM (
                                SELECT
                                    episode_id_uint,
                                    function_name,
                                    groupUniqArray(variant_name) as unique_variants
                                FROM InferenceByEpisodeId
                                WHERE episode_id_uint IN (SELECT target_id_uint FROM boolean_feedback)
                                GROUP BY (episode_id_uint, function_name)
                            )
                            WHERE length(unique_variants) = 1
                        )

                    SELECT
                        t.function_name as function_name,
                        t.variant_name as variant_name,
                        f.metric_name as metric_name,
                        f.id_uint as id_uint,
                        f.target_id_uint as target_id_uint,
                        f.value as value,
                        f.tags as feedback_tags
                    FROM boolean_feedback f
                    JOIN targets t ON f.target_id = t.target_id
                    "
                );
                self.clickhouse
                    .run_query_synchronous_no_params(query)
                    .await?;
            } else {
                tracing::warn!("Materialized view `BooleanMetricFeedbackByVariantView` was not written because it was recently created. This is likely due to a concurrent migration. Unless the other migration failed, no action is required.");
            }
            let create_float_feedback_by_variant_statistics_view = self
                .clickhouse
                .run_query_synchronous_no_params(
                    "SHOW CREATE TABLE FloatMetricFeedbackByVariantStatisticsView".to_string(),
                )
                .await?
                .response;
            if create_float_feedback_by_variant_statistics_view
                .contains(&view_timestamp_nanos_string)
            {
                // Run backfill for FloatMetricFeedbackByVariantStatisticsView if the json timestamps match
                let query = format!(
                    r"
                    INSERT INTO FeedbackByVariantStatistics
                    SELECT
                        function_name,
                        variant_name,
                        metric_name,
                        toStartOfMinute(UUIDv7ToDateTime(uint_to_uuid(id_uint))) as minute,
                        avgState(value) as feedback_mean,
                        varSampStableState(value) as feedback_variance,
                        count() as count
                    FROM FloatMetricFeedbackByVariant
                    WHERE UUIDv7ToDateTime(uint_to_uuid(id_uint)) < fromUnixTimestamp64Nano({view_timestamp_nanos})
                    GROUP BY function_name, metric_name, variant_name, minute;
                    "
                );
                self.clickhouse
                    .run_query_synchronous_no_params(query)
                    .await?;
            } else {
                tracing::warn!("Materialized view `FloatMetricFeedbackByVariantStatisticsView` was not written because it was recently created. This is likely due to a concurrent migration. Unless the other migration failed, no action is required.");
            }
            let create_boolean_feedback_by_variant_statistics_view = self
                .clickhouse
                .run_query_synchronous_no_params(
                    "SHOW CREATE TABLE BooleanMetricFeedbackByVariantStatisticsView".to_string(),
                )
                .await?
                .response;
            if create_boolean_feedback_by_variant_statistics_view
                .contains(&view_timestamp_nanos_string)
            {
                // Run backfill for BooleanMetricFeedbackByVariantStatisticsView if the json timestamps match
                let query = format!(
                    r"
                    INSERT INTO FeedbackByVariantStatistics
                    SELECT
                        function_name,
                        variant_name,
                        metric_name,
                        toStartOfMinute(UUIDv7ToDateTime(uint_to_uuid(id_uint))) as minute,
                        avgState(toFloat32(value)) as feedback_mean,
                        varSampStableState(toFloat32(value)) as feedback_variance,
                        count() as count
                    FROM BooleanMetricFeedbackByVariant
                    WHERE UUIDv7ToDateTime(uint_to_uuid(id_uint)) < fromUnixTimestamp64Nano({view_timestamp_nanos})
                    GROUP BY function_name, metric_name, variant_name, minute;
                    "
                );
                self.clickhouse
                    .run_query_synchronous_no_params(query)
                    .await?;
            } else {
                tracing::warn!("Materialized view `BooleanMetricFeedbackByVariantStatisticsView` was not written because it was recently created. This is likely due to a concurrent migration. Unless the other migration failed, no action is required.");
            }
        }

        Ok(())
    }

    fn rollback_instructions(&self) -> String {
        let on_cluster_name = self.clickhouse.get_on_cluster_name();
        format!(
            r"
        DROP TABLE IF EXISTS BooleanMetricFeedbackByVariantStatisticsView{on_cluster_name} SYNC;
        DROP TABLE IF EXISTS FloatMetricFeedbackByVariantStatisticsView{on_cluster_name} SYNC;
        DROP TABLE IF EXISTS BooleanMetricFeedbackByVariantView{on_cluster_name} SYNC;
        DROP TABLE IF EXISTS FloatMetricFeedbackByVariantView{on_cluster_name} SYNC;
        DROP TABLE IF EXISTS FeedbackByVariantStatistics{on_cluster_name} SYNC;
        DROP TABLE IF EXISTS BooleanMetricFeedbackByVariant{on_cluster_name} SYNC;
        DROP TABLE IF EXISTS FloatMetricFeedbackByVariant{on_cluster_name} SYNC;
        "
        )
    }

    async fn has_succeeded(&self) -> Result<bool, Error> {
        let should_apply = self.should_apply().await?;
        Ok(!should_apply)
    }
}
