use super::check_table_exists;
use crate::db::clickhouse::migration_manager::migration_trait::Migration;
use crate::db::clickhouse::{ClickHouseConnectionInfo, GetMaybeReplicatedTableEngineNameArgs};
use crate::error::{Error, ErrorDetails};
use crate::uuid_util::get_dynamic_evaluation_cutoff_uuid;
use async_trait::async_trait;
use std::time::Duration;

/*
 * This migration sets up the EpisodeById table.
 * This should allow consumers to easily query episode data by id.
 */

pub struct Migration0039<'a> {
    pub clickhouse: &'a ClickHouseConnectionInfo,
}

const MIGRATION_ID: &str = "0039";

#[async_trait]
impl Migration for Migration0039<'_> {
    async fn can_apply(&self) -> Result<(), Error> {
        if !check_table_exists(self.clickhouse, "InferenceById", MIGRATION_ID).await? {
            return Err(Error::new(ErrorDetails::ClickHouseMigration {
                id: MIGRATION_ID.to_string(),
                message: "InferenceById table does not exist".to_string(),
            }));
        }
        if !check_table_exists(self.clickhouse, "InferenceByEpisodeId", MIGRATION_ID).await? {
            return Err(Error::new(ErrorDetails::ClickHouseMigration {
                id: MIGRATION_ID.to_string(),
                message: "InferenceByEpisodeId table does not exist".to_string(),
            }));
        }
        if !check_table_exists(self.clickhouse, "FloatMetricFeedback", MIGRATION_ID).await? {
            return Err(Error::new(ErrorDetails::ClickHouseMigration {
                id: MIGRATION_ID.to_string(),
                message: "FloatMetricFeedback table does not exist".to_string(),
            }));
        }
        if !check_table_exists(self.clickhouse, "BooleanMetricFeedback", MIGRATION_ID).await? {
            return Err(Error::new(ErrorDetails::ClickHouseMigration {
                id: MIGRATION_ID.to_string(),
                message: "BooleanMetricFeedback table does not exist".to_string(),
            }));
        }
        Ok(())
    }

    async fn should_apply(&self) -> Result<bool, Error> {
        if !check_table_exists(self.clickhouse, "FeedbackByVariantStatistics", MIGRATION_ID).await?
        {
            return Ok(true);
        }
        if !check_table_exists(self.clickhouse, "FloatFeedbackByVariant", MIGRATION_ID).await? {
            return Ok(true);
        }
        if !check_table_exists(self.clickhouse, "BooleanFeedbackByVariant", MIGRATION_ID).await? {
            return Ok(true);
        }
        if !check_table_exists(self.clickhouse, "FloatFeedbackByVariantView", MIGRATION_ID).await? {
            return Ok(true);
        }
        if !check_table_exists(
            self.clickhouse,
            "BooleanFeedbackByVariantView",
            MIGRATION_ID,
        )
        .await?
        {
            return Ok(true);
        }
        if !check_table_exists(
            self.clickhouse,
            "FloatFeedbackByVariantStatisticsView",
            MIGRATION_ID,
        )
        .await?
        {
            return Ok(true);
        }
        if !check_table_exists(
            self.clickhouse,
            "BooleanFeedbackByVariantStatisticsView",
            MIGRATION_ID,
        )
        .await?
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
                table_name: "FloatFeedbackByVariant",
                table_engine_name: "MergeTree",
                engine_args: &[],
            },
        );

        self.clickhouse
            .run_query_synchronous_no_params(format!(
                r"CREATE TABLE IF NOT EXISTS FloatFeedbackByVariant{on_cluster_name} (
                        function_name LowCardinality(String),
                        variant_name LowCardinality(String),
                        metric_name LowCardinality(String),
                        id_uint UInt128,
                        target_id_uint UInt128,
                        value Float32
                    )
                    ENGINE = {table_engine_name}
                    ORDER BY (function_name, variant_name, metric_name, id_uint);
                    "
            ))
            .await?;

        let table_engine_name = self.clickhouse.get_maybe_replicated_table_engine_name(
            GetMaybeReplicatedTableEngineNameArgs {
                table_name: "BooleanFeedbackByVariant",
                table_engine_name: "MergeTree",
                engine_args: &[],
            },
        );

        self.clickhouse
            .run_query_synchronous_no_params(format!(
                r"CREATE TABLE IF NOT EXISTS BooleanFeedbackByVariant{on_cluster_name} (
                        function_name LowCardinality(String),
                        variant_name LowCardinality(String),
                        metric_name LowCardinality(String),
                        id_uint UInt128,
                        target_id_uint UInt128,
                        value Bool,
                    )
                    ENGINE = {table_engine_name}
                    ORDER BY (function_name, variant_name, metric_name, id_uint);
                    "
            ))
            .await?;

        let table_engine_name = self.clickhouse.get_maybe_replicated_table_engine_name(
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
                    count SimpleAggregateFunction(count, UInt32)
                )
                Engine = {table_engine_name}
                ORDER BY (function_name, variant_name, metric_name, minute);
                    "
            ))
            .await?;

        // If not a clean start, restrict MV ingestion to rows >= view timestamp.
        let view_condition = if clean_start {
            "1=1".to_string()
        } else {
            format!("UUIDv7ToDateTime(id) >= fromUnixTimestamp64Nano({view_timestamp_nanos})")
        };
        let cutoff_uuid = get_dynamic_evaluation_cutoff_uuid();

        // Build MV for FloatFeedbackByVariant table
        let query = format!(
            r"
            CREATE MATERIALIZED VIEW IF NOT EXISTS FloatFeedbackByVariantView{on_cluster_name}
            TO FloatFeedbackByVariant
            AS
            WITH
                float_feedback AS (
                    SELECT
                        toUInt128(id) as id_uint,
                        metric_name,
                        target_id,
                        toUInt128(target_id) as target_id_uint,
                        value
                    FROM FloatMetricFeedback
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
                        any(variant_name) as variant_name
                    FROM InferenceByEpisodeID
                    WHERE episode_id_uint IN (SELECT target_id_uint FROM float_feedback)
                    GROUP BY (episode_id_uint, function_name)
                    HAVING uniqExact(variant_name) > 1
                )

                --- end

                )
            SELECT
                toUInt128(episode_id) as episode_id_uint,
                1 as count,
                groupArrayState()(id) as inference_ids,
                toUInt128(min(id)) as min_inference_id_uint,
                toUInt128(max(id)) as max_inference_id_uint
            FROM ChatInference
            WHERE {view_condition} AND toUInt128(episode_id) < toUInt128(toUUID('{cutoff_uuid}'))
            GROUP BY toUInt128(episode_id)
            "
        );
        self.clickhouse
            .run_query_synchronous_no_params(query)
            .await?;
        // Build MV for JsonInference table
        let query = format!(
            r"
            CREATE MATERIALIZED VIEW IF NOT EXISTS EpisodeByIdJsonView{on_cluster_name}
            TO EpisodeById
            AS
            SELECT
                toUInt128(episode_id) as episode_id_uint,
                1 as count,
                groupArrayState()(id) as inference_ids,
                toUInt128(min(id)) as min_inference_id_uint,
                toUInt128(max(id)) as max_inference_id_uint
            FROM JsonInference
            WHERE {view_condition} AND toUInt128(episode_id) < toUInt128(toUUID('{cutoff_uuid}'))
            GROUP BY toUInt128(episode_id)
            "
        );
        self.clickhouse
            .run_query_synchronous_no_params(query)
            .await?;

        // Backfill if needed
        if !clean_start {
            tokio::time::sleep(view_offset).await;

            let create_chat_table = self
                .clickhouse
                .run_query_synchronous_no_params(
                    "SHOW CREATE TABLE EpisodeByIdChatView".to_string(),
                )
                .await?
                .response;

            let view_timestamp_nanos_string = view_timestamp_nanos.to_string();
            if create_chat_table.contains(&view_timestamp_nanos_string) {
                // Run backfill for EpisodeByIdChatView if the chat timestamps match
                let query = format!(
                    r"
                    INSERT INTO EpisodeById
                    SELECT
                        toUInt128(episode_id) as episode_id_uint,
                        1 as count,
                        groupArrayState()(id) as inference_ids,
                        toUInt128(min(id)) as min_inference_id_uint,
                        toUInt128(max(id)) as max_inference_id_uint
                    FROM ChatInference
                    WHERE UUIDv7ToDateTime(id) < fromUnixTimestamp64Nano({view_timestamp_nanos})
                        AND toUInt128(episode_id) < toUInt128(toUUID('{cutoff_uuid}'))
                    GROUP BY toUInt128(episode_id)
                    "
                );
                self.clickhouse
                    .run_query_synchronous_no_params(query)
                    .await?;
            } else {
                tracing::warn!("Materialized view `EpisodeByIdChatView` was not written because it was recently created. This is likely due to a concurrent migration. Unless the other migration failed, no action is required.");
            }

            let create_json_table = self
                .clickhouse
                .run_query_synchronous_no_params(
                    "SHOW CREATE TABLE EpisodeByIdJsonView".to_string(),
                )
                .await?
                .response;

            if create_json_table.contains(&view_timestamp_nanos_string) {
                // Run backfill for EpisodeByIdJsonView if the json timestamps match
                let query = format!(
                    r"
                    INSERT INTO EpisodeById
                    SELECT
                        toUInt128(episode_id) as episode_id_uint,
                        1 as count,
                        groupArrayState()(id) as inference_ids,
                        toUInt128(min(id)) as min_inference_id_uint,
                        toUInt128(max(id)) as max_inference_id_uint
                    FROM JsonInference
                    WHERE UUIDv7ToDateTime(id) < fromUnixTimestamp64Nano({view_timestamp_nanos})
                          AND toUInt128(episode_id) < toUInt128(toUUID('{cutoff_uuid}'))
                    GROUP BY toUInt128(episode_id)
                    "
                );
                self.clickhouse
                    .run_query_synchronous_no_params(query)
                    .await?;
            } else {
                tracing::warn!("Materialized view `EpisodeByIdJsonView` was not written because it was recently created. This is likely due to a concurrent migration. Unless the other migration failed, no action is required.");
            }
        }

        Ok(())
    }

    fn rollback_instructions(&self) -> String {
        let on_cluster_name = self.clickhouse.get_on_cluster_name();
        format!(
            r"
        DROP TABLE IF EXISTS EpisodeByIdJsonView{on_cluster_name} SYNC;
        DROP TABLE IF EXISTS EpisodeByIdChatView{on_cluster_name} SYNC;
        DROP TABLE IF EXISTS EpisodeById{on_cluster_name} SYNC;"
        )
    }

    async fn has_succeeded(&self) -> Result<bool, Error> {
        let should_apply = self.should_apply().await?;
        Ok(!should_apply)
    }
}
