use super::check_table_exists;
use crate::db::clickhouse::migration_manager::migration_trait::Migration;
use crate::db::clickhouse::{ClickHouseConnectionInfo, GetMaybeReplicatedTableEngineNameArgs};
use crate::error::{Error, ErrorDetails};
use crate::utils::uuid::get_workflow_evaluation_cutoff_uuid;
use async_trait::async_trait;
use std::time::Duration;

/*
 * This migration sets up the EpisodeById table.
 * This should allow consumers to easily query episode data by id.
 * Note: this migration contains a silent syntax error in the materialized view:
 * `groupArrayState()(id)` in the EpisodeByIdChatView and EpisodeByIdJsonView creation queries should be `groupArrayState(id)`.
 * We address this in Migration 0041.
 */

pub struct Migration0038<'a> {
    pub clickhouse: &'a ClickHouseConnectionInfo,
}

const MIGRATION_ID: &str = "0038";

#[async_trait]
impl Migration for Migration0038<'_> {
    async fn can_apply(&self) -> Result<(), Error> {
        if !check_table_exists(self.clickhouse, "ChatInference", MIGRATION_ID).await? {
            return Err(Error::new(ErrorDetails::ClickHouseMigration {
                id: MIGRATION_ID.to_string(),
                message: "ChatInference table does not exist".to_string(),
            }));
        }
        if !check_table_exists(self.clickhouse, "JsonInference", MIGRATION_ID).await? {
            return Err(Error::new(ErrorDetails::ClickHouseMigration {
                id: MIGRATION_ID.to_string(),
                message: "JsonInference table does not exist".to_string(),
            }));
        }
        Ok(())
    }

    async fn should_apply(&self) -> Result<bool, Error> {
        if !check_table_exists(self.clickhouse, "EpisodeById", MIGRATION_ID).await? {
            return Ok(true);
        }
        if !check_table_exists(self.clickhouse, "EpisodeByIdChatView", MIGRATION_ID).await? {
            return Ok(true);
        }
        if !check_table_exists(self.clickhouse, "EpisodeByIdJsonView", MIGRATION_ID).await? {
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
                table_name: "EpisodeById",
                table_engine_name: "AggregatingMergeTree",
                engine_args: &[],
            },
        );

        self.clickhouse
            .run_query_synchronous_no_params(format!(
                r"CREATE TABLE IF NOT EXISTS EpisodeById{on_cluster_name} (
                        episode_id_uint UInt128,
                        count SimpleAggregateFunction(sum, UInt64),
                        inference_ids AggregateFunction(groupArray, UUID),
                        min_inference_id_uint SimpleAggregateFunction(min, UInt128),
                        max_inference_id_uint SimpleAggregateFunction(max, UInt128)
                    )
                    ENGINE = {table_engine_name}
                    ORDER BY episode_id_uint
                    "
            ))
            .await?;

        // If not a clean start, restrict MV ingestion to rows >= view timestamp.
        let view_condition = if clean_start {
            "1=1".to_string()
        } else {
            format!("UUIDv7ToDateTime(id) >= fromUnixTimestamp64Nano({view_timestamp_nanos})")
        };
        let cutoff_uuid = get_workflow_evaluation_cutoff_uuid();

        // Build MV for ChatInference table
        let query = format!(
            r"
            CREATE MATERIALIZED VIEW IF NOT EXISTS EpisodeByIdChatView{on_cluster_name}
            TO EpisodeById
            AS
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
