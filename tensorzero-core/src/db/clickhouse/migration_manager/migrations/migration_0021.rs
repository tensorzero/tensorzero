use std::time::Duration;

use async_trait::async_trait;

use super::{check_column_exists, check_table_exists, get_default_expression};
use crate::db::clickhouse::migration_manager::migration_trait::Migration;
use crate::db::clickhouse::{ClickHouseConnectionInfo, GetMaybeReplicatedTableEngineNameArgs};
use crate::error::{Error, ErrorDetails};

/// This migration adds a ReplacingMergeTree table TagInference.
/// This table stores data that allows us to efficiently query for tags by key and value
/// and to efficiently find inferences and feedbacks associated with them.
///
/// We also add materialized views TagChatInferenceView and TagJsonInferenceView,
/// These views will insert data into the TagInference table
/// when data is inserted into the original tables.
///
/// Additionally, this migration adds a `staled_at` column to the ChatInferenceDatapoint
/// and JsonInferenceDatapoint tables. This allows us to express that a datapoint is `stale`
/// and has been edited or deleted.
///
/// Additionally, we fixed the default for the updated_at column of ChatInferenceDatapoint
/// and JsonInferenceDatapoint to now64() from now() since now() is only second resolution.
pub struct Migration0021<'a> {
    pub clickhouse: &'a ClickHouseConnectionInfo,
}

const MIGRATION_ID: &str = "0021";

#[async_trait]
impl Migration for Migration0021<'_> {
    async fn can_apply(&self) -> Result<(), Error> {
        let chat_inference_table_exists =
            check_table_exists(self.clickhouse, "ChatInference", MIGRATION_ID).await?;
        let json_inference_table_exists =
            check_table_exists(self.clickhouse, "JsonInference", MIGRATION_ID).await?;
        let chat_inference_datapoint_table_exists =
            check_table_exists(self.clickhouse, "ChatInferenceDatapoint", MIGRATION_ID).await?;
        let json_inference_datapoint_table_exists =
            check_table_exists(self.clickhouse, "JsonInferenceDatapoint", MIGRATION_ID).await?;

        if !chat_inference_table_exists || !json_inference_table_exists {
            return Err(Error::new(ErrorDetails::ClickHouseMigration {
                id: MIGRATION_ID.to_string(),
                message: "One or more of the inference tables do not exist".to_string(),
            }));
        }
        if !chat_inference_datapoint_table_exists || !json_inference_datapoint_table_exists {
            return Err(Error::new(ErrorDetails::ClickHouseMigration {
                id: MIGRATION_ID.to_string(),
                message: "One or more of the inference datapoint tables do not exist".to_string(),
            }));
        }

        Ok(())
    }

    async fn should_apply(&self) -> Result<bool, Error> {
        let tag_inference_table_exists =
            check_table_exists(self.clickhouse, "TagInference", MIGRATION_ID).await?;
        let tag_chat_inference_view_exists =
            check_table_exists(self.clickhouse, "TagChatInferenceView", MIGRATION_ID).await?;
        let tag_json_inference_view_exists =
            check_table_exists(self.clickhouse, "TagJsonInferenceView", MIGRATION_ID).await?;
        let chat_inference_datapoint_staled_at_column_exists = check_column_exists(
            self.clickhouse,
            "ChatInferenceDatapoint",
            "staled_at",
            MIGRATION_ID,
        )
        .await?;
        let json_inference_datapoint_staled_at_column_exists = check_column_exists(
            self.clickhouse,
            "JsonInferenceDatapoint",
            "staled_at",
            MIGRATION_ID,
        )
        .await?;
        let chat_default_updated_at = get_default_expression(
            self.clickhouse,
            "ChatInferenceDatapoint",
            "updated_at",
            MIGRATION_ID,
        )
        .await?;
        let chat_default_updated_at_correct = chat_default_updated_at == "now64()";
        let json_default_updated_at = get_default_expression(
            self.clickhouse,
            "JsonInferenceDatapoint",
            "updated_at",
            MIGRATION_ID,
        )
        .await?;
        let json_default_updated_at_correct = json_default_updated_at == "now64()";
        Ok(!tag_inference_table_exists
            || !tag_chat_inference_view_exists
            || !tag_json_inference_view_exists
            || !chat_inference_datapoint_staled_at_column_exists
            || !json_inference_datapoint_staled_at_column_exists
            || !chat_default_updated_at_correct
            || !json_default_updated_at_correct)
    }

    async fn apply(&self, clean_start: bool) -> Result<(), Error> {
        // Only gets used when we are not doing a clean start
        let view_offset = Duration::from_secs(15);
        let view_timestamp = (std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_err(|e| {
                Error::new(ErrorDetails::ClickHouseMigration {
                    id: MIGRATION_ID.to_string(),
                    message: e.to_string(),
                })
            })?
            + view_offset)
            .as_secs();
        let on_cluster_name = self.clickhouse.get_on_cluster_name();
        let table_engine_name = self.clickhouse.get_maybe_replicated_table_engine_name(
            GetMaybeReplicatedTableEngineNameArgs {
                table_engine_name: "ReplacingMergeTree",
                table_name: "TagInference",
                engine_args: &["updated_at", "is_deleted"],
            },
        );

        let query = format!(
            r"
            CREATE TABLE IF NOT EXISTS TagInference{on_cluster_name}
                (
                    key String,
                    value String,
                    function_name LowCardinality(String),
                    variant_name LowCardinality(String),
                    episode_id UUID,
                    inference_id UUID,
                    function_type Enum8('chat' = 1, 'json' = 2),
                    is_deleted Bool DEFAULT false,
                    updated_at DateTime64(6, 'UTC') DEFAULT now64()
                ) ENGINE = {table_engine_name}
                ORDER BY (key, value, inference_id)",
        );
        let _ = self
            .clickhouse
            .run_query_synchronous_no_params(query.to_string())
            .await?;

        // Add the staled_at column to both datapoint tables
        let query = r"
            ALTER TABLE ChatInferenceDatapoint ADD COLUMN IF NOT EXISTS staled_at Nullable(DateTime64(6, 'UTC'));
        ";
        let _ = self
            .clickhouse
            .run_query_synchronous_no_params(query.to_string())
            .await?;

        let query = r"
            ALTER TABLE JsonInferenceDatapoint ADD COLUMN IF NOT EXISTS staled_at Nullable(DateTime64(6, 'UTC'));
        ";
        let _ = self
            .clickhouse
            .run_query_synchronous_no_params(query.to_string())
            .await?;

        // Update the defaults of updated_at for the Datapoint tables to be now64
        let query = r"
            ALTER TABLE ChatInferenceDatapoint MODIFY COLUMN updated_at DateTime64(6, 'UTC') default now64();
        ";
        let _ = self
            .clickhouse
            .run_query_synchronous_no_params(query.to_string())
            .await?;

        let query = r"
            ALTER TABLE JsonInferenceDatapoint MODIFY COLUMN updated_at DateTime64(6, 'UTC') default now64();
        ";
        let _ = self
            .clickhouse
            .run_query_synchronous_no_params(query.to_string())
            .await?;

        // If we are not doing a clean start, we need to add a where clause to the view to only include rows that have been created after the view_timestamp
        let view_where_clause = if clean_start {
            String::new()
        } else {
            format!("WHERE UUIDv7ToDateTime(id) >= toDateTime(toUnixTimestamp({view_timestamp}))")
        };

        let query = format!(
            r"
            CREATE MATERIALIZED VIEW IF NOT EXISTS TagChatInferenceView{on_cluster_name}
            TO TagInference
            AS
                SELECT
                    function_name,
                    variant_name,
                    episode_id,
                    id as inference_id,
                    'chat' as function_type,
                    key,
                    tags[key] as value
                FROM ChatInference
                ARRAY JOIN mapKeys(tags) as key
                {view_where_clause};
        "
        );
        let _ = self
            .clickhouse
            .run_query_synchronous_no_params(query.to_string())
            .await?;

        let query = format!(
            r"
            CREATE MATERIALIZED VIEW IF NOT EXISTS TagJsonInferenceView{on_cluster_name}
            TO TagInference
            AS
                SELECT
                    function_name,
                    variant_name,
                    episode_id,
                    id as inference_id,
                    'json' as function_type,
                    key,
                    tags[key] as value
                FROM JsonInference
                ARRAY JOIN mapKeys(tags) as key
                {view_where_clause};
        "
        );
        let _ = self
            .clickhouse
            .run_query_synchronous_no_params(query.to_string())
            .await?;

        if !clean_start {
            // Sleep for the duration specified by view_offset to allow the materialized views to catch up
            tokio::time::sleep(view_offset).await;

            let insert_chat_inference = async {
                let query = format!(
                    r"
                    INSERT INTO TagInference (key, value, function_name, variant_name, episode_id, inference_id, function_type)
                    SELECT
                        key,
                        tags[key] as value,
                        function_name,
                        variant_name,
                        episode_id,
                        id as inference_id,
                        'chat' as function_type
                    FROM ChatInference
                    ARRAY JOIN mapKeys(tags) as key
                    WHERE UUIDv7ToDateTime(id) < toDateTime(toUnixTimestamp({view_timestamp}));
                "
                );
                self.clickhouse
                    .run_query_synchronous_no_params(query.to_string())
                    .await
            };

            let insert_json_inference = async {
                let query = format!(
                    r"
                    INSERT INTO TagInference (key, value, function_name, variant_name, episode_id, inference_id, function_type)
                    SELECT
                        key,
                        tags[key] as value,
                        function_name,
                        variant_name,
                        episode_id,
                        id as inference_id,
                        'json' as function_type
                    FROM JsonInference
                    ARRAY JOIN mapKeys(tags) as key
                    WHERE UUIDv7ToDateTime(id) < toDateTime(toUnixTimestamp({view_timestamp}));
                "
                );
                self.clickhouse
                    .run_query_synchronous_no_params(query.to_string())
                    .await
            };

            tokio::try_join!(insert_chat_inference, insert_json_inference)?;
        }

        Ok(())
    }

    fn rollback_instructions(&self) -> String {
        let on_cluster_name = self.clickhouse.get_on_cluster_name();
        format!("/* Drop the materialized views */\
        DROP VIEW IF EXISTS TagChatInferenceView{on_cluster_name};
        DROP VIEW IF EXISTS TagJsonInferenceView{on_cluster_name};
        \n
        /* Drop the `TagInference` table */\
        DROP TABLE IF EXISTS TagInference{on_cluster_name} SYNC;
        /* Drop the `staled_at` column in the datapoint tables */\
        ALTER TABLE ChatInferenceDatapoint DROP COLUMN staled_at;
        ALTER TABLE JsonInferenceDatapoint DROP COLUMN staled_at;
        /* Revert the change to the default of `updated_at` in the datapoint tables */\
        ALTER TABLE ChatInferenceDatapoint MODIFY COLUMN updated_at DateTime64(6, 'UTC') default now();
        ALTER TABLE JsonInferenceDatapoint MODIFY COLUMN updated_at DateTime64(6, 'UTC') default now();
    ")
    }

    async fn has_succeeded(&self) -> Result<bool, Error> {
        let should_apply = self.should_apply().await?;
        Ok(!should_apply)
    }
}
