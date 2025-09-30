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
        
        self.clickhouse
            .get_create_table_statements(
                "TagInference",
                &format!(
                    r"
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
                    )"
                ),
                &GetMaybeReplicatedTableEngineNameArgs {
                    table_engine_name: "ReplacingMergeTree",
                    table_name: "TagInference",
                    engine_args: &["updated_at", "is_deleted"],
                },
                Some("ORDER BY (key, value, inference_id)"),
                Some("cityHash64(episode_id)"),
            )
            .await?;

        // Add the staled_at column to both datapoint tables using sharding-aware ALTER
        self.clickhouse
            .get_alter_table_statements(
                "ChatInferenceDatapoint",
                "ADD COLUMN IF NOT EXISTS staled_at Nullable(DateTime64(6, 'UTC'))",
                false,
            )
            .await?;

        self.clickhouse
            .get_alter_table_statements(
                "JsonInferenceDatapoint", 
                "ADD COLUMN IF NOT EXISTS staled_at Nullable(DateTime64(6, 'UTC'))",
                false,
            )
            .await?;

        // Update the defaults of updated_at for the Datapoint tables to be now64
        self.clickhouse
            .get_alter_table_statements(
                "ChatInferenceDatapoint",
                "MODIFY COLUMN updated_at DateTime64(6, 'UTC') default now64()",
                false,
            )
            .await?;

        self.clickhouse
            .get_alter_table_statements(
                "JsonInferenceDatapoint",
                "MODIFY COLUMN updated_at DateTime64(6, 'UTC') default now64()",
                false,
            )
            .await?;

        // If we are not doing a clean start, we need to add a where clause to the view to only include rows that have been created after the view_timestamp
        let view_where_clause = if clean_start {
            String::new()
        } else {
            format!("WHERE UUIDv7ToDateTime(id) >= toDateTime(toUnixTimestamp({view_timestamp}))")
        };

        // Create materialized view for TagInference from ChatInference
        // Note: This uses ARRAY JOIN which doesn't fit the standard helper pattern,
        // so we create it manually with proper sharding logic
        let tag_inference_target = "TagInference"; // Always use distributed table for materialized view targets
        let chat_inference_source = if self.clickhouse.is_sharding_enabled() {
            self.clickhouse.get_local_table_name("ChatInference")
        } else {
            "ChatInference".to_string()
        };
        
        let view_where_condition = if view_where_clause.is_empty() {
            String::new()
        } else {
            format!(" {}", view_where_clause)
        };
        
        let on_cluster_name = self.clickhouse.get_on_cluster_name();
        let query = format!(
            r"
            CREATE MATERIALIZED VIEW IF NOT EXISTS TagChatInferenceView{on_cluster_name}
            TO {tag_inference_target}
            AS
                SELECT
                    function_name,
                    variant_name,
                    episode_id,
                    id as inference_id,
                    'chat' as function_type,
                    key,
                    tags[key] as value
                FROM {chat_inference_source}
                ARRAY JOIN mapKeys(tags) as key{view_where_condition}
            "
        );
        self.clickhouse.run_query_synchronous_no_params(query).await?;

        // Create materialized view for TagInference from JsonInference
        let json_inference_source = if self.clickhouse.is_sharding_enabled() {
            self.clickhouse.get_local_table_name("JsonInference")
        } else {
            "JsonInference".to_string()
        };
        
        let query = format!(
            r"
            CREATE MATERIALIZED VIEW IF NOT EXISTS TagJsonInferenceView{on_cluster_name}
            TO {tag_inference_target}
            AS
                SELECT
                    function_name,
                    variant_name,
                    episode_id,
                    id as inference_id,
                    'json' as function_type,
                    key,
                    tags[key] as value
                FROM {json_inference_source}
                ARRAY JOIN mapKeys(tags) as key{view_where_condition}
            "
        );
        self.clickhouse.run_query_synchronous_no_params(query).await?;

        if !clean_start {
            // Sleep for the duration specified by view_offset to allow the materialized views to catch up
            tokio::time::sleep(view_offset).await;

            // For INSERT operations and SELECT queries, we use distributed table names in sharding environments
            let tag_inference_insert_target = "TagInference"; // Always use distributed name for INSERTs
            let chat_inference_select_source = "ChatInference"; // Always use distributed name for SELECTs
            let json_inference_select_source = "JsonInference"; // Always use distributed name for SELECTs

            let insert_chat_inference = async {
                let query = format!(
                    r"
                    INSERT INTO {tag_inference_insert_target} (key, value, function_name, variant_name, episode_id, inference_id, function_type)
                    SELECT
                        key,
                        tags[key] as value,
                        function_name,
                        variant_name,
                        episode_id,
                        id as inference_id,
                        'chat' as function_type
                    FROM {chat_inference_select_source}
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
                    INSERT INTO {tag_inference_insert_target} (key, value, function_name, variant_name, episode_id, inference_id, function_type)
                    SELECT
                        key,
                        tags[key] as value,
                        function_name,
                        variant_name,
                        episode_id,
                        id as inference_id,
                        'json' as function_type
                    FROM {json_inference_select_source}
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
        
        format!(
            "/* Drop the materialized views */\
            DROP VIEW IF EXISTS TagChatInferenceView{on_cluster_name};\
            DROP VIEW IF EXISTS TagJsonInferenceView{on_cluster_name};\
            /* Drop the TagInference table */\
            {}\
            /* Drop the staled_at column in the datapoint tables */\
            {}\
            {}\
            /* Revert the change to the default of updated_at in the datapoint tables */\
            {}\
            {}",
            self.clickhouse.get_drop_table_rollback_statements("TagInference"),
            self.clickhouse.get_alter_table_rollback_statements("ChatInferenceDatapoint", "DROP COLUMN staled_at", false),
            self.clickhouse.get_alter_table_rollback_statements("JsonInferenceDatapoint", "DROP COLUMN staled_at", false),
            self.clickhouse.get_alter_table_rollback_statements("ChatInferenceDatapoint", "MODIFY COLUMN updated_at DateTime64(6, 'UTC') default now()", false),
            self.clickhouse.get_alter_table_rollback_statements("JsonInferenceDatapoint", "MODIFY COLUMN updated_at DateTime64(6, 'UTC') default now()", false)
        )
    }

    async fn has_succeeded(&self) -> Result<bool, Error> {
        let should_apply = self.should_apply().await?;
        Ok(!should_apply)
    }
}
