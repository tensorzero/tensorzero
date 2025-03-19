use std::time::Duration;

use async_trait::async_trait;

use super::check_table_exists;
use crate::clickhouse::migration_manager::migration_trait::Migration;
use crate::clickhouse::ClickHouseConnectionInfo;
use crate::error::{Error, ErrorDetails};

/// This migration adds a replacing mergetree table TagInference.
/// This table stores data that allows us to efficiently query for tags by key and value
/// and to efficiently find inferences and feedbacks associated with them
///
/// We also add materialized views TagChatInferenceView and TagJsonInferenceView,
/// These views will insert data into the TagInference table
/// when data is inserted into the original tables.
pub struct Migration0019<'a> {
    pub clickhouse: &'a ClickHouseConnectionInfo,
    pub clean_start: bool,
}

#[async_trait]
impl Migration for Migration0019<'_> {
    async fn can_apply(&self) -> Result<(), Error> {
        let chat_inference_table_exists =
            check_table_exists(self.clickhouse, "ChatInference", "0019").await?;
        let json_inference_table_exists =
            check_table_exists(self.clickhouse, "JsonInference", "0019").await?;

        if !chat_inference_table_exists || !json_inference_table_exists {
            return Err(Error::new(ErrorDetails::ClickHouseMigration {
                id: "0019".to_string(),
                message: "One or more of the inference tables do not exist".to_string(),
            }));
        }

        Ok(())
    }

    async fn should_apply(&self) -> Result<bool, Error> {
        let tag_inference_table_exists =
            check_table_exists(self.clickhouse, "TagInference", "0019").await?;
        let tag_chat_inference_view_exists =
            check_table_exists(self.clickhouse, "TagChatInferenceView", "0019").await?;
        let tag_json_inference_view_exists =
            check_table_exists(self.clickhouse, "TagJsonInferenceView", "0019").await?;
        Ok(!tag_inference_table_exists
            || !tag_chat_inference_view_exists
            || !tag_json_inference_view_exists)
    }

    async fn apply(&self) -> Result<(), Error> {
        // Only gets used when we are not doing a clean start
        let view_offset = Duration::from_secs(15);
        let view_timestamp = (std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_err(|e| {
                Error::new(ErrorDetails::ClickHouseMigration {
                    id: "0019".to_string(),
                    message: e.to_string(),
                })
            })?
            + view_offset)
            .as_secs();

        let query = r#"
            CREATE TABLE IF NOT EXISTS TagInference
                (
                    key String,
                    value String,
                    function_name LowCardinality(String),
                    variant_name LowCardinality(String),
                    episode_id UUID,
                    inference_id UUID,
                    function_type Enum8('chat' = 1, 'json' = 2),
                    is_deleted Bool DEFAULT false,
                    updated_at DateTime64(6, 'UTC') DEFAULT now()
                ) ENGINE = ReplacingMergeTree(updated_at, is_deleted)
                ORDER BY (key, value, inference_id)"#;
        let _ = self.clickhouse.run_query(query.to_string(), None).await?;

        // If we are not doing a clean start, we need to add a where clause to the view to only include rows that have been created after the view_timestamp
        let view_where_clause = if !self.clean_start {
            format!("WHERE UUIDv7ToDateTime(id) >= toDateTime(toUnixTimestamp({view_timestamp}))")
        } else {
            String::new()
        };

        let query = format!(
            r#"
            CREATE MATERIALIZED VIEW IF NOT EXISTS TagChatInferenceView
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
        "#,
            view_where_clause = view_where_clause
        );
        let _ = self.clickhouse.run_query(query.to_string(), None).await?;

        let query = format!(
            r#"
            CREATE MATERIALIZED VIEW IF NOT EXISTS TagJsonInferenceView
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
        "#,
            view_where_clause = view_where_clause
        );
        let _ = self.clickhouse.run_query(query.to_string(), None).await?;

        if !self.clean_start {
            // Sleep for the duration specified by view_offset to allow the materialized views to catch up
            tokio::time::sleep(view_offset).await;

            let insert_chat_inference = async {
                let query = format!(
                    r#"
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
                "#,
                    view_timestamp = view_timestamp
                );
                self.clickhouse.run_query(query, None).await
            };

            let insert_json_inference = async {
                let query = format!(
                    r#"
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
                "#,
                    view_timestamp = view_timestamp
                );
                self.clickhouse.run_query(query, None).await
            };

            tokio::try_join!(insert_chat_inference, insert_json_inference)?;
        }

        Ok(())
    }

    fn rollback_instructions(&self) -> String {
        "\
        -- Drop the materialized views\n\
        DROP MATERIALIZED VIEW IF EXISTS TagChatInferenceView;\n\
        DROP MATERIALIZED VIEW IF EXISTS TagJsonInferenceView;\n\
        \n
        -- Drop the `TagInference` table\n\
        DROP TABLE IF EXISTS TagInference;
    "
        .to_string()
    }

    async fn has_succeeded(&self) -> Result<bool, Error> {
        let should_apply = self.should_apply().await?;
        Ok(!should_apply)
    }
}
