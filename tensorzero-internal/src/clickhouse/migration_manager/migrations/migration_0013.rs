use std::time::Duration;

use crate::clickhouse::migration_manager::migration_trait::Migration;
use crate::clickhouse::ClickHouseConnectionInfo;
use crate::error::{Error, ErrorDetails};

use super::check_table_exists;
use async_trait::async_trait;

/// This migration reinitializes the `InferenceById` and `InferenceByEpisodeId` tables and
/// their associated materialized views.
///
/// As ClickHouse stores UUIDs big-endian which for UUIDv7 gives a sorting order that
/// ignores the embedded timestamps. To rectify this we should sort by
/// id_uint (the id converted to a UInt128), which correctly handles the timestamp.
///
/// This migration also installs a user-defined function `uint_to_uuid` which converts a UInt128 to a UUID
/// assuming the UInt128 was created from toUInt128(uuid).
///
/// This migration should subsume migrations 0007 and 0010.
/// They should have been removed from the binary upon merging of this migration.
pub struct Migration0013<'a> {
    pub clickhouse: &'a ClickHouseConnectionInfo,
    pub clean_start: bool,
}

#[async_trait]
impl Migration for Migration0013<'_> {
    /// Check if the two inference tables exist as the sources for the materialized views
    /// If all of this is OK, then we can apply the migration
    async fn can_apply(&self) -> Result<(), Error> {
        let tables = vec!["ChatInference", "JsonInference"];

        for table in tables {
            if !check_table_exists(self.clickhouse, table, "0013").await? {
                return Err(ErrorDetails::ClickHouseMigration {
                    id: "0013".to_string(),
                    message: format!("Table {} does not exist", table),
                }
                .into());
            }
        }

        Ok(())
    }

    /// Check if the migration has already been applied
    /// This should be equivalent to checking if `InferenceById` and `InferenceByEpisodeId` exist
    /// We also need to check if the materialized views have been created.
    async fn should_apply(&self) -> Result<bool, Error> {
        let inference_by_id_exists =
            check_table_exists(self.clickhouse, "InferenceById", "0013").await?;
        if !inference_by_id_exists {
            return Ok(true);
        }
        let inference_by_episode_id_exists =
            check_table_exists(self.clickhouse, "InferenceByEpisodeId", "0013").await?;
        if !inference_by_episode_id_exists {
            return Ok(true);
        }
        let json_inference_by_id_view_exists =
            check_table_exists(self.clickhouse, "JsonInferenceByIdView", "0013").await?;
        if !json_inference_by_id_view_exists {
            return Ok(true);
        }
        let chat_inference_by_id_view_exists =
            check_table_exists(self.clickhouse, "ChatInferenceByIdView", "0013").await?;
        if !chat_inference_by_id_view_exists {
            return Ok(true);
        }
        let json_inference_by_episode_id_view_exists =
            check_table_exists(self.clickhouse, "JsonInferenceByEpisodeIdView", "0013").await?;
        if !json_inference_by_episode_id_view_exists {
            return Ok(true);
        }
        let chat_inference_by_episode_id_view_exists =
            check_table_exists(self.clickhouse, "ChatInferenceByEpisodeIdView", "0013").await?;
        if !chat_inference_by_episode_id_view_exists {
            return Ok(true);
        }
        let query = "SHOW CREATE TABLE InferenceById".to_string();
        let result = self.clickhouse.run_query(query, None).await?;
        if !result.contains("UInt128") {
            return Err(ErrorDetails::ClickHouseMigration {
                id: "0013".to_string(),
                message:
                    "InferenceById table is in an invalid state. Please contact TensorZero team."
                        .to_string(),
            }
            .into());
        }
        let query = "SHOW CREATE TABLE InferenceByEpisodeId".to_string();
        let result = self.clickhouse.run_query(query, None).await?;
        if !result.contains("UInt128") {
            return Err(ErrorDetails::ClickHouseMigration {
                id: "0013".to_string(),
                message:
                    "InferenceByEpisodeId table is in an invalid state. Please contact TensorZero team."
                        .to_string(),
            }
            .into());
        }
        let query = "SELECT 1 FROM system.functions WHERE name = 'uint_to_uuid'".to_string();
        let result = self.clickhouse.run_query(query, None).await?;
        if !result.contains("1") {
            return Ok(true);
        }
        Ok(false)
    }

    async fn apply(&self) -> Result<(), Error> {
        // Only gets used when we are not doing a clean start
        let view_offset = Duration::from_secs(15);
        let view_timestamp = (std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_err(|e| {
                Error::new(ErrorDetails::ClickHouseMigration {
                    id: "0013".to_string(),
                    message: e.to_string(),
                })
            })?
            + view_offset)
            .as_secs();
        let query = "SELECT toUInt32(COUNT())  FROM ChatInference".to_string();
        let chat_count: usize = self
            .clickhouse
            .run_query(query, None)
            .await?
            .trim()
            .parse()
            .map_err(|e| {
                Error::new(ErrorDetails::ClickHouseMigration {
                    id: "0013".to_string(),
                    message: format!("failed to query if data is in Chat table: {e}"),
                })
            })?;
        let query = "SELECT toUInt32(COUNT())  FROM JsonInference".to_string();
        let json_count: usize = self
            .clickhouse
            .run_query(query, None)
            .await?
            .trim()
            .parse()
            .map_err(|e| {
                Error::new(ErrorDetails::ClickHouseMigration {
                    id: "0013".to_string(),
                    message: format!("failed to query if data is in Json table: {e}"),
                })
            })?;
        let inference_by_id_exists =
            check_table_exists(self.clickhouse, "InferenceById", "0013").await?;
        let inference_by_episode_id_exists =
            check_table_exists(self.clickhouse, "InferenceByEpisodeId", "0013").await?;
        let json_has_data = json_count > 0;
        let chat_has_data = chat_count > 0;
        if (json_has_data || chat_has_data)
            && (!inference_by_id_exists || !inference_by_episode_id_exists)
        {
            return Err(ErrorDetails::ClickHouseMigration {
                id: "0013".to_string(),
                message: "Data already exists in the ChatInference or JsonInference tables and InferenceById or InferenceByEpisodeId is missing. Please contact TensorZero team.".to_string(),
            }
            .into());
        }
        // Drop the original tables and materialized views (if they exist)
        // NOTE: We are removing these drops to ensure idempotency in migrations
        //       This should not affect any TensorZero users that are up to date as of 2025-03
        //       If you are seeing issues with this migration please contact the TensorZero team.
        //       We can drop these because we are now erroring if the database is not up to date.
        // let query = "DROP TABLE IF EXISTS InferenceById".to_string();
        // let _ = self.clickhouse.run_query(query, None).await?;
        // let query = "DROP VIEW IF EXISTS ChatInferenceByIdView".to_string();
        // let _ = self.clickhouse.run_query(query, None).await?;
        // let query = "DROP VIEW IF EXISTS JsonInferenceByIdView".to_string();
        // let _ = self.clickhouse.run_query(query, None).await?;
        // let query = "DROP TABLE IF EXISTS InferenceByEpisodeId".to_string();
        // let _ = self.clickhouse.run_query(query, None).await?;
        // let query = "DROP VIEW IF EXISTS ChatInferenceByEpisodeIdView".to_string();
        // let _ = self.clickhouse.run_query(query, None).await?;
        // let query = "DROP VIEW IF EXISTS JsonInferenceByEpisodeIdView".to_string();
        // let _ = self.clickhouse.run_query(query, None).await?;
        // Create the new tables with UInt128 primary keys
        // Create the `InferenceById` table
        let query = r#"
            CREATE TABLE IF NOT EXISTS InferenceById
            (
                id_uint UInt128,
                function_name LowCardinality(String),
                variant_name LowCardinality(String),
                episode_id UUID, -- must be a UUIDv7
                function_type Enum8('chat' = 1, 'json' = 2)
            ) ENGINE = MergeTree()
            ORDER BY id_uint;
        "#;
        let _ = self.clickhouse.run_query(query.to_string(), None).await?;
        // Create the `InferenceByEpisodeId` table
        let query = r#"
            CREATE TABLE IF NOT EXISTS InferenceByEpisodeId
            (
                episode_id_uint UInt128,
                id_uint UInt128,
                function_name LowCardinality(String),
                variant_name LowCardinality(String),
                function_type Enum8('chat' = 1, 'json' = 2)
            )
            ENGINE = MergeTree()
            ORDER BY (episode_id_uint, id_uint);
        "#;
        let _ = self.clickhouse.run_query(query.to_string(), None).await?;
        // Create the `uint_to_uuid` function
        let query = r#"CREATE FUNCTION IF NOT EXISTS uint_to_uuid AS (x) -> reinterpretAsUUID(
            concat(
                substring(reinterpretAsString(x), 9, 8),
                substring(reinterpretAsString(x), 1, 8)
            )
        );"#;
        let _ = self.clickhouse.run_query(query.to_string(), None).await?;

        // If we are not doing a clean start, we need to add a where clause to the view to only include rows that have been created after the view_timestamp
        let view_where_clause = if !self.clean_start {
            format!("WHERE UUIDv7ToDateTime(id) >= toDateTime(toUnixTimestamp({view_timestamp}))")
        } else {
            String::new()
        };
        // Create the materialized views for the `InferenceById` table
        // IMPORTANT: The function_type column is now correctly set to 'chat'
        let query = format!(
            r#"
            CREATE MATERIALIZED VIEW IF NOT EXISTS ChatInferenceByIdView
            TO InferenceById
            AS
                SELECT
                    toUInt128(id) as id_uint,
                    function_name,
                    variant_name,
                    episode_id,
                    'chat' AS function_type
                FROM ChatInference
                {view_where_clause};
            "#,
            view_where_clause = view_where_clause
        );
        let _ = self.clickhouse.run_query(query, None).await?;

        // IMPORTANT: The function_type column is now correctly set to 'json'
        let query = format!(
            r#"
            CREATE MATERIALIZED VIEW IF NOT EXISTS JsonInferenceByIdView
            TO InferenceById
            AS
                SELECT
                    toUInt128(id) as id_uint,
                    function_name,
                    variant_name,
                    episode_id,
                    'json' AS function_type
                FROM JsonInference
                {view_where_clause};
            "#,
            view_where_clause = view_where_clause
        );
        let _ = self.clickhouse.run_query(query, None).await?;

        // Create the materialized view for the `InferenceByEpisodeId` table from ChatInference
        // IMPORTANT: The function_type column is now correctly set to 'chat'
        let query = format!(
            r#"
            CREATE MATERIALIZED VIEW IF NOT EXISTS ChatInferenceByEpisodeIdView
            TO InferenceByEpisodeId
            AS
                SELECT
                    toUInt128(episode_id) as episode_id_uint,
                    toUInt128(id) as id_uint,
                    function_name,
                    variant_name,
                    'chat' as function_type
                FROM ChatInference
                {view_where_clause};
            "#,
            view_where_clause = view_where_clause
        );
        let _ = self.clickhouse.run_query(query, None).await?;

        // Create the materialized view for the `InferenceByEpisodeId` table from JsonInference
        // IMPORTANT: The function_type column is now correctly set to 'json'
        let query = format!(
            r#"
            CREATE MATERIALIZED VIEW IF NOT EXISTS JsonInferenceByEpisodeIdView
            TO InferenceByEpisodeId
            AS
                SELECT
                    toUInt128(episode_id) as episode_id_uint,
                    toUInt128(id) as id_uint,
                    function_name,
                    variant_name,
                    'json' as function_type
                FROM JsonInference
                {view_where_clause};
            "#,
            view_where_clause = view_where_clause
        );
        let _ = self.clickhouse.run_query(query, None).await?;

        /*
        if !self.clean_start {
            // Sleep for the duration specified by view_offset to allow the materialized views to catch up
            tokio::time::sleep(view_offset).await;

            let insert_chat_inference = async {
                let query = format!(
                    r#"
                    INSERT INTO InferenceById
                    SELECT
                        toUInt128(id) as id_uint,
                        function_name,
                        variant_name,
                        episode_id,
                        'chat' AS function_type
                    FROM ChatInference
                    WHERE UUIDv7ToDateTime(id) < toDateTime(toUnixTimestamp({view_timestamp}));
                "#,
                    view_timestamp = view_timestamp
                );
                self.clickhouse.run_query(query, None).await
            };

            let insert_json_inference = async {
                let query = format!(
                    r#"
                    INSERT INTO InferenceById
                    SELECT
                        toUInt128(id) as id_uint,
                        function_name,
                        variant_name,
                        episode_id,
                        'json' AS function_type
                    FROM JsonInference
                    WHERE UUIDv7ToDateTime(id) < toDateTime(toUnixTimestamp({view_timestamp}));
                "#,
                    view_timestamp = view_timestamp
                );
                self.clickhouse.run_query(query, None).await
            };

            // Insert the data from the original tables into the new table (we do this concurrently since it could theoretically take a long time)
            let insert_chat_inference_by_episode_id = async {
                let query = format!(
                    r#"
                    INSERT INTO InferenceByEpisodeId
                    SELECT
                        toUInt128(episode_id) as episode_id_uint,
                        toUInt128(id) as id_uint,
                        function_name,
                        variant_name,
                        'chat' as function_type
                    FROM ChatInference
                    WHERE UUIDv7ToDateTime(id) < toDateTime(toUnixTimestamp({view_timestamp}));
                "#,
                    view_timestamp = view_timestamp
                );
                self.clickhouse.run_query(query, None).await
            };

            let insert_json_inference_by_episode_id = async {
                let query = format!(
                    r#"
                    INSERT INTO InferenceByEpisodeId
                    SELECT
                        toUInt128(episode_id) as episode_id_uint,
                        toUInt128(id) as id_uint,
                        function_name,
                        variant_name,
                        'json' as function_type
                    FROM JsonInference
                    WHERE UUIDv7ToDateTime(id) < toDateTime(toUnixTimestamp({view_timestamp}));
                "#,
                    view_timestamp = view_timestamp
                );
                self.clickhouse.run_query(query, None).await
            };

            tokio::try_join!(
                insert_chat_inference,
                insert_json_inference,
                insert_chat_inference_by_episode_id,
                insert_json_inference_by_episode_id
            )?;
        }
        */
        Ok(())
    }

    fn rollback_instructions(&self) -> String {
        "\
            -- Drop the materialized views\n\
            DROP VIEW IF EXISTS ChatInferenceByIdView;\n\
            DROP VIEW IF EXISTS JsonInferenceByIdView;\n\
            DROP VIEW IF EXISTS ChatInferenceByEpisodeIdView;\n\
            DROP VIEW IF EXISTS JsonInferenceByEpisodeIdView;\n\
            \n\
            -- Drop the function\n\
            DROP FUNCTION IF EXISTS uint_to_uuid;\n\
            \n\
            -- Drop the tables\n\
            DROP TABLE IF EXISTS InferenceById;\n\
            DROP TABLE IF EXISTS InferenceByEpisodeId;\n\
            "
        .to_string()
    }

    /// Check if the migration has succeeded (i.e. it should not be applied again)
    async fn has_succeeded(&self) -> Result<bool, Error> {
        let should_apply = self.should_apply().await?;
        Ok(!should_apply)
    }
}
