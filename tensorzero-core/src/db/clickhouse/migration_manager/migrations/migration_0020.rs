use rand::prelude::*;
use std::time::Duration;

use crate::db::clickhouse::migration_manager::migration_trait::Migration;
use crate::db::clickhouse::{ClickHouseConnectionInfo, GetMaybeReplicatedTableEngineNameArgs};
use crate::error::{Error, ErrorDetails};

use super::{check_table_exists, get_table_engine};
use async_trait::async_trait;

/// This migration reinitializes the `InferenceById` and `InferenceByEpisodeId` tables and
/// their associated materialized views.
///
/// As ClickHouse stores UUIDs big-endian which for UUIDv7 gives a sorting order that
/// ignores the embedded timestamps. To rectify this we should sort by
/// `id_uint` (the id converted to a UInt128), which correctly handles the timestamp.
///
/// This migration also installs a user-defined function `uint_to_uuid` which converts a UInt128 to a UUID
/// assuming the UInt128 was created from toUInt128(uuid).
///
/// This migration should subsume migrations 0007, 0010, and 0013.
/// They should have been removed from the binary upon merging of this migration.
/// 0013 is essentially the same migration but this one uses a ReplacingMergeTree engine for idempotency.
pub struct Migration0020<'a> {
    pub clickhouse: &'a ClickHouseConnectionInfo,
}

const MIGRATION_ID: &str = "0020";

#[async_trait]
impl Migration for Migration0020<'_> {
    /// Check if the two inference tables exist as the sources for the materialized views
    /// If all of this is OK, then we can apply the migration
    async fn can_apply(&self) -> Result<(), Error> {
        let tables = vec!["ChatInference", "JsonInference"];

        for table in tables {
            if !check_table_exists(self.clickhouse, table, MIGRATION_ID).await? {
                return Err(ErrorDetails::ClickHouseMigration {
                    id: MIGRATION_ID.to_string(),
                    message: format!("Table {table} does not exist"),
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
            check_table_exists(self.clickhouse, "InferenceById", MIGRATION_ID).await?;
        if !inference_by_id_exists {
            return Ok(true);
        }
        let inference_by_id_engine = get_table_engine(self.clickhouse, "InferenceById").await?;
        if !inference_by_id_engine.contains("ReplacingMergeTree") {
            // Table was created by an older migration. We should drop and recreate
            return Ok(true);
        }
        let inference_by_episode_id_exists =
            check_table_exists(self.clickhouse, "InferenceByEpisodeId", MIGRATION_ID).await?;
        if !inference_by_episode_id_exists {
            return Ok(true);
        }
        let inference_by_episode_id_engine =
            get_table_engine(self.clickhouse, "InferenceByEpisodeId").await?;
        if !inference_by_episode_id_engine.contains("ReplacingMergeTree") {
            // Table was created by an older migration. We should drop and recreate
            return Ok(true);
        }
        let json_inference_by_id_view_exists =
            check_table_exists(self.clickhouse, "JsonInferenceByIdView", MIGRATION_ID).await?;
        if !json_inference_by_id_view_exists {
            return Ok(true);
        }
        let chat_inference_by_id_view_exists =
            check_table_exists(self.clickhouse, "ChatInferenceByIdView", MIGRATION_ID).await?;
        if !chat_inference_by_id_view_exists {
            return Ok(true);
        }
        let json_inference_by_episode_id_view_exists = check_table_exists(
            self.clickhouse,
            "JsonInferenceByEpisodeIdView",
            MIGRATION_ID,
        )
        .await?;
        if !json_inference_by_episode_id_view_exists {
            return Ok(true);
        }
        let chat_inference_by_episode_id_view_exists = check_table_exists(
            self.clickhouse,
            "ChatInferenceByEpisodeIdView",
            MIGRATION_ID,
        )
        .await?;
        if !chat_inference_by_episode_id_view_exists {
            return Ok(true);
        }
        let query = "SHOW CREATE TABLE InferenceById".to_string();
        let result = self
            .clickhouse
            .run_query_synchronous_no_params(query)
            .await?;
        if !result.response.contains("UInt128") {
            // Table was created by an older migration. We should drop and recreate
            return Ok(true);
        }
        let query = "SHOW CREATE TABLE InferenceByEpisodeId".to_string();
        let result = self
            .clickhouse
            .run_query_synchronous_no_params(query)
            .await?;
        if !result.response.contains("UInt128") {
            // Table was created by an older migration. We should drop and recreate
            return Ok(true);
        }
        let query = "SELECT 1 FROM system.functions WHERE name = 'uint_to_uuid'".to_string();
        let result = self
            .clickhouse
            .run_query_synchronous_no_params(query)
            .await?;
        if !result.response.contains("1") {
            return Ok(true);
        }
        Ok(false)
    }

    async fn apply(&self, clean_start: bool) -> Result<(), Error> {
        // Only gets used when we are not doing a clean start
        let view_offset = Duration::from_secs(15);

        // Check if the InferenceById table exists
        let inference_by_id_exists =
            check_table_exists(self.clickhouse, "InferenceById", MIGRATION_ID).await?;
        // Create the `InferenceById` table using a random suffix to avoid conflicts with other concurrent migrations
        let create_table_name = if inference_by_id_exists {
            let mut rng = rand::rng();
            let random_suffix: String = (0..16)
                .map(|_| rng.sample(rand::distr::Alphanumeric) as char)
                .collect();
            format!("InferenceById_temp0020_{random_suffix}")
        } else {
            "InferenceById".to_string()
        };
        let table_engine_name = self.clickhouse.get_maybe_replicated_table_engine_name(
            GetMaybeReplicatedTableEngineNameArgs {
                table_engine_name: "ReplacingMergeTree",
                table_name: &create_table_name,
                engine_args: &["id_uint"],
            },
        );
        let on_cluster_name = self.clickhouse.get_on_cluster_name();
        let query = format!(
            r"
            CREATE TABLE IF NOT EXISTS {create_table_name}{on_cluster_name}
            (
                id_uint UInt128,
                function_name LowCardinality(String),
                variant_name LowCardinality(String),
                episode_id UUID, -- must be a UUIDv7
                function_type Enum8('chat' = 1, 'json' = 2)
            ) ENGINE = {table_engine_name}
            ORDER BY id_uint;
        "
        );
        let _ = self
            .clickhouse
            .run_query_synchronous_no_params(query.to_string())
            .await?;
        // If the InferenceById table exists then we need to swap this table in and drop old one.
        if inference_by_id_exists {
            let query = format!("EXCHANGE TABLES InferenceById AND {create_table_name}");
            let _ = self
                .clickhouse
                .run_query_synchronous_no_params(query.to_string())
                .await?;
            let query = format!("DROP TABLE IF EXISTS {create_table_name}");
            let _ = self
                .clickhouse
                .run_query_synchronous_no_params(query.to_string())
                .await?;
        }
        // Check if the InferenceByEpisodeId table exists
        let inference_by_episode_id_exists =
            check_table_exists(self.clickhouse, "InferenceByEpisodeId", MIGRATION_ID).await?;
        // Create the `InferenceByEpisodeId` table
        let create_table_name = if inference_by_episode_id_exists {
            let mut rng = rand::rng();
            let random_suffix: String = (0..16)
                .map(|_| rng.sample(rand::distr::Alphanumeric) as char)
                .collect();
            format!("InferenceByEpisodeId_temp0020_{random_suffix}")
        } else {
            "InferenceByEpisodeId".to_string()
        };
        let table_engine_name = self.clickhouse.get_maybe_replicated_table_engine_name(
            GetMaybeReplicatedTableEngineNameArgs {
                table_engine_name: "ReplacingMergeTree",
                table_name: &create_table_name,
                engine_args: &["id_uint"],
            },
        );
        let query = format!(
            r"
            CREATE TABLE IF NOT EXISTS {create_table_name}{on_cluster_name}
            (
                episode_id_uint UInt128,
                id_uint UInt128,
                function_name LowCardinality(String),
                variant_name LowCardinality(String),
                function_type Enum8('chat' = 1, 'json' = 2)
            )
            ENGINE = {table_engine_name}
            ORDER BY (episode_id_uint, id_uint);
        "
        );
        let _ = self
            .clickhouse
            .run_query_synchronous_no_params(query.to_string())
            .await?;
        // If the InferenceByEpisodeId table exists then we need to swap this table in and drop old one.
        if inference_by_episode_id_exists {
            let query = format!("EXCHANGE TABLES InferenceByEpisodeId AND {create_table_name}");
            let _ = self
                .clickhouse
                .run_query_synchronous_no_params(query.to_string())
                .await?;
            let query = format!("DROP TABLE IF EXISTS {create_table_name}");
            let _ = self
                .clickhouse
                .run_query_synchronous_no_params(query.to_string())
                .await?;
        }
        // Create the `uint_to_uuid` function
        let query = format!(
            r"CREATE FUNCTION IF NOT EXISTS uint_to_uuid{on_cluster_name} AS (x) -> reinterpretAsUUID(
            concat(
                substring(reinterpretAsString(x), 9, 8),
                substring(reinterpretAsString(x), 1, 8)
            )
        );",
        );
        let _ = self
            .clickhouse
            .run_query_synchronous_no_params(query.to_string())
            .await?;
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

        // If we are not doing a clean start, we need to add a where clause to the view to only include rows that have been created after the view_timestamp
        let view_where_clause = if clean_start {
            String::new()
        } else {
            format!("WHERE UUIDv7ToDateTime(id) >= toDateTime(toUnixTimestamp({view_timestamp}))")
        };
        // Create the materialized views for the `InferenceById` table
        // IMPORTANT: The function_type column is now correctly set to 'chat'
        let query = format!(
            r"
            CREATE MATERIALIZED VIEW IF NOT EXISTS ChatInferenceByIdView{on_cluster_name}
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
            "
        );
        let _ = self
            .clickhouse
            .run_query_synchronous_no_params(query.to_string())
            .await?;

        // IMPORTANT: The function_type column is now correctly set to 'json'
        let query = format!(
            r"
            CREATE MATERIALIZED VIEW IF NOT EXISTS JsonInferenceByIdView{on_cluster_name}
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
            "
        );
        let _ = self
            .clickhouse
            .run_query_synchronous_no_params(query.to_string())
            .await?;

        // Create the materialized view for the `InferenceByEpisodeId` table from ChatInference
        // IMPORTANT: The function_type column is now correctly set to 'chat'
        let query = format!(
            r"
            CREATE MATERIALIZED VIEW IF NOT EXISTS ChatInferenceByEpisodeIdView{on_cluster_name}
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
            "
        );
        let _ = self
            .clickhouse
            .run_query_synchronous_no_params(query.to_string())
            .await?;

        // Create the materialized view for the `InferenceByEpisodeId` table from JsonInference
        // IMPORTANT: The function_type column is now correctly set to 'json'
        let query = format!(
            r"
            CREATE MATERIALIZED VIEW IF NOT EXISTS JsonInferenceByEpisodeIdView{on_cluster_name}
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
            "
        );
        let _ = self
            .clickhouse
            .run_query_synchronous_no_params(query.to_string())
            .await?;

        if !clean_start {
            // Sleep for the duration specified by view_offset to allow the materialized views to catch up
            tokio::time::sleep(view_offset).await;

            // Insert the data from the original tables into the new tables sequentially
            // First, insert data into InferenceById from ChatInference
            let query = format!(
                r"
                INSERT INTO InferenceById
                SELECT
                    toUInt128(id) as id_uint,
                    function_name,
                    variant_name,
                    episode_id,
                    'chat' AS function_type
                FROM ChatInference
                WHERE UUIDv7ToDateTime(id) < toDateTime(toUnixTimestamp({view_timestamp}));
            "
            );
            self.clickhouse
                .run_query_synchronous_no_params(query.to_string())
                .await?;

            // Then, insert data into InferenceById from JsonInference
            let query = format!(
                r"
                INSERT INTO InferenceById
                SELECT
                    toUInt128(id) as id_uint,
                    function_name,
                    variant_name,
                    episode_id,
                    'json' AS function_type
                FROM JsonInference
                WHERE UUIDv7ToDateTime(id) < toDateTime(toUnixTimestamp({view_timestamp}));
            "
            );
            self.clickhouse
                .run_query_synchronous_no_params(query.to_string())
                .await?;

            // Next, insert data into InferenceByEpisodeId from ChatInference
            let query = format!(
                r"
                INSERT INTO InferenceByEpisodeId
                SELECT
                    toUInt128(episode_id) as episode_id_uint,
                    toUInt128(id) as id_uint,
                    function_name,
                    variant_name,
                    'chat' as function_type
                FROM ChatInference
                WHERE UUIDv7ToDateTime(id) < toDateTime(toUnixTimestamp({view_timestamp}));
            "
            );
            self.clickhouse
                .run_query_synchronous_no_params(query.to_string())
                .await?;

            // Finally, insert data into InferenceByEpisodeId from JsonInference
            let query = format!(
                r"
                INSERT INTO InferenceByEpisodeId
                SELECT
                    toUInt128(episode_id) as episode_id_uint,
                    toUInt128(id) as id_uint,
                    function_name,
                    variant_name,
                    'json' as function_type
                FROM JsonInference
                WHERE UUIDv7ToDateTime(id) < toDateTime(toUnixTimestamp({view_timestamp}));
            "
            );
            self.clickhouse
                .run_query_synchronous_no_params(query.to_string())
                .await?;
        }
        Ok(())
    }

    // NOTE - we do *not* drop the uint_to_uuid function here, as ClickHouse user-defined functions are globally scoped.
    // Dropping the function can break other migrations running concurrently in our test suite.
    fn rollback_instructions(&self) -> String {
        let on_cluster_name = self.clickhouse.get_on_cluster_name();
        format!(
            "/* Drop the materialized views */\
            DROP VIEW IF EXISTS ChatInferenceByIdView{on_cluster_name};
            DROP VIEW IF EXISTS JsonInferenceByIdView{on_cluster_name};
            DROP VIEW IF EXISTS ChatInferenceByEpisodeIdView{on_cluster_name};
            DROP VIEW IF EXISTS JsonInferenceByEpisodeIdView{on_cluster_name};
            /* Drop the tables */\
            DROP TABLE IF EXISTS InferenceById{on_cluster_name} SYNC;
            DROP TABLE IF EXISTS InferenceByEpisodeId{on_cluster_name} SYNC;
            "
        )
        .to_string()
    }

    /// Check if the migration has succeeded (i.e. it should not be applied again)
    async fn has_succeeded(&self) -> Result<bool, Error> {
        let should_apply = self.should_apply().await?;
        Ok(!should_apply)
    }
}
