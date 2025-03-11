use std::time::Duration;

use crate::clickhouse::migration_manager::migration_trait::Migration;
use crate::clickhouse::ClickHouseConnectionInfo;
use crate::error::{Error, ErrorDetails};

use super::check_table_exists;

/// This migration is used to set up the ClickHouse database to allow for efficient querying by episode ID
/// To do this, we need to be able to efficiently query what function and variant was called for a
/// particular episode ID.
///
/// As the original table was not set up to index on episode ID we instead need to create
/// a materialized view that is indexed by episode ID and maps to the function_name and variant_name that was used.
/// Since we also would like to be able to get the original row from the episode ID we will keep the function_name,
/// variant_name and inference ID in the materialized view so that we can use them to query the original table.
pub struct Migration0007<'a> {
    pub clickhouse: &'a ClickHouseConnectionInfo,
    pub clean_start: bool,
}

impl Migration for Migration0007<'_> {
    /// Check if you can connect to the database
    /// Then check if the two inference tables exist as the sources for the materialized views
    /// If all of this is OK, then we can apply the migration
    async fn can_apply(&self) -> Result<(), Error> {
        self.clickhouse.health().await.map_err(|e| {
            Error::new(ErrorDetails::ClickHouseMigration {
                id: "0007".to_string(),
                message: e.to_string(),
            })
        })?;

        let tables = vec!["ChatInference", "JsonInference"];

        for table in tables {
            if !check_table_exists(self.clickhouse, table, "0007").await? {
                return Err(ErrorDetails::ClickHouseMigration {
                    id: "0007".to_string(),
                    message: format!("Table {} does not exist", table),
                }
                .into());
            }
        }

        Ok(())
    }

    /// Check if the migration has already been applied
    /// This should be equivalent to checking if `InferenceByEpisodeId` exists
    async fn should_apply(&self) -> Result<bool, Error> {
        let exists = check_table_exists(self.clickhouse, "InferenceByEpisodeId", "0007").await?;
        if !exists {
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
                    id: "0007".to_string(),
                    message: e.to_string(),
                })
            })?
            + view_offset)
            .as_secs();

        // Create the `InferenceByEpisodeId` table
        let query = r#"
            CREATE TABLE IF NOT EXISTS InferenceByEpisodeId
            (
                episode_id UUID,
                id UUID,
                function_name LowCardinality(String),
                variant_name LowCardinality(String),
                function_type Enum8('chat' = 1, 'json' = 2),
            )
            ENGINE = MergeTree
            ORDER BY (episode_id, id);
        "#;
        let _ = self.clickhouse.run_query(query.to_string(), None).await?;

        // If we are not doing a clean start, we need to add a where clause to the view to only include rows that have been created after the view_timestamp
        let view_where_clause = if !self.clean_start {
            format!("WHERE UUIDv7ToDateTime(id) >= toDateTime(toUnixTimestamp({view_timestamp}))")
        } else {
            String::new()
        };
        let query = format!(
            r#"
            CREATE MATERIALIZED VIEW ChatInferenceByEpisodeIdView
            TO InferenceByEpisodeId
            AS
                SELECT
                    episode_id,
                    id,
                    function_name,
                    variant_name,
                    'chat'
                FROM ChatInference
                {view_where_clause};
            "#,
            view_where_clause = view_where_clause
        );
        let _ = self.clickhouse.run_query(query, None).await?;

        // Create the materialized view for the `InferenceByEpisodeId` table from JsonInference
        let query = format!(
            r#"
            CREATE MATERIALIZED VIEW JsonInferenceByEpisodeIdView
            TO InferenceByEpisodeId
            AS
                SELECT
                    episode_id,
                    id,
                    function_name,
                    variant_name,
                    'json'
                FROM JsonInference
                {view_where_clause};
            "#,
            view_where_clause = view_where_clause
        );
        let _ = self.clickhouse.run_query(query, None).await?;

        if !self.clean_start {
            // Sleep for the duration specified by view_offset to allow the materialized views to catch up
            tokio::time::sleep(view_offset).await;

            // Insert the data from the original tables into the new table (we do this concurrently since it could theoretically take a long time)
            let insert_chat_inference = async {
                let query = format!(
                    r#"
                    INSERT INTO InferenceByEpisodeId
                    SELECT
                        episode_id,
                        id,
                        function_name,
                        variant_name,
                        'chat'
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
                    INSERT INTO InferenceByEpisodeId
                    SELECT
                        episode_id,
                        id,
                        function_name,
                        variant_name,
                        'json'
                    FROM JsonInference
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
            DROP VIEW IF EXISTS ChatInferenceByEpisodeView;\n\
            DROP VIEW IF EXISTS JsonInferenceByEpisodeIdView;\n\
            \n\
            -- Drop the table\n\
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
