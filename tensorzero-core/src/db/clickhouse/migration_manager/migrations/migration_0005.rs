use crate::db::clickhouse::migration_manager::migration_trait::Migration;
use crate::db::clickhouse::{ClickHouseConnectionInfo, GetMaybeReplicatedTableEngineNameArgs};
use crate::error::{Error, ErrorDetails};
use async_trait::async_trait;

use super::{check_column_exists, check_table_exists};

/// This migration is used to set up the ClickHouse database for tagged inferences
/// The primary queries we contemplate are: Select all inferences for a given tag, or select all tags for a given inference
/// We will store the tags in a new table `InferenceTag` and create a materialized view for each original inference table that writes them
/// We will also denormalize and store the tags on the original tables for efficiency
/// There are 3 main changes:
///
///  - First, we create a new table `InferenceTag` to store the tags
///  - Second, we add a column `tags` to each original inference table
///  - Third, we create a materialized view for each original inference table that writes the tags to the `InferenceTag` table
pub struct Migration0005<'a> {
    pub clickhouse: &'a ClickHouseConnectionInfo,
}

const MIGRATION_ID: &str = "0005";

#[async_trait]
impl Migration for Migration0005<'_> {
    /// Check if the two inference tables exist as the sources for the materialized views
    /// If all of this is OK, then we can apply the migration
    async fn can_apply(&self) -> Result<(), Error> {
        let tables = vec!["ChatInference", "JsonInference"];
        for table in tables {
            match check_table_exists(self.clickhouse, table, MIGRATION_ID).await {
                Ok(exists) => {
                    if !exists {
                        return Err(ErrorDetails::ClickHouseMigration {
                            id: MIGRATION_ID.to_string(),
                            message: format!("Table {table} does not exist"),
                        }
                        .into());
                    }
                }
                Err(e) => return Err(e),
            }
        }

        Ok(())
    }

    /// Check if the migration has already been applied by checking if the new columns exist
    async fn should_apply(&self) -> Result<bool, Error> {
        if !check_table_exists(self.clickhouse, "InferenceTag", MIGRATION_ID).await? {
            return Ok(true);
        }

        // Check each of the original inference tables for a `tags` column
        let tables = vec!["ChatInference", "JsonInference"];

        for table in tables {
            check_column_exists(self.clickhouse, table, "tags", MIGRATION_ID).await?;
        }

        // Check that each of the materialized views exists
        let views = vec!["ChatInferenceTagView", "JsonInferenceTagView"];

        for view in views {
            if !check_table_exists(self.clickhouse, view, "0005").await? {
                return Ok(true);
            }
        }
        // Everything is in place, so we should not apply the migration
        Ok(false)
    }

    async fn apply(&self, _clean_start: bool) -> Result<(), Error> {
        // Create the `InferenceTag` table
        let table_engine_name = self.clickhouse.get_maybe_replicated_table_engine_name(
            GetMaybeReplicatedTableEngineNameArgs {
                table_engine_name: "MergeTree",
                table_name: "InferenceTag",
                engine_args: &[],
            },
        );
        let on_cluster_name = self.clickhouse.get_on_cluster_name();
        let query = format!(
            r"
            CREATE TABLE IF NOT EXISTS InferenceTag{on_cluster_name}
            (
                function_name LowCardinality(String),
                key String,
                value String,
                inference_id UUID, -- must be a UUIDv7
            ) ENGINE = {table_engine_name}
            ORDER BY (function_name, key, value);
        ",
        );
        let _ = self
            .clickhouse
            .run_query_synchronous_no_params(query.to_string())
            .await?;

        // Add a column `tags` to the `BooleanMetricFeedback` table
        let query = r"
            ALTER TABLE ChatInference
            ADD COLUMN IF NOT EXISTS tags Map(String, String) DEFAULT map();";
        let _ = self
            .clickhouse
            .run_query_synchronous_no_params(query.to_string())
            .await?;

        // Add a column `tags` to the `JsonInference` table
        let query = r"
            ALTER TABLE JsonInference
            ADD COLUMN IF NOT EXISTS tags Map(String, String) DEFAULT map();";
        let _ = self
            .clickhouse
            .run_query_synchronous_no_params(query.to_string())
            .await?;

        // In the following few queries we create the materialized views that map the tags from the original tables to the new `InferenceTag` table
        // We do not need to handle the case where there are already tags in the table since we created those columns just now.
        // So, we don't worry about timestamps for cutting over to the materialized views.
        // Create the materialized view for the `InferenceTag` table from ChatInference
        let query = format!(
            r"
            CREATE MATERIALIZED VIEW IF NOT EXISTS ChatInferenceTagView{on_cluster_name}
            TO InferenceTag
            AS
                SELECT
                    function_name,
                    key,
                    tags[key] as value,
                    id as inference_id
                FROM ChatInference
                ARRAY JOIN mapKeys(tags) as key
            "
        );
        let _ = self
            .clickhouse
            .run_query_synchronous_no_params(query.to_string())
            .await?;

        // Create the materialized view for the `InferenceTag` table from JsonInference
        let query = format!(
            r"
            CREATE MATERIALIZED VIEW IF NOT EXISTS JsonInferenceTagView{on_cluster_name}
            TO InferenceTag
            AS
                SELECT
                    function_name,
                    key,
                    tags[key] as value,
                    id as inference_id
                FROM JsonInference
                ARRAY JOIN mapKeys(tags) as key
            "
        );
        let _ = self
            .clickhouse
            .run_query_synchronous_no_params(query.to_string())
            .await?;
        Ok(())
    }

    fn rollback_instructions(&self) -> String {
        let on_cluster_name = self.clickhouse.get_on_cluster_name();
        format!(
            "/* Drop the materialized views */\
            DROP VIEW IF EXISTS ChatInferenceTagView{on_cluster_name};
            DROP VIEW IF EXISTS JsonInferenceTagView{on_cluster_name};
            /* Drop the `InferenceTag` table */\
            DROP TABLE IF EXISTS InferenceTag{on_cluster_name} SYNC;
            /* Drop the `tags` column from the original inference tables */\
            ALTER TABLE ChatInference DROP COLUMN tags;
            ALTER TABLE JsonInference DROP COLUMN tags;
        "
        )
    }

    /// Check if the migration has succeeded (i.e. it should not be applied again)
    async fn has_succeeded(&self) -> Result<bool, Error> {
        let should_apply = self.should_apply().await?;
        Ok(!should_apply)
    }
}
