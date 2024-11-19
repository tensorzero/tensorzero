use crate::clickhouse::ClickHouseConnectionInfo;
use crate::clickhouse_migration_manager::migration_trait::Migration;
use crate::error::Error;

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

impl<'a> Migration for Migration0005<'a> {
    /// Check if you can connect to the database
    /// Then check if the two inference tables exist as the sources for the materialized views
    /// If all of this is OK, then we can apply the migration
    async fn can_apply(&self) -> Result<(), Error> {
        self.clickhouse
            .health()
            .await
            .map_err(|e| Error::ClickHouseMigration {
                id: "0005".to_string(),
                message: e.to_string(),
            })?;
        let database = self.clickhouse.database();

        let tables = vec!["ChatInference", "JsonInference"];

        for table in tables {
            let query = format!(
                r#"SELECT EXISTS(
                        SELECT 1
                        FROM system.tables
                        WHERE database = '{database}' AND name = '{table}'
                    )"#
            );

            match self.clickhouse.run_query(query).await {
                Err(e) => {
                    return Err(Error::ClickHouseMigration {
                        id: "0005".to_string(),
                        message: e.to_string(),
                    })
                }
                Ok(response) => {
                    if response.trim() != "1" {
                        return Err(Error::ClickHouseMigration {
                            id: "0005".to_string(),
                            message: format!("Table {} does not exist", table),
                        });
                    }
                }
            }
        }

        Ok(())
    }

    /// Check if the migration has already been applied by checking if the new columns exist
    async fn should_apply(&self) -> Result<bool, Error> {
        let database = self.clickhouse.database();

        let query = format!(
            r#"SELECT EXISTS(
                SELECT 1
                FROM system.tables
                WHERE database = '{database}' AND name = 'InferenceTag'
            )"#
        );

        match self.clickhouse.run_query(query).await {
            Err(e) => {
                return Err(Error::ClickHouseMigration {
                    id: "0005".to_string(),
                    message: e.to_string(),
                })
            }
            Ok(response) => {
                if response.trim() != "1" {
                    return Ok(true);
                }
            }
        }

        // Check each of the original inference tables for a `tags` column
        let tables = vec!["ChatInference", "JsonInference"];

        for table in tables {
            let query = format!(
                r#"SELECT EXISTS(
                    SELECT 1
                    FROM system.columns
                    WHERE database = '{}'
                      AND table = '{}'
                      AND name = 'tags'
                )"#,
                database, table
            );
            match self.clickhouse.run_query(query).await {
                Err(e) => {
                    return Err(Error::ClickHouseMigration {
                        id: "0005".to_string(),
                        message: e.to_string(),
                    });
                }
                Ok(response) => {
                    if response.trim() != "1" {
                        return Ok(true);
                    }
                }
            }
        }

        // Check that each of the materialized views exists
        let views = vec!["ChatInferenceTagView", "JsonInferenceTagView"];

        for view in views {
            let query = format!(
                r#"SELECT EXISTS(
                    SELECT 1
                    FROM system.tables
                    WHERE database = '{database}' AND name = '{view}'
                )"#,
            );
            match self.clickhouse.run_query(query).await {
                Err(e) => {
                    return Err(Error::ClickHouseMigration {
                        id: "0005".to_string(),
                        message: e.to_string(),
                    });
                }
                Ok(response) => {
                    if response.trim() != "1" {
                        return Ok(true);
                    }
                }
            }
        }
        // Everything is in place, so we should not apply the migration
        Ok(false)
    }

    async fn apply(&self) -> Result<(), Error> {
        // Create the `InferenceTag` table
        let query = r#"
            CREATE TABLE IF NOT EXISTS InferenceTag
            (
                function_name LowCardinality(String),
                key String,
                value String,
                inference_id UUID, -- must be a UUIDv7
            ) ENGINE = MergeTree()
            ORDER BY (function_name, key, value);
        "#;
        let _ = self.clickhouse.run_query(query.to_string()).await?;

        // Add a column `tags` to the `BooleanMetricFeedback` table
        let query = r#"
            ALTER TABLE ChatInference
            ADD COLUMN IF NOT EXISTS tags Map(String, String) DEFAULT map();"#;
        let _ = self.clickhouse.run_query(query.to_string()).await?;

        // Add a column `tags` to the `JsonInference` table
        let query = r#"
            ALTER TABLE JsonInference
            ADD COLUMN IF NOT EXISTS tags Map(String, String) DEFAULT map();"#;
        let _ = self.clickhouse.run_query(query.to_string()).await?;

        // In the following few queries we create the materialized views that map the tags from the original tables to the new `InferenceTag` table
        // We do not need to handle the case where there are already tags in the table since we created those columns just now.
        // So, we don't worry about timestamps for cutting over to the materialized views.
        // Create the materialized view for the `InferenceTag` table from ChatInference
        let query = r#"
            CREATE MATERIALIZED VIEW IF NOT EXISTS ChatInferenceTagView
            TO InferenceTag
            AS
                SELECT
                    function_name,
                    key,
                    tags[key] as value,
                    id as inference_id
                FROM ChatInference
                ARRAY JOIN mapKeys(tags) as key
            "#;
        let _ = self.clickhouse.run_query(query.to_string()).await?;

        // Create the materialized view for the `InferenceTag` table from JsonInference
        let query = r#"
            CREATE MATERIALIZED VIEW IF NOT EXISTS JsonInferenceTagView
            TO InferenceTag
            AS
                SELECT
                    function_name,
                    key,
                    tags[key] as value,
                    id as inference_id
                FROM JsonInference
                ARRAY JOIN mapKeys(tags) as key
            "#;
        let _ = self.clickhouse.run_query(query.to_string()).await?;
        Ok(())
    }

    fn rollback_instructions(&self) -> String {
        "\
            -- Drop the materialized views\n\
            DROP MATERIALIZED VIEW IF EXISTS ChatInferenceTagView;\n\
            DROP MATERIALIZED VIEW IF EXISTS JsonInferenceTagView;\n\
            \n
            -- Drop the `InferenceTag` table\n\
            DROP TABLE IF EXISTS InferenceTag;\n\
            \n\
            -- Drop the `tags` column from the original inference tables\n\
            ALTER TABLE ChatInference DROP COLUMN tags;\n\
            ALTER TABLE JsonInference DROP COLUMN tags;\n\
        "
        .to_string()
    }

    /// Check if the migration has succeeded (i.e. it should not be applied again)
    async fn has_succeeded(&self) -> Result<bool, Error> {
        let should_apply = self.should_apply().await?;
        Ok(!should_apply)
    }
}
