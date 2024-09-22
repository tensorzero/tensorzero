use crate::clickhouse::ClickHouseConnectionInfo;
use crate::clickhouse_migration_manager::migration_trait::Migration;
use crate::error::Error;

/// This migration is used to set up the ClickHouse database to efficiently validate
/// the type of demonstrations.
/// To do this, we need to be able to efficiently query what function was called for a
/// particular inference ID.
///
/// As the original table was not set up to index on inference ID we instead need to create
/// a materialized view that is indexed by inference ID and maps to the function_name that was used.
/// Since we also would like to be able to get the original row from the inference ID we will keep the function_name,
/// variant_name and episode_id in the materialized view so that we can use them to query the original table.

pub struct Migration0001<'a> {
    pub clickhouse: &'a ClickHouseConnectionInfo,
}

impl<'a> Migration for Migration0001<'a> {
    /// Check if you can connect to the database
    async fn can_apply(&self) -> Result<(), Error> {
        self.clickhouse
            .health()
            .await
            .map_err(|e| Error::ClickHouseMigration {
                id: "0001".to_string(),
                message: e.to_string(),
            })
    }

    /// Check if the tables exist
    async fn should_apply(&self) -> Result<bool, Error> {
        let database = self.clickhouse.database();

        let tables = vec!["InferenceView"];

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
                        id: "0001".to_string(),
                        message: e.to_string(),
                    })
                }
                Ok(response) => {
                    if response.trim() != "1" {
                        return Ok(true);
                    }
                }
            }
        }

        Ok(false)
    }

    async fn apply(&self) -> Result<(), Error> {
        // Create the database if it doesn't exist
        self.clickhouse.create_database().await?;

        // Create the `InferencesById` table
        let query = r#"
            CREATE TABLE IF NOT EXISTS InferenceById
            (
                id UUID, -- must be a UUIDv7
                function_name LowCardinality(String),
                variant_name LowCardinality(String),
                episode_id UUID, -- must be a UUIDv7
            ) ENGINE = MergeTree()
            ORDER BY id;
        "#;
        let _ = self.clickhouse.run_query(query.to_string()).await?;

        // Create the materialized view for the `InferencesById` table from ChatInference
        let query = r#"
            CREATE MATERIALIZED VIEW ChatInferenceByIdView
            TO InferenceById
            AS
                SELECT
                    id,
                    function_name,
                    variant_name,
                    episode_id
                FROM ChatInferences;
        "#;
        let _ = self.clickhouse.run_query(query.to_string()).await?;

        // Create the materialized view for the `InferencesById` table from JsonInference
        let query = r#"
            CREATE MATERIALIZED VIEW JsonInferenceByIdView
            TO InferenceById
            AS
                SELECT
                    id,
                    function_name,
                    variant_name,
                    episode_id
                FROM JsonInference;
        "#;
        let _ = self.clickhouse.run_query(query.to_string()).await?;

        // Insert the data from the original tables into the new table (we do this concurrently since it could theoretically take a long time)
        let insert_chat_inference = async {
            let query = r#"
                INSERT INTO InferenceById
                SELECT
                    id,
                    function_name,
                    variant_name,
                    episode_id
                FROM ChatInference
                WHERE UUIDv7ToDateTime(id) < now();
            "#;
            self.clickhouse.run_query(query.to_string()).await
        };

        let insert_json_inference = async {
            let query = r#"
                INSERT INTO InferenceById
                SELECT
                    id,
                    function_name,
                    variant_name,
                    episode_id
                FROM JsonInference
                WHERE UUIDv7ToDateTime(id) < now();
            "#;
            self.clickhouse.run_query(query.to_string()).await
        };

        tokio::try_join!(insert_chat_inference, insert_json_inference)?;

        Ok(())
    }

    fn rollback_instructions(&self) -> String {
        let database = self.clickhouse.database();

        format!(
            "\
            **CAREFUL: THIS WILL DELETE ALL DATA**\n\
            \n\
            -- Drop the database\n\
            DROP DATABASE IF EXISTS {database};\n\
            \n\
            -- Drop the materialized views\n\
            DROP MATERIALIZED VIEW IF EXISTS ChatInferenceByIdView;\n\
            DROP MATERIALIZED VIEW IF EXISTS JsonInferenceByIdView;\n\
            \n\
            -- Drop the table\n\
            DROP TABLE IF EXISTS InferenceById;\n\
            "
        )
    }

    /// Check if the migration has succeeded (i.e. it should not be applied again)
    async fn has_succeeded(&self) -> Result<bool, Error> {
        let should_apply = self.should_apply().await?;
        Ok(!should_apply)
    }
}
