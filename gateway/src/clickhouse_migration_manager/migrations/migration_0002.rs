use crate::clickhouse::ClickHouseConnectionInfo;
use crate::clickhouse_migration_manager::migration_trait::Migration;
use crate::error::Error;

/// This migration is used to set up the ClickHouse database to store examples
/// for dynamic in-context learning.
pub struct Migration0002<'a> {
    pub clickhouse: &'a ClickHouseConnectionInfo,
}

impl<'a> Migration for Migration0002<'a> {
    /// Check if you can connect to the database
    async fn can_apply(&self) -> Result<(), Error> {
        self.clickhouse
            .health()
            .await
            .map_err(|e| Error::ClickHouseMigration {
                id: "0002".to_string(),
                message: e.to_string(),
            })
    }

    /// Check if the migration has already been applied
    /// This should be equivalent to checking if `DiclExample` exists
    async fn should_apply(&self) -> Result<bool, Error> {
        let database = self.clickhouse.database();

        let query = format!(
            r#"SELECT EXISTS(
                SELECT 1
                FROM system.tables
                WHERE database = '{database}' AND name = 'DiclExample'
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

        Ok(false)
    }

    async fn apply(&self) -> Result<(), Error> {
        // Create the `DiclExample` table
        let query = r#"
            CREATE TABLE IF NOT EXISTS DiclExample
            (
                id UUID, -- must be a UUIDv7
                function_name LowCardinality(String),
                variant_name LowCardinality(String),
                input String,
                output String,
                embedding Array(Float32),
                timestamp DateTime MATERIALIZED UUIDv7ToDateTime(id)
            ) ENGINE = MergeTree()
            ORDER BY (function_name, variant_name);
        "#;
        let _ = self.clickhouse.run_query(query.to_string()).await?;
        Ok(())
    }

    fn rollback_instructions(&self) -> String {
        "\
            -- Drop the table\n\
            DROP TABLE IF EXISTS DiclExample;\n\
            \n\
            "
        .to_string()
    }

    /// Check if the migration has succeeded (i.e. it should not be applied again)
    async fn has_succeeded(&self) -> Result<bool, Error> {
        let should_apply = self.should_apply().await?;
        Ok(!should_apply)
    }
}
