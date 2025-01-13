use crate::clickhouse::ClickHouseConnectionInfo;
use crate::clickhouse_migration_manager::migration_trait::Migration;
use crate::error::{Error, ErrorDetails};

use super::check_table_exists;

/// This migration is used to set up the ClickHouse database to store examples
/// for dynamic in-context learning.
pub struct Migration0002<'a> {
    pub clickhouse: &'a ClickHouseConnectionInfo,
}

impl Migration for Migration0002<'_> {
    /// Check if you can connect to the database
    async fn can_apply(&self) -> Result<(), Error> {
        self.clickhouse.health().await.map_err(|e| {
            Error::new(ErrorDetails::ClickHouseMigration {
                id: "0002".to_string(),
                message: e.to_string(),
            })
        })
    }

    /// Check if the migration has already been applied
    /// This should be equivalent to checking if `DynamicInContextLearningExample` exists
    async fn should_apply(&self) -> Result<bool, Error> {
        let exists =
            check_table_exists(self.clickhouse, "DynamicInContextLearningExample", "0002").await?;
        if !exists {
            return Ok(true);
        }

        Ok(false)
    }

    async fn apply(&self) -> Result<(), Error> {
        // Create the `DynamicInContextLearningExample` table
        let query = r#"
            CREATE TABLE IF NOT EXISTS DynamicInContextLearningExample
            (
                id UUID, -- must be a UUIDv7
                function_name LowCardinality(String),
                variant_name LowCardinality(String),
                namespace String,
                input String,
                output String,
                embedding Array(Float32),
                timestamp DateTime MATERIALIZED UUIDv7ToDateTime(id)
            ) ENGINE = MergeTree()
            ORDER BY (function_name, variant_name, namespace);
        "#;
        let _ = self.clickhouse.run_query(query.to_string()).await?;
        Ok(())
    }

    fn rollback_instructions(&self) -> String {
        "\
            -- Drop the table\n\
            DROP TABLE IF EXISTS DynamicInContextLearningExample;\n\
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
