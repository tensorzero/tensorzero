use crate::clickhouse::ClickHouseConnectionInfo;
use crate::clickhouse_migration_manager::migration_trait::Migration;
use crate::error::{Error, ErrorDetails};

use super::{check_column_exists, check_table_exists};

/// This migration is used to set up the ClickHouse database for caching.
/// We create a table `ModelInferenceCache` that stores the `short_cache_key`, `long_cache_key`, `timestamp`, and `output`
/// The `short_cache_key` is the first 8 bytes of the `long_cache_key`, a 32 byte array. We use it as the primary key in clickhouse for fast lookups.
/// The `long_cache_key` is the cache key for the model provider request
/// The `timestamp` is the timestamp of the request
/// The `output` is the output of the request, serialized using serde_json::to_string
/// We also add a column `cache_hit` to indicate if a ModelInference was a cache hit
pub struct Migration0011<'a> {
    pub clickhouse: &'a ClickHouseConnectionInfo,
}

impl Migration for Migration0011<'_> {
    /// Check if you can connect to the database
    /// Then check if the two inference tables exist as the sources for the materialized views
    /// If all of this is OK, then we can apply the migration
    async fn can_apply(&self) -> Result<(), Error> {
        self.clickhouse.health().await.map_err(|e| {
            Error::new(ErrorDetails::ClickHouseMigration {
                id: "0011".to_string(),
                message: e.to_string(),
            })
        })?;

        Ok(())
    }

    /// Check if the migration needs to be applied
    /// This should be equivalent to checking if `ModelInferenceCache` is missing or
    /// if the `cache_hit` column is missing from `ModelInference`
    async fn should_apply(&self) -> Result<bool, Error> {
        let table_exists =
            check_table_exists(self.clickhouse, "ModelInferenceCache", "0011").await?;
        let cache_hit_column_exists =
            check_column_exists(self.clickhouse, "ModelInference", "cache_hit", "0011").await?;
        Ok(!table_exists || !cache_hit_column_exists)
    }

    async fn apply(&self) -> Result<(), Error> {
        // Create the `ModelInferenceCache` table
        let query = r#"
            CREATE TABLE IF NOT EXISTS ModelInferenceCache
            (
                short_cache_key UInt64,
                long_cache_key FixedString(64), -- for a hex-encoded 256-bit key
                timestamp DateTime DEFAULT now(),
                output String,
                raw_request String,
                raw_response String
            ) ENGINE = MergeTree()
            ORDER BY short_cache_key;
            -- TODO: consider setting a smaller granule size for improved query latency
            -- and a partitioning scheme to enable a global TTL later
        "#;
        let _ = self.clickhouse.run_query(query.to_string(), None).await?;

        // Add the `cache_hit` column to ModelInference
        let query = r#"
            ALTER TABLE ModelInference ADD COLUMN cache_hit Bool DEFAULT false;        
        "#;
        let _ = self.clickhouse.run_query(query.to_string(), None).await?;

        Ok(())
    }

    fn rollback_instructions(&self) -> String {
        "\
            -- Drop the table\n\
            DROP TABLE IF EXISTS ModelInferenceCache;\n\
            -- Drop the `cache_hit` column from ModelInference
            ALTER TABLE ModelInference DROP COLUMN cache_hit;
            "
        .to_string()
    }

    /// Check if the migration has succeeded (i.e. it should not be applied again)
    async fn has_succeeded(&self) -> Result<bool, Error> {
        let should_apply = self.should_apply().await?;
        Ok(!should_apply)
    }
}
