use crate::clickhouse::migration_manager::migration_trait::Migration;
use crate::clickhouse::migration_manager::migrations::{
    check_column_exists, check_table_exists, create_replacing_table_engine, create_cluster_clause
};
use crate::clickhouse::ClickHouseConnectionInfo;
use crate::config_parser::Config;
use crate::error::Error;
use async_trait::async_trait;

/// This migration is used to set up the ClickHouse database for caching.
/// We create a table `ModelInferenceCache` that stores the `short_cache_key`, `long_cache_key`, `timestamp`, and `output`
/// The `short_cache_key` is the first 8 bytes of the `long_cache_key`, a 32 byte array. We use it as the primary key in clickhouse for fast lookups.
/// The `long_cache_key` is the cache key for the model provider request
/// The `timestamp` is the timestamp of the request
/// The `output` is the output of the request, serialized using serde_json::to_string
/// We also add a column `cached` to indicate if a ModelInference was a cache hit
/// 
/// As of the replication-aware migration system, this migration creates tables with
/// the appropriate engine (replicated vs non-replicated) based on configuration.
pub struct Migration0011<'a> {
    pub clickhouse: &'a ClickHouseConnectionInfo,
    pub config: &'a Config,
}

#[async_trait]
impl Migration for Migration0011<'_> {
    async fn can_apply(&self) -> Result<(), Error> {
        Ok(())
    }

    /// Check if the migration needs to be applied
    /// This should be equivalent to checking if `ModelInferenceCache` is missing or
    /// if the `cached` column is missing from `ModelInference`
    async fn should_apply(&self) -> Result<bool, Error> {
        let table_exists =
            check_table_exists(self.clickhouse, "ModelInferenceCache", "0011").await?;
        let cached_column_exists =
            check_column_exists(self.clickhouse, "ModelInference", "cached", "0011").await?;
        Ok(!table_exists || !cached_column_exists)
    }

    async fn apply(&self, _clean_start: bool) -> Result<(), Error> {
        let cluster_clause = create_cluster_clause(
            self.config.clickhouse.replication_enabled, 
            &self.config.clickhouse.cluster_name
        );
        
        // Create the `ModelInferenceCache` table
        let engine = create_replacing_table_engine(
            self.config.clickhouse.replication_enabled,
            &self.config.clickhouse.cluster_name,
            "ModelInferenceCache",
            Some("timestamp, is_deleted")
        );
        let query = format!(
            r"CREATE TABLE IF NOT EXISTS ModelInferenceCache {cluster_clause}
            (
                short_cache_key UInt64,
                long_cache_key FixedString(64), -- for a hex-encoded 256-bit key
                timestamp DateTime DEFAULT now(),
                output String,
                raw_request String,
                raw_response String,
                is_deleted Bool DEFAULT false,
                INDEX idx_long_cache_key long_cache_key TYPE bloom_filter GRANULARITY 100
            ) ENGINE = {engine}
            PARTITION BY toYYYYMM(timestamp)
            ORDER BY (short_cache_key, long_cache_key)
            PRIMARY KEY (short_cache_key)
            SETTINGS index_granularity = 256"
        );
        let _ = self
            .clickhouse
            .run_query_synchronous_no_params(query)
            .await?;

        // Add the `cached` column to ModelInference
        let query = r"
            ALTER TABLE ModelInference ADD COLUMN IF NOT EXISTS cached Bool DEFAULT false;
        ";
        let _ = self
            .clickhouse
            .run_query_synchronous_no_params(query.to_string())
            .await?;

        Ok(())
    }

    fn rollback_instructions(&self) -> String {
        "/* Drop the table */\
            DROP TABLE IF EXISTS ModelInferenceCache;
            /* Drop the `cached` column from ModelInference */\
            ALTER TABLE ModelInference DROP COLUMN cached;
            "
        .to_string()
    }

    /// Check if the migration has succeeded (i.e. it should not be applied again)
    async fn has_succeeded(&self) -> Result<bool, Error> {
        let should_apply = self.should_apply().await?;
        Ok(!should_apply)
    }
}
