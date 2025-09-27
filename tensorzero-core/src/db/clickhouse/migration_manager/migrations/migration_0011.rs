use crate::db::clickhouse::migration_manager::migration_trait::Migration;
use crate::db::clickhouse::{ClickHouseConnectionInfo, GetMaybeReplicatedTableEngineNameArgs};
use crate::error::Error;
use async_trait::async_trait;

use super::{check_column_exists, check_table_exists};

/// This migration is used to set up the ClickHouse database for caching.
/// We create a table `ModelInferenceCache` that stores the `short_cache_key`, `long_cache_key`, `timestamp`, and `output`
/// The `short_cache_key` is the first 8 bytes of the `long_cache_key`, a 32 byte array. We use it as the primary key in clickhouse for fast lookups.
/// The `long_cache_key` is the cache key for the model provider request
/// The `timestamp` is the timestamp of the request
/// The `output` is the output of the request, serialized using serde_json::to_string
/// We also add a column `cached` to indicate if a ModelInference was a cache hit
pub struct Migration0011<'a> {
    pub clickhouse: &'a ClickHouseConnectionInfo,
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
        // Create the `ModelInferenceCache` table with sharding support
        let table_schema = r"
            (
                short_cache_key UInt64,
                long_cache_key FixedString(64), -- for a hex-encoded 256-bit key
                timestamp DateTime DEFAULT now(),
                output String,
                raw_request String,
                raw_response String,
                is_deleted Bool DEFAULT false,
                INDEX idx_long_cache_key long_cache_key TYPE bloom_filter GRANULARITY 100
            )";

        let table_engine_args = GetMaybeReplicatedTableEngineNameArgs {
            table_engine_name: "ReplacingMergeTree",
            table_name: "ModelInferenceCache",
            engine_args: &["timestamp", "is_deleted"],
        };

        self.clickhouse.get_create_table_statements(
            "ModelInferenceCache",
            table_schema,
            &table_engine_args,
            Some("ORDER BY (short_cache_key, long_cache_key) PARTITION BY toYYYYMM(timestamp) SETTINGS index_granularity = 256"),
        ).await?;

        // Add the `cached` column to ModelInference using sharding-aware ALTER
        self.clickhouse
            .get_alter_table_statements(
                "ModelInference",
                "ADD COLUMN IF NOT EXISTS cached Bool DEFAULT false",
                false,
            )
            .await?;

        Ok(())
    }

    fn rollback_instructions(&self) -> String {
        let model_cache_rollback = self.clickhouse.get_drop_table_rollback_statements("ModelInferenceCache");
        format!(
            "{model_cache_rollback}\
            /* Drop the `cached` column from ModelInference */\
            {}",
            self.clickhouse.get_alter_table_rollback_statements("ModelInference", "DROP COLUMN cached", false)
        )
    }

    /// Check if the migration has succeeded (i.e. it should not be applied again)
    async fn has_succeeded(&self) -> Result<bool, Error> {
        let should_apply = self.should_apply().await?;
        Ok(!should_apply)
    }
}
