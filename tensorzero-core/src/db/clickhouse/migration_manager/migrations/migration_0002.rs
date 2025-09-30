use async_trait::async_trait;

use crate::db::clickhouse::migration_manager::migration_trait::Migration;
use crate::db::clickhouse::{ClickHouseConnectionInfo, GetMaybeReplicatedTableEngineNameArgs};
use crate::error::Error;

use super::check_table_exists;

/// This migration is used to set up the ClickHouse database to store examples
/// for dynamic in-context learning.
pub struct Migration0002<'a> {
    pub clickhouse: &'a ClickHouseConnectionInfo,
}

#[async_trait]
impl Migration for Migration0002<'_> {
    async fn can_apply(&self) -> Result<(), Error> {
        Ok(())
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

    async fn apply(&self, _clean_start: bool) -> Result<(), Error> {
        // Create the `DynamicInContextLearningExample` table with sharding support
        let table_schema = r"
            (
                id UUID, -- must be a UUIDv7
                function_name LowCardinality(String),
                variant_name LowCardinality(String),
                namespace String,
                input String,
                output String,
                embedding Array(Float32),
                timestamp DateTime MATERIALIZED UUIDv7ToDateTime(id)
            )";

        let table_engine_args = GetMaybeReplicatedTableEngineNameArgs {
            table_engine_name: "MergeTree",
            table_name: "DynamicInContextLearningExample",
            engine_args: &[],
        };

        self.clickhouse.get_create_table_statements(
            "DynamicInContextLearningExample",
            table_schema,
            &table_engine_args,
            Some("ORDER BY (function_name, variant_name, namespace)"),
            Some("cityHash64(function_name)"),
        ).await?;

        Ok(())
    }

    fn rollback_instructions(&self) -> String {
        self.clickhouse.get_drop_table_rollback_statements("DynamicInContextLearningExample")
    }

    /// Check if the migration has succeeded (i.e. it should not be applied again)
    async fn has_succeeded(&self) -> Result<bool, Error> {
        let should_apply = self.should_apply().await?;
        Ok(!should_apply)
    }
}
