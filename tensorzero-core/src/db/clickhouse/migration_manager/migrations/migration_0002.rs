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
        // Create the `DynamicInContextLearningExample` table
        let table_engine_name = self.clickhouse.get_maybe_replicated_table_engine_name(
            GetMaybeReplicatedTableEngineNameArgs {
                table_engine_name: "MergeTree",
                table_name: "DynamicInContextLearningExample",
                engine_args: &[],
            },
        );
        let on_cluster_name = self.clickhouse.get_on_cluster_name();
        let query = format!(
            r"
            CREATE TABLE IF NOT EXISTS DynamicInContextLearningExample{on_cluster_name}
            (
                id UUID, -- must be a UUIDv7
                function_name LowCardinality(String),
                variant_name LowCardinality(String),
                namespace String,
                input String,
                output String,
                embedding Array(Float32),
                timestamp DateTime MATERIALIZED UUIDv7ToDateTime(id)
            ) ENGINE = {table_engine_name}
            ORDER BY (function_name, variant_name, namespace);
        ",
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
            "/* Drop the table */\
            DROP TABLE IF EXISTS DynamicInContextLearningExample{on_cluster_name} SYNC;"
        )
    }

    /// Check if the migration has succeeded (i.e. it should not be applied again)
    async fn has_succeeded(&self) -> Result<bool, Error> {
        let should_apply = self.should_apply().await?;
        Ok(!should_apply)
    }
}
