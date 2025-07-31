use crate::clickhouse::migration_manager::migration_trait::Migration;
use crate::clickhouse::migration_manager::migrations::create_cluster_clause;
use crate::clickhouse::ClickHouseConnectionInfo;
use crate::config_parser::Config;
use crate::error::Error;
use async_trait::async_trait;

use super::get_column_type;

/// This migration is used to mark the `input_tokens` and `output_tokens` columns as nullable in the `ModelInference` table.
pub struct Migration0015<'a> {
    pub clickhouse: &'a ClickHouseConnectionInfo,
    pub config: &'a Config,
}

#[async_trait]
impl Migration for Migration0015<'_> {
    async fn can_apply(&self) -> Result<(), Error> {
        Ok(())
    }

    /// Check if the migration has already been applied by checking if
    /// the `input_tokens` and `output_tokens` columns are nullable
    async fn should_apply(&self) -> Result<bool, Error> {
        if get_column_type(self.clickhouse, "ModelInference", "input_tokens", "0015").await?
            != "Nullable(UInt32)"
        {
            return Ok(true);
        }

        if get_column_type(self.clickhouse, "ModelInference", "output_tokens", "0015").await?
            != "Nullable(UInt32)"
        {
            return Ok(true);
        }

        // Everything is in place, so we should not apply the migration
        Ok(false)
    }

    async fn apply(&self, _clean_start: bool) -> Result<(), Error> {
        let cluster_clause = create_cluster_clause(
            self.config.clickhouse.replication_enabled, 
            &self.config.clickhouse.cluster_name
        );
        
        // Alter the `input_tokens` column of `ModelInference` to be a nullable column
        let query = format!(
            "ALTER TABLE ModelInference {cluster_clause}
            MODIFY COLUMN input_tokens Nullable(UInt32)"
        );
        let _ = self
            .clickhouse
            .run_query_synchronous_no_params(query)
            .await?;

        // Alter the `output_tokens` column of `ModelInference` to be a nullable column
        let query = format!(
            "ALTER TABLE ModelInference {cluster_clause}
            MODIFY COLUMN output_tokens Nullable(UInt32)"
        );
        let _ = self
            .clickhouse
            .run_query_synchronous_no_params(query)
            .await?;

        Ok(())
    }

    fn rollback_instructions(&self) -> String {
        let cluster_clause = create_cluster_clause(
            self.config.clickhouse.replication_enabled, 
            &self.config.clickhouse.cluster_name
        );
        format!(
            "/* Change the columns back to non-nullable types */\
            ALTER TABLE ModelInference {cluster_clause} MODIFY COLUMN input_tokens UInt32;
            ALTER TABLE ModelInference {cluster_clause} MODIFY COLUMN output_tokens UInt32;"
        )
    }

    /// Check if the migration has succeeded (i.e. it should not be applied again)
    async fn has_succeeded(&self) -> Result<bool, Error> {
        let should_apply = self.should_apply().await?;
        Ok(!should_apply)
    }
}
