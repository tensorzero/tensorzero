use crate::clickhouse::migration_manager::migration_trait::Migration;
use crate::clickhouse::migration_manager::migrations::create_cluster_clause;
use crate::clickhouse::ClickHouseConnectionInfo;
use crate::config_parser::Config;
use crate::error::Error;
use async_trait::async_trait;

use super::check_column_exists;

/// This migration adds a `ttft_ms` column to the `ChatInference` and `JsonInference` tables.
pub struct Migration0031<'a> {
    pub clickhouse: &'a ClickHouseConnectionInfo,
    pub config: &'a Config,
}

const MIGRATION_ID: &str = "0031";

#[async_trait]
impl Migration for Migration0031<'_> {
    async fn can_apply(&self) -> Result<(), Error> {
        Ok(())
    }

    async fn should_apply(&self) -> Result<bool, Error> {
        let chat_ttft_ms_column_exists =
            check_column_exists(self.clickhouse, "ChatInference", "ttft_ms", MIGRATION_ID).await?;
        let json_ttft_ms_column_exists =
            check_column_exists(self.clickhouse, "JsonInference", "ttft_ms", MIGRATION_ID).await?;
        // We need to run this migration if either column is missing
        Ok(!chat_ttft_ms_column_exists || !json_ttft_ms_column_exists)
    }

    async fn apply(&self, _clean_start: bool) -> Result<(), Error> {
        let cluster_clause = create_cluster_clause(
            self.config.clickhouse.replication_enabled, 
            &self.config.clickhouse.cluster_name
        );
        
        self.clickhouse
            .run_query_synchronous_no_params(
                format!("ALTER TABLE ChatInference {cluster_clause} ADD COLUMN IF NOT EXISTS ttft_ms Nullable(UInt32);")
            )
            .await?;
        self.clickhouse
            .run_query_synchronous_no_params(
                format!("ALTER TABLE JsonInference {cluster_clause} ADD COLUMN IF NOT EXISTS ttft_ms Nullable(UInt32);")
            )
            .await?;

        Ok(())
    }

    fn rollback_instructions(&self) -> String {
        let cluster_clause = create_cluster_clause(
            self.config.clickhouse.replication_enabled, 
            &self.config.clickhouse.cluster_name
        );
        format!(
            "ALTER TABLE ChatInference {cluster_clause} DROP COLUMN ttft_ms;
        ALTER TABLE JsonInference {cluster_clause} DROP COLUMN ttft_ms;"
        )
    }

    async fn has_succeeded(&self) -> Result<bool, Error> {
        let should_apply = self.should_apply().await?;
        Ok(!should_apply)
    }
}
