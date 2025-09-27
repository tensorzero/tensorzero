use crate::db::clickhouse::migration_manager::migration_trait::Migration;
use crate::db::clickhouse::ClickHouseConnectionInfo;
use crate::error::Error;
use async_trait::async_trait;

use super::check_column_exists;

/// This migration adds an `extra_body` column to the  `ChatInference`/`JsonInference` tables.
pub struct Migration0019<'a> {
    pub clickhouse: &'a ClickHouseConnectionInfo,
}

#[async_trait]
impl Migration for Migration0019<'_> {
    async fn can_apply(&self) -> Result<(), Error> {
        Ok(())
    }

    async fn should_apply(&self) -> Result<bool, Error> {
        let chat_extra_body_column_exists =
            check_column_exists(self.clickhouse, "ChatInference", "extra_body", "0019").await?;
        let json_extra_body_column_exists =
            check_column_exists(self.clickhouse, "JsonInference", "extra_body", "0019").await?;

        Ok(!chat_extra_body_column_exists || !json_extra_body_column_exists)
    }

    async fn apply(&self, _clean_start: bool) -> Result<(), Error> {
        // Add extra_body column to both ChatInference and JsonInference using sharding-aware ALTER
        self.clickhouse
            .get_alter_table_statements(
                "ChatInference",
                "ADD COLUMN IF NOT EXISTS extra_body Nullable(String)",
                false,
            )
            .await?;

        self.clickhouse
            .get_alter_table_statements(
                "JsonInference",
                "ADD COLUMN IF NOT EXISTS extra_body Nullable(String)",
                false,
            )
            .await?;

        Ok(())
    }

    fn rollback_instructions(&self) -> String {
        format!(
            "/* Drop the columns */\
            {}\
            {}",
            self.clickhouse.get_alter_table_rollback_statements("ChatInference", "DROP COLUMN IF EXISTS extra_body", false),
            self.clickhouse.get_alter_table_rollback_statements("JsonInference", "DROP COLUMN IF EXISTS extra_body", false)
        )
    }

    async fn has_succeeded(&self) -> Result<bool, Error> {
        let should_apply = self.should_apply().await?;
        Ok(!should_apply)
    }
}
