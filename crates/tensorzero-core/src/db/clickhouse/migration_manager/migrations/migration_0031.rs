use crate::db::clickhouse::ClickHouseConnectionInfo;
use crate::db::clickhouse::migration_manager::migration_trait::Migration;
use crate::error::delayed_error::DelayedError;
use async_trait::async_trait;

use super::check_column_exists;

/// This migration adds a `ttft_ms` column to the `ChatInference` and `JsonInference` tables.
pub struct Migration0031<'a> {
    pub clickhouse: &'a ClickHouseConnectionInfo,
}

const MIGRATION_ID: &str = "0031";

#[async_trait]
impl Migration for Migration0031<'_> {
    async fn can_apply(&self) -> Result<(), DelayedError> {
        Ok(())
    }

    async fn should_apply(&self) -> Result<bool, DelayedError> {
        let chat_ttft_ms_column_exists =
            check_column_exists(self.clickhouse, "ChatInference", "ttft_ms", MIGRATION_ID).await?;
        let json_ttft_ms_column_exists =
            check_column_exists(self.clickhouse, "JsonInference", "ttft_ms", MIGRATION_ID).await?;
        // We need to run this migration if either column is missing
        Ok(!chat_ttft_ms_column_exists || !json_ttft_ms_column_exists)
    }

    async fn apply(&self, _clean_start: bool) -> Result<(), DelayedError> {
        self.clickhouse
            .run_query_synchronous_no_params_delayed_err(
                r"ALTER TABLE ChatInference ADD COLUMN IF NOT EXISTS ttft_ms Nullable(UInt32);"
                    .to_string(),
            )
            .await?;
        self.clickhouse
            .run_query_synchronous_no_params_delayed_err(
                r"ALTER TABLE JsonInference ADD COLUMN IF NOT EXISTS ttft_ms Nullable(UInt32);"
                    .to_string(),
            )
            .await?;

        Ok(())
    }

    fn rollback_instructions(&self) -> String {
        r"ALTER TABLE ChatInference DROP COLUMN ttft_ms;
        ALTER TABLE JsonInference DROP COLUMN ttft_ms;"
            .to_string()
    }

    async fn has_succeeded(&self) -> Result<bool, DelayedError> {
        let should_apply = self.should_apply().await?;
        Ok(!should_apply)
    }
}
