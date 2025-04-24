use crate::clickhouse::migration_manager::migration_trait::Migration;
use crate::clickhouse::ClickHouseConnectionInfo;
use crate::error::Error;
use async_trait::async_trait;

use super::check_column_exists;

/// This migration adds an `extra_headers` column to the `ChatInference`/`JsonInference` tables.
pub struct Migration0025<'a> {
    pub clickhouse: &'a ClickHouseConnectionInfo,
}

#[async_trait]
impl Migration for Migration0025<'_> {
    async fn can_apply(&self) -> Result<(), Error> {
        Ok(())
    }

    async fn should_apply(&self) -> Result<bool, Error> {
        let chat_extra_headers_column_exists =
            check_column_exists(self.clickhouse, "ChatInference", "extra_headers", "0025").await?;
        let json_extra_headers_column_exists =
            check_column_exists(self.clickhouse, "JsonInference", "extra_headers", "0025").await?;

        Ok(!chat_extra_headers_column_exists || !json_extra_headers_column_exists)
    }

    async fn apply(&self) -> Result<(), Error> {
        self.clickhouse
            .run_query_synchronous(
                "ALTER TABLE ChatInference ADD COLUMN IF NOT EXISTS extra_headers Nullable(String)"
                    .to_string(),
                None,
            )
            .await?;

        self.clickhouse
            .run_query_synchronous(
                "ALTER TABLE JsonInference ADD COLUMN IF NOT EXISTS extra_headers Nullable(String)"
                    .to_string(),
                None,
            )
            .await?;

        Ok(())
    }

    fn rollback_instructions(&self) -> String {
        "-- Drop the columns\n\
                ALTER TABLE ChatInference DROP COLUMN IF EXISTS extra_headers;\n\
                ALTER TABLE JsonInference DROP COLUMN IF EXISTS extra_headers;\n\
                "
        .to_string()
    }

    async fn has_succeeded(&self) -> Result<bool, Error> {
        let should_apply = self.should_apply().await?;
        Ok(!should_apply)
    }
}
