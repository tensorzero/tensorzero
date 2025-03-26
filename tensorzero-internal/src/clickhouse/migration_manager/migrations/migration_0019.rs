use crate::clickhouse::migration_manager::migration_trait::Migration;
use crate::clickhouse::ClickHouseConnectionInfo;
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

    async fn apply(&self) -> Result<(), Error> {
        self.clickhouse
            .run_query(
                "ALTER TABLE ChatInference ADD COLUMN IF NOT EXISTS extra_body Nullable(String)"
                    .to_string(),
                None,
            )
            .await?;

        self.clickhouse
            .run_query(
                "ALTER TABLE JsonInference ADD COLUMN IF NOT EXISTS extra_body Nullable(String)"
                    .to_string(),
                None,
            )
            .await?;

        Ok(())
    }

    fn rollback_instructions(&self) -> String {
        "-- Drop the columns\n\
                ALTER TABLE ChatInference DROP COLUMN IF EXISTS extra_body;\n\
                ALTER TABLE JsonInference DROP COLUMN IF EXISTS extra_body;\n\
                "
        .to_string()
    }

    async fn has_succeeded(&self) -> Result<bool, Error> {
        let should_apply = self.should_apply().await?;
        Ok(!should_apply)
    }
}
