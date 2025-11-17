use crate::db::clickhouse::migration_manager::migration_trait::Migration;
use crate::db::clickhouse::ClickHouseConnectionInfo;
use crate::error::Error;
use async_trait::async_trait;

use super::get_column_type;

/// This migration makes the `input_tokens` and `output_tokens` columns nullable
/// in the `ModelInferenceCache` table to match the Rust type definition (Option<u32>)
/// and handle cases where providers don't return usage information.
pub struct Migration0042<'a> {
    pub clickhouse: &'a ClickHouseConnectionInfo,
}

const MIGRATION_ID: &str = "0042";

#[async_trait]
impl Migration for Migration0042<'_> {
    async fn can_apply(&self) -> Result<(), Error> {
        Ok(())
    }

    async fn should_apply(&self) -> Result<bool, Error> {
        if get_column_type(
            self.clickhouse,
            "ModelInferenceCache",
            "input_tokens",
            MIGRATION_ID,
        )
        .await?
            != "Nullable(UInt32)"
        {
            return Ok(true);
        }

        if get_column_type(
            self.clickhouse,
            "ModelInferenceCache",
            "output_tokens",
            MIGRATION_ID,
        )
        .await?
            != "Nullable(UInt32)"
        {
            return Ok(true);
        }

        Ok(false)
    }

    async fn apply(&self, _clean_start: bool) -> Result<(), Error> {
        self.clickhouse
            .run_query_synchronous_no_params(
                r"ALTER TABLE ModelInferenceCache
                MODIFY COLUMN input_tokens Nullable(UInt32)"
                    .to_string(),
            )
            .await?;

        self.clickhouse
            .run_query_synchronous_no_params(
                r"ALTER TABLE ModelInferenceCache
                MODIFY COLUMN output_tokens Nullable(UInt32)"
                    .to_string(),
            )
            .await?;

        Ok(())
    }

    fn rollback_instructions(&self) -> String {
        r"ALTER TABLE ModelInferenceCache MODIFY COLUMN input_tokens UInt32;
          ALTER TABLE ModelInferenceCache MODIFY COLUMN output_tokens UInt32;"
            .to_string()
    }

    async fn has_succeeded(&self) -> Result<bool, Error> {
        let should_apply = self.should_apply().await?;
        Ok(!should_apply)
    }
}
