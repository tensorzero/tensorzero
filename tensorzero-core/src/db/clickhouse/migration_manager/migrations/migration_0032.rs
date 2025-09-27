use crate::db::clickhouse::migration_manager::migration_trait::Migration;
use crate::db::clickhouse::ClickHouseConnectionInfo;
use crate::error::Error;
use async_trait::async_trait;

use super::check_column_exists;

/// This migration adds a boolean column `is_custom` to the `ChatInferenceDatapoint` and `JsonInferenceDatapoint` tables,
/// denoting whether the datapoint has been customized by the user beyond the choice of output taken from
/// the historical inference, a demonstration, or none.
///
/// This is used to determine whether the datapoint is a custom datapoint or not for the purposes of deduplication.
pub struct Migration0032<'a> {
    pub clickhouse: &'a ClickHouseConnectionInfo,
}

const MIGRATION_ID: &str = "0032";

#[async_trait]
impl Migration for Migration0032<'_> {
    async fn can_apply(&self) -> Result<(), Error> {
        Ok(())
    }

    async fn should_apply(&self) -> Result<bool, Error> {
        let chat_is_custom_column_exists = check_column_exists(
            self.clickhouse,
            "ChatInferenceDatapoint",
            "is_custom",
            MIGRATION_ID,
        )
        .await?;
        let json_is_custom_column_exists = check_column_exists(
            self.clickhouse,
            "JsonInferenceDatapoint",
            "is_custom",
            MIGRATION_ID,
        )
        .await?;
        // We need to run this migration if either column is missing
        Ok(!chat_is_custom_column_exists || !json_is_custom_column_exists)
    }

    async fn apply(&self, _clean_start: bool) -> Result<(), Error> {
        // Add is_custom column to both ChatInferenceDatapoint and JsonInferenceDatapoint using sharding-aware ALTER
        self.clickhouse
            .get_alter_table_statements(
                "ChatInferenceDatapoint",
                "ADD COLUMN IF NOT EXISTS is_custom Bool DEFAULT false",
                false,
            )
            .await?;

        self.clickhouse
            .get_alter_table_statements(
                "JsonInferenceDatapoint",
                "ADD COLUMN IF NOT EXISTS is_custom Bool DEFAULT false",
                false,
            )
            .await?;

        Ok(())
    }

    fn rollback_instructions(&self) -> String {
        format!(
            "{}\
            {}",
            self.clickhouse.get_alter_table_rollback_statements("ChatInferenceDatapoint", "DROP COLUMN is_custom", false),
            self.clickhouse.get_alter_table_rollback_statements("JsonInferenceDatapoint", "DROP COLUMN is_custom", false)
        )
    }

    async fn has_succeeded(&self) -> Result<bool, Error> {
        let should_apply = self.should_apply().await?;
        Ok(!should_apply)
    }
}
