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
        let on_cluster_name = self.clickhouse.get_on_cluster_name();
        self.clickhouse
            .run_query_synchronous_no_params(
                format!(r"ALTER TABLE ChatInferenceDatapoint{on_cluster_name} ADD COLUMN IF NOT EXISTS is_custom Bool DEFAULT false;"),
            )
            .await?;
        self.clickhouse
            .run_query_synchronous_no_params(
                format!(r"ALTER TABLE JsonInferenceDatapoint{on_cluster_name} ADD COLUMN IF NOT EXISTS is_custom Bool DEFAULT false;"),
            )
            .await?;

        Ok(())
    }

    fn rollback_instructions(&self) -> String {
        let on_cluster_name = self.clickhouse.get_on_cluster_name();
        format!(r"ALTER TABLE ChatInferenceDatapoint{on_cluster_name} DROP COLUMN is_custom;
        ALTER TABLE JsonInferenceDatapoint{on_cluster_name} DROP COLUMN is_custom;")
    }

    async fn has_succeeded(&self) -> Result<bool, Error> {
        let should_apply = self.should_apply().await?;
        Ok(!should_apply)
    }
}
