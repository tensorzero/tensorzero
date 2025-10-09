use super::{check_column_exists, check_table_exists};
use crate::db::clickhouse::migration_manager::migration_trait::Migration;
use crate::db::clickhouse::ClickHouseConnectionInfo;
use crate::error::{Error, ErrorDetails};
use async_trait::async_trait;

/// Adds an optional `name` column to datapoint tables for a human-readable label/name. This doesn't have to be
/// unique across datapoints, and can show up in the UI.
pub struct Migration0040<'a> {
    pub clickhouse: &'a ClickHouseConnectionInfo,
}

const MIGRATION_ID: &str = "0040";

#[async_trait]
impl Migration for Migration0040<'_> {
    async fn can_apply(&self) -> Result<(), Error> {
        let tables_to_check = ["ChatInferenceDatapoint", "JsonInferenceDatapoint"];

        for table_name in tables_to_check {
            if !check_table_exists(self.clickhouse, table_name, MIGRATION_ID).await? {
                return Err(Error::new(ErrorDetails::ClickHouseMigration {
                    id: MIGRATION_ID.to_string(),
                    message: format!("{table_name} table does not exist"),
                }));
            }
        }
        Ok(())
    }

    async fn should_apply(&self) -> Result<bool, Error> {
        let chat_inference_datapoint_has_name = check_column_exists(
            self.clickhouse,
            "ChatInferenceDatapoint",
            "name",
            MIGRATION_ID,
        )
        .await?;
        let json_inference_datapoint_has_name = check_column_exists(
            self.clickhouse,
            "JsonInferenceDatapoint",
            "name",
            MIGRATION_ID,
        )
        .await?;

        Ok(!chat_inference_datapoint_has_name || !json_inference_datapoint_has_name)
    }

    async fn apply(&self, _clean_start: bool) -> Result<(), Error> {
        self.clickhouse
            .run_query_synchronous_no_params(
                r"ALTER TABLE ChatInferenceDatapoint ADD COLUMN IF NOT EXISTS name Nullable(String);"
                    .to_string(),
            )
            .await?;
        self.clickhouse
            .run_query_synchronous_no_params(
                r"ALTER TABLE JsonInferenceDatapoint ADD COLUMN IF NOT EXISTS name Nullable(String);"
                    .to_string(),
            )
            .await?;

        Ok(())
    }

    fn rollback_instructions(&self) -> String {
        r"ALTER TABLE ChatInferenceDatapoint DROP COLUMN name;
        ALTER TABLE JsonInferenceDatapoint DROP COLUMN name;"
            .to_string()
    }

    async fn has_succeeded(&self) -> Result<bool, Error> {
        let should_apply = self.should_apply().await?;
        Ok(!should_apply)
    }
}
