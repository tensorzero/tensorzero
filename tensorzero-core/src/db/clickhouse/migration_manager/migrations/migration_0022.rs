use crate::db::clickhouse::migration_manager::migration_trait::Migration;
use crate::db::clickhouse::ClickHouseConnectionInfo;
use crate::error::{Error, ErrorDetails};
use async_trait::async_trait;

use super::{check_column_exists, check_table_exists};

/// This migration adds a column to the `ChatInferenceDatapoint` and `JsonInferenceDatapoint` tables to store the
/// `source_inference_id` of the datapoint.
/// This is a Nullable(UUID) in both cases.
pub struct Migration0022<'a> {
    pub clickhouse: &'a ClickHouseConnectionInfo,
}

#[async_trait]
impl Migration for Migration0022<'_> {
    async fn can_apply(&self) -> Result<(), Error> {
        let chat_inference_datapoint_table_exists =
            check_table_exists(self.clickhouse, "ChatInferenceDatapoint", "0022").await?;
        if !chat_inference_datapoint_table_exists {
            return Err(Error::new(ErrorDetails::ClickHouseMigration {
                id: "0022".to_string(),
                message: "ChatInferenceDatapoint table does not exist".to_string(),
            }));
        }
        let json_inference_datapoint_table_exists =
            check_table_exists(self.clickhouse, "JsonInferenceDatapoint", "0022").await?;
        if !json_inference_datapoint_table_exists {
            return Err(Error::new(ErrorDetails::ClickHouseMigration {
                id: "0022".to_string(),
                message: "JsonInferenceDatapoint table does not exist".to_string(),
            }));
        }

        Ok(())
    }

    async fn should_apply(&self) -> Result<bool, Error> {
        let chat_source_inference_id_column_exists = check_column_exists(
            self.clickhouse,
            "ChatInferenceDatapoint",
            "source_inference_id",
            "0022",
        )
        .await?;
        let json_source_inference_id_column_exists = check_column_exists(
            self.clickhouse,
            "JsonInferenceDatapoint",
            "source_inference_id",
            "0022",
        )
        .await?;

        Ok(!chat_source_inference_id_column_exists || !json_source_inference_id_column_exists)
    }

    async fn apply(&self, _clean_start: bool) -> Result<(), Error> {
        let on_cluster_name = self.clickhouse.get_on_cluster_name();
        self.clickhouse
            .run_query_synchronous_no_params(
                format!("ALTER TABLE ChatInferenceDatapoint{on_cluster_name} ADD COLUMN IF NOT EXISTS source_inference_id Nullable(UUID)"),
            )
            .await?;

        self.clickhouse
            .run_query_synchronous_no_params(
                format!("ALTER TABLE JsonInferenceDatapoint{on_cluster_name} ADD COLUMN IF NOT EXISTS source_inference_id Nullable(UUID)"),
            )
            .await?;

        Ok(())
    }

    fn rollback_instructions(&self) -> String {
        let on_cluster_name = self.clickhouse.get_on_cluster_name();
        format!("ALTER TABLE ChatInferenceDatapoint{on_cluster_name} DROP COLUMN source_inference_id")
    }

    async fn has_succeeded(&self) -> Result<bool, Error> {
        let should_apply = self.should_apply().await?;
        Ok(!should_apply)
    }
}
