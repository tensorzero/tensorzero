use crate::db::clickhouse::ClickHouseConnectionInfo;
use crate::db::clickhouse::migration_manager::migration_trait::Migration;
use crate::error::{ErrorDetails, delayed_error::DelayedError};
use async_trait::async_trait;

use super::{check_column_exists, check_table_exists};

/// This migration adds a column to the `ModelInference` table to store the
/// finish_reason field from the model inference.
/// It is an enum Nullable(Enum8('stop', 'length', 'tool_call', 'content_filter', 'unknown'))
///
/// We also add the same column to the ModelInferenceCache table so that we can restore it from cache.
pub struct Migration0018<'a> {
    pub clickhouse: &'a ClickHouseConnectionInfo,
}

#[async_trait]
impl Migration for Migration0018<'_> {
    async fn can_apply(&self) -> Result<(), DelayedError> {
        let model_inference_table_exists =
            check_table_exists(self.clickhouse, "ModelInference", "0018").await?;
        if !model_inference_table_exists {
            return Err(DelayedError::new(ErrorDetails::ClickHouseMigration {
                id: "0018".to_string(),
                message: "ModelInference table does not exist".to_string(),
            }));
        }
        let cache_table_exists =
            check_table_exists(self.clickhouse, "ModelInferenceCache", "0018").await?;
        if !cache_table_exists {
            return Err(DelayedError::new(ErrorDetails::ClickHouseMigration {
                id: "0018".to_string(),
                message: "ModelInferenceCache table does not exist".to_string(),
            }));
        }

        Ok(())
    }

    async fn should_apply(&self) -> Result<bool, DelayedError> {
        let finish_reason_column_exists =
            check_column_exists(self.clickhouse, "ModelInference", "finish_reason", "0018").await?;
        let cache_finish_reason_column_exists = check_column_exists(
            self.clickhouse,
            "ModelInferenceCache",
            "finish_reason",
            "0018",
        )
        .await?;

        Ok(!finish_reason_column_exists || !cache_finish_reason_column_exists)
    }

    async fn apply(&self, _clean_start: bool) -> Result<(), DelayedError> {
        self.clickhouse
            .run_query_synchronous_no_params_delayed_err(
                "ALTER TABLE ModelInference ADD COLUMN IF NOT EXISTS finish_reason Nullable(Enum8('stop', 'length', 'tool_call', 'content_filter', 'unknown'))".to_string(),
            )
            .await?;

        self.clickhouse
            .run_query_synchronous_no_params_delayed_err(
                "ALTER TABLE ModelInferenceCache ADD COLUMN IF NOT EXISTS finish_reason Nullable(Enum8('stop', 'length', 'tool_call', 'content_filter', 'unknown'))".to_string(),
            )
            .await?;

        Ok(())
    }

    fn rollback_instructions(&self) -> String {
        "ALTER TABLE ModelInference DROP COLUMN finish_reason".to_string()
    }

    async fn has_succeeded(&self) -> Result<bool, DelayedError> {
        let should_apply = self.should_apply().await?;
        Ok(!should_apply)
    }
}
