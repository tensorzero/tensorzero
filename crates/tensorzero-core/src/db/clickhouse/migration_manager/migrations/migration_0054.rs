use super::check_column_exists;
use super::check_table_exists;
use crate::db::clickhouse::ClickHouseConnectionInfo;
use crate::db::clickhouse::migration_manager::migration_trait::Migration;
use crate::error::{ErrorDetails, delayed_error::DelayedError};
use async_trait::async_trait;

/// Adds OTel GenAI fields to the `ModelInference` table:
///
/// - `provider_response_id`: provider-native response id (e.g. OpenAI `chatcmpl-…`)
/// - `response_model_name`: actual model returned by the provider
/// - `operation`: OTel `gen_ai.operation.name` ("chat", "embeddings", etc.)
///
/// No materialized view changes are required — none of the existing MVs
/// select these columns.
pub struct Migration0054<'a> {
    pub clickhouse: &'a ClickHouseConnectionInfo,
}

const MIGRATION_ID: &str = "0054";

#[async_trait]
impl Migration for Migration0054<'_> {
    async fn can_apply(&self) -> Result<(), DelayedError> {
        if !check_table_exists(self.clickhouse, "ModelInference", MIGRATION_ID).await? {
            return Err(DelayedError::new(ErrorDetails::ClickHouseMigration {
                id: MIGRATION_ID.to_string(),
                message: "`ModelInference` table does not exist".to_string(),
            }));
        }
        Ok(())
    }

    async fn should_apply(&self) -> Result<bool, DelayedError> {
        // Apply if any of the three columns is missing.
        for col in ["provider_response_id", "response_model_name", "operation"] {
            if !check_column_exists(self.clickhouse, "ModelInference", col, MIGRATION_ID).await? {
                return Ok(true);
            }
        }
        Ok(false)
    }

    async fn apply(&self, _clean_start: bool) -> Result<(), DelayedError> {
        let on_cluster_name = self.clickhouse.get_on_cluster_name();

        self.clickhouse
            .run_query_synchronous_no_params_delayed_err(format!(
                "ALTER TABLE ModelInference{on_cluster_name} ADD COLUMN IF NOT EXISTS provider_response_id Nullable(String) DEFAULT NULL"
            ))
            .await?;

        self.clickhouse
            .run_query_synchronous_no_params_delayed_err(format!(
                "ALTER TABLE ModelInference{on_cluster_name} ADD COLUMN IF NOT EXISTS response_model_name LowCardinality(Nullable(String)) DEFAULT NULL"
            ))
            .await?;

        self.clickhouse
            .run_query_synchronous_no_params_delayed_err(format!(
                "ALTER TABLE ModelInference{on_cluster_name} ADD COLUMN IF NOT EXISTS operation LowCardinality(Nullable(String)) DEFAULT NULL"
            ))
            .await?;

        Ok(())
    }

    fn rollback_instructions(&self) -> String {
        let on_cluster_name = self.clickhouse.get_on_cluster_name();
        format!(
            r"
            ALTER TABLE ModelInference{on_cluster_name} DROP COLUMN IF EXISTS provider_response_id;
            ALTER TABLE ModelInference{on_cluster_name} DROP COLUMN IF EXISTS response_model_name;
            ALTER TABLE ModelInference{on_cluster_name} DROP COLUMN IF EXISTS operation;"
        )
    }

    async fn has_succeeded(&self) -> Result<bool, DelayedError> {
        let should_apply = self.should_apply().await?;
        Ok(!should_apply)
    }
}
