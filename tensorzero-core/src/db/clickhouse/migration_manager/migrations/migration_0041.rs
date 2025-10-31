use super::{check_column_exists, check_table_exists};
use crate::db::clickhouse::migration_manager::migration_trait::Migration;
use crate::db::clickhouse::ClickHouseConnectionInfo;
use crate::error::{Error, ErrorDetails};
use async_trait::async_trait;

/// Adds columns `dynamic_tools`, `dynamic_provider_tools`, `allowed_tools`, `tool_choice`, `parallel_tool_calls`
/// to ChatInference, ChatInferenceDatapoint, and BatchModelInference so that we can store more comprehensive info about
/// what happened with tool configuration.
pub struct Migration0041<'a> {
    pub clickhouse: &'a ClickHouseConnectionInfo,
}

const MIGRATION_ID: &str = "0041";

#[async_trait]
impl Migration for Migration0041<'_> {
    async fn can_apply(&self) -> Result<(), Error> {
        let tables_to_check = [
            "ChatInferenceDatapoint",
            "ChatInference",
            "BatchModelInference",
        ];

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
        let tables_to_check = [
            "ChatInferenceDatapoint",
            "ChatInference",
            "BatchModelInference",
        ];
        let columns_to_add = [
            "dynamic_tools",
            "dynamic_provider_tools",
            "allowed_tools",
            "tool_choice",
            "parallel_tool_calls",
        ];
        for table in &tables_to_check {
            for column in &columns_to_add {
                if !check_column_exists(self.clickhouse, table, column, MIGRATION_ID).await? {
                    return Ok(true);
                }
            }
        }
        Ok(false)
    }

    async fn apply(&self, _clean_start: bool) -> Result<(), Error> {
        self.clickhouse
            .run_query_synchronous_no_params(
                r"ALTER TABLE ChatInference ADD COLUMN IF NOT EXISTS dynamic_tools Array(String)"
                    .to_string(),
            )
            .await?;
        self.clickhouse
            .run_query_synchronous_no_params(
                r"ALTER TABLE ChatInference ADD COLUMN IF NOT EXISTS dynamic_provider_tools Array(String)"
                    .to_string(),
            )
            .await?;
        self.clickhouse
            .run_query_synchronous_no_params(
                r"ALTER TABLE ChatInference ADD COLUMN IF NOT EXISTS allowed_tools Nullable(String)"
                    .to_string(),
            )
            .await?;
        self.clickhouse
            .run_query_synchronous_no_params(
                r"ALTER TABLE ChatInference ADD COLUMN IF NOT EXISTS tool_choice Nullable(String)"
                    .to_string(),
            )
            .await?;
        self.clickhouse
            .run_query_synchronous_no_params(
                r"ALTER TABLE ChatInference ADD COLUMN IF NOT EXISTS parallel_tool_calls Nullable(Bool)"
                    .to_string(),
            )
            .await?;
        self.clickhouse
            .run_query_synchronous_no_params(
                r"ALTER TABLE ChatInferenceDatapoint ADD COLUMN IF NOT EXISTS dynamic_tools Array(String)"
                    .to_string(),
            )
            .await?;
        self.clickhouse
            .run_query_synchronous_no_params(
                r"ALTER TABLE ChatInferenceDatapoint ADD COLUMN IF NOT EXISTS dynamic_provider_tools Array(String)"
                    .to_string(),
            )
            .await?;
        self.clickhouse
            .run_query_synchronous_no_params(
                r"ALTER TABLE ChatInferenceDatapoint ADD COLUMN IF NOT EXISTS allowed_tools Nullable(String)"
                    .to_string(),
            )
            .await?;
        self.clickhouse
            .run_query_synchronous_no_params(
                r"ALTER TABLE ChatInferenceDatapoint ADD COLUMN IF NOT EXISTS tool_choice Nullable(String)"
                    .to_string(),
            )
            .await?;
        self.clickhouse
            .run_query_synchronous_no_params(
                r"ALTER TABLE ChatInferenceDatapoint ADD COLUMN IF NOT EXISTS parallel_tool_calls Nullable(Bool)"
                    .to_string(),
            )
            .await?;
        self.clickhouse
            .run_query_synchronous_no_params(
                r"ALTER TABLE BatchModelInference ADD COLUMN IF NOT EXISTS dynamic_tools Array(String)"
                    .to_string(),
            )
            .await?;
        self.clickhouse
            .run_query_synchronous_no_params(
                r"ALTER TABLE BatchModelInference ADD COLUMN IF NOT EXISTS dynamic_provider_tools Array(String)"
                    .to_string(),
            )
            .await?;
        self.clickhouse
            .run_query_synchronous_no_params(
                r"ALTER TABLE BatchModelInference ADD COLUMN IF NOT EXISTS allowed_tools Nullable(String)"
                    .to_string(),
            )
            .await?;
        self.clickhouse
            .run_query_synchronous_no_params(
                r"ALTER TABLE BatchModelInference ADD COLUMN IF NOT EXISTS tool_choice Nullable(String)"
                    .to_string(),
            )
            .await?;
        self.clickhouse
            .run_query_synchronous_no_params(
                r"ALTER TABLE BatchModelInference ADD COLUMN IF NOT EXISTS parallel_tool_calls Nullable(Bool)"
                    .to_string(),
            )
            .await?;

        Ok(())
    }

    fn rollback_instructions(&self) -> String {
        r"ALTER TABLE ChatInference DROP COLUMN dynamic_tools;
          ALTER TABLE ChatInference DROP COLUMN dynamic_provider_tools;
          ALTER TABLE ChatInference DROP COLUMN allowed_tools;
          ALTER TABLE ChatInference DROP COLUMN tool_choice;
          ALTER TABLE ChatInference DROP COLUMN parallel_tool_calls;
          ALTER TABLE ChatInferenceDatapoint DROP COLUMN dynamic_tools;
          ALTER TABLE ChatInferenceDatapoint DROP COLUMN dynamic_provider_tools;
          ALTER TABLE ChatInferenceDatapoint DROP COLUMN allowed_tools;
          ALTER TABLE ChatInferenceDatapoint DROP COLUMN tool_choice;
          ALTER TABLE ChatInferenceDatapoint DROP COLUMN parallel_tool_calls;
          ALTER TABLE BatchModelInference DROP COLUMN dynamic_tools;
          ALTER TABLE BatchModelInference DROP COLUMN dynamic_provider_tools;
          ALTER TABLE BatchModelInference DROP COLUMN allowed_tools;
          ALTER TABLE BatchModelInference DROP COLUMN tool_choice;
          ALTER TABLE BatchModelInference DROP COLUMN parallel_tool_calls;"
            .to_string()
    }

    async fn has_succeeded(&self) -> Result<bool, Error> {
        let should_apply = self.should_apply().await?;
        Ok(!should_apply)
    }
}
