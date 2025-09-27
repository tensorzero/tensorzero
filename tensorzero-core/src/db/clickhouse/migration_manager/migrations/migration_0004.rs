use crate::db::clickhouse::migration_manager::migration_trait::Migration;
use crate::db::clickhouse::ClickHouseConnectionInfo;
use crate::error::{Error, ErrorDetails};

use super::{check_column_exists, check_table_exists};
use async_trait::async_trait;

/// This migration adds additional columns to the `ModelInference` table
/// The goal of this is to improve observability of each model inference at an intermediate level of granularity.
/// Prior to this migration, we only stored the raw request and response, which vary by provider and are therefore
/// hard to use in any structured way, though useful for debugging.
///
/// In this migration, we add columns `system`, `input_messages`, and `output` that all will be strings with the latter two structured as JSON.
/// These will contain the system message, the input messages (as a List[List[ContentBlock]]) and the output (as a List[ContentBlock]).
///
/// This will be useful to all who need to understand exactly what went in and out of the LLM.
pub struct Migration0004<'a> {
    pub clickhouse: &'a ClickHouseConnectionInfo,
}

#[async_trait]
impl Migration for Migration0004<'_> {
    /// Check if the ModelInference table exists
    /// If all of this is OK, then we can apply the migration
    async fn can_apply(&self) -> Result<(), Error> {
        if !check_table_exists(self.clickhouse, "ModelInference", "0004").await? {
            return Err(ErrorDetails::ClickHouseMigration {
                id: "0004".to_string(),
                message: "ModelInference table does not exist".to_string(),
            }
            .into());
        }

        Ok(())
    }

    /// Check if the migration has already been applied by checking if the new columns exist
    async fn should_apply(&self) -> Result<bool, Error> {
        let system_column_exists =
            check_column_exists(self.clickhouse, "ModelInference", "system", "0004").await?;
        let input_messages_column_exists =
            check_column_exists(self.clickhouse, "ModelInference", "input_messages", "0004").await?;
        let output_column_exists =
            check_column_exists(self.clickhouse, "ModelInference", "output", "0004").await?;
            
        if system_column_exists && input_messages_column_exists && output_column_exists {
            Ok(false)
        } else {
            Ok(true)
        }
    }

    async fn apply(&self, _clean_start: bool) -> Result<(), Error> {
        // Add system, input_messages, and output columns to ModelInference using sharding-aware ALTER
        self.clickhouse
            .get_alter_table_statements(
                "ModelInference",
                "ADD COLUMN IF NOT EXISTS system Nullable(String), ADD COLUMN IF NOT EXISTS input_messages String, ADD COLUMN IF NOT EXISTS output String",
                false,
            )
            .await?;

        Ok(())
    }

    fn rollback_instructions(&self) -> String {
        format!(
            "/* Drop the columns */\
            {}",
            self.clickhouse.get_alter_table_rollback_statements(
                "ModelInference", 
                "DROP COLUMN system, DROP COLUMN input_messages, DROP COLUMN output",
                false
            )
        )
    }

    /// Check if the migration has succeeded (i.e. it should not be applied again)
    async fn has_succeeded(&self) -> Result<bool, Error> {
        let should_apply = self.should_apply().await?;
        Ok(!should_apply)
    }
}
