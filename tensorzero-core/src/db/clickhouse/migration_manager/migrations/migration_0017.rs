use crate::db::clickhouse::migration_manager::migration_trait::Migration;
use crate::db::clickhouse::ClickHouseConnectionInfo;
use crate::error::{Error, ErrorDetails};
use async_trait::async_trait;

use super::{check_column_exists, check_table_exists};

/// This migration updates the cache to begin storing usage data for cached examples
/// To do this we update the `ModelInferenceCache` table by adding a column for `input_tokens` and `output_tokens`
/// These are UInt32 valued columns that will default to 0 for previously cached examples.
pub struct Migration0017<'a> {
    pub clickhouse: &'a ClickHouseConnectionInfo,
}

#[async_trait]
impl Migration for Migration0017<'_> {
    async fn can_apply(&self) -> Result<(), Error> {
        let model_inference_cache_table_exists =
            check_table_exists(self.clickhouse, "ModelInferenceCache", "0017").await?;
        if !model_inference_cache_table_exists {
            return Err(Error::new(ErrorDetails::ClickHouseMigration {
                id: "0017".to_string(),
                message: "ModelInferenceCache table does not exist".to_string(),
            }));
        }

        Ok(())
    }

    /// Check if the migration needs to be applied
    /// This should be equivalent to checking if the `input_tokens` or `output_tokens` columns
    /// are missing from the `ModelInferenceCache` table
    async fn should_apply(&self) -> Result<bool, Error> {
        let input_tokens_column_exists = check_column_exists(
            self.clickhouse,
            "ModelInferenceCache",
            "input_tokens",
            "0017",
        )
        .await?;
        let output_tokens_column_exists = check_column_exists(
            self.clickhouse,
            "ModelInferenceCache",
            "output_tokens",
            "0017",
        )
        .await?;
        Ok(!input_tokens_column_exists || !output_tokens_column_exists)
    }

    async fn apply(&self, _clean_start: bool) -> Result<(), Error> {
        // Add the `input_tokens` and `output_tokens` columns to the `ModelInferenceCache` table
        let query = r"
            ALTER TABLE ModelInferenceCache
            ADD COLUMN IF NOT EXISTS input_tokens UInt32,
            ADD COLUMN IF NOT EXISTS output_tokens UInt32
        ";
        let _ = self
            .clickhouse
            .run_query_synchronous_no_params(query.to_string())
            .await?;

        Ok(())
    }

    fn rollback_instructions(&self) -> String {
        "/* Drop the columns */\
            ALTER TABLE ModelInferenceCache \
            DROP COLUMN IF EXISTS input_tokens,\
            DROP COLUMN IF EXISTS output_tokens;\
            "
        .to_string()
    }

    /// Check if the migration has succeeded (i.e. it should not be applied again)
    async fn has_succeeded(&self) -> Result<bool, Error> {
        let should_apply = self.should_apply().await?;
        Ok(!should_apply)
    }
}
