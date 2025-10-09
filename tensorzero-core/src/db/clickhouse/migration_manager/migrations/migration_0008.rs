use crate::db::clickhouse::migration_manager::migration_trait::Migration;
use crate::db::clickhouse::ClickHouseConnectionInfo;
use crate::error::{Error, ErrorDetails};

use super::{check_column_exists, check_table_exists, get_column_type};
use async_trait::async_trait;

/// This migration continues setting up the ClickHouse database for batch inference.
///
/// It changes the `response_time_ms` column of `ModelInference` and the `processing_time_ms` columns of
/// `ChatInference` and `JsonInference` to be nullable.
/// This is required since we don't really get latency measurements for batch requests.
///
/// It also adds a `raw_request` and `raw_response` column to the `BatchRequest` table for
/// debugging and observability of batch requests.
pub struct Migration0008<'a> {
    pub clickhouse: &'a ClickHouseConnectionInfo,
}

#[async_trait]
impl Migration for Migration0008<'_> {
    /// Ccheck that the tables that need altering already exist
    async fn can_apply(&self) -> Result<(), Error> {
        let tables = [
            "ModelInference",
            "ChatInference",
            "JsonInference",
            "BatchRequest",
        ];
        for table in tables {
            if !check_table_exists(self.clickhouse, table, "0006").await? {
                return Err(ErrorDetails::ClickHouseMigration {
                    id: "0008".to_string(),
                    message: format!("{table} table does not exist"),
                }
                .into());
            }
        }
        Ok(())
    }

    /// Check if the migration has already been applied by checking if the `raw_request`, `raw_response`, `function_name` or and `variant_name`
    /// columns exist in BatchRequest
    /// and if the `processing_time_ms` columns in ChatInference and JsonInference and `response_time_ms` column in ModelInference is Nullable
    async fn should_apply(&self) -> Result<bool, Error> {
        if !check_column_exists(self.clickhouse, "BatchRequest", "raw_request", "0008").await? {
            return Ok(true);
        }
        if !check_column_exists(self.clickhouse, "BatchRequest", "raw_response", "0008").await? {
            return Ok(true);
        }
        if !check_column_exists(self.clickhouse, "BatchRequest", "function_name", "0008").await? {
            return Ok(true);
        }
        if !check_column_exists(self.clickhouse, "BatchRequest", "variant_name", "0008").await? {
            return Ok(true);
        }
        if get_column_type(self.clickhouse, "BatchRequest", "errors", "0008").await?
            != "Array(String)"
        {
            return Ok(true);
        }
        if get_column_type(
            self.clickhouse,
            "ModelInference",
            "response_time_ms",
            "0008",
        )
        .await?
            != "Nullable(UInt32)"
        {
            return Ok(true);
        }
        if get_column_type(
            self.clickhouse,
            "JsonInference",
            "processing_time_ms",
            "0008",
        )
        .await?
            != "Nullable(UInt32)"
        {
            return Ok(true);
        }
        if get_column_type(
            self.clickhouse,
            "ChatInference",
            "processing_time_ms",
            "0008",
        )
        .await?
            != "Nullable(UInt32)"
        {
            return Ok(true);
        }

        // Everything is in place, so we should not apply the migration
        Ok(false)
    }

    async fn apply(&self, _clean_start: bool) -> Result<(), Error> {
        // Add a `raw_request` column, a `raw_response` column, a `function_name` column and a `variant_name` column
        // to the `BatchRequest` table
        let query = r"
            ALTER TABLE BatchRequest
            ADD COLUMN IF NOT EXISTS raw_request String,
            ADD COLUMN IF NOT EXISTS raw_response String,
            ADD COLUMN IF NOT EXISTS function_name LowCardinality(String),
            ADD COLUMN IF NOT EXISTS variant_name LowCardinality(String),
            MODIFY COLUMN errors Array(String);";
        // NOTE: this MODIFY COLUMN errors statement would convert data in bad ways
        // HOWEVER, TensorZero at the point of writing has never actually written any errors to the errors column
        // so this is safe to do.
        let _ = self
            .clickhouse
            .run_query_synchronous_no_params(query.to_string())
            .await?;

        // Alter the `response_time_ms` column of `ModelInference` to be a nullable column
        let query = r"
            ALTER TABLE ModelInference
            MODIFY COLUMN response_time_ms Nullable(UInt32)
        ";
        let _ = self
            .clickhouse
            .run_query_synchronous_no_params(query.to_string())
            .await?;

        // Alter the `processing_time_ms` column of `ChatInference` to be a nullable column
        let query = r"
            ALTER TABLE ChatInference
            MODIFY COLUMN processing_time_ms Nullable(UInt32)
        ";
        let _ = self
            .clickhouse
            .run_query_synchronous_no_params(query.to_string())
            .await?;

        // Alter the `processing_time_ms` column of `JsonInference` to be a nullable column
        let query = r"
            ALTER TABLE JsonInference
            MODIFY COLUMN processing_time_ms Nullable(UInt32)
        ";
        let _ = self
            .clickhouse
            .run_query_synchronous_no_params(query.to_string())
            .await?;

        Ok(())
    }

    /// Check if the migration has succeeded (i.e. it should not be applied again)
    async fn has_succeeded(&self) -> Result<bool, Error> {
        let should_apply = self.should_apply().await?;
        Ok(!should_apply)
    }

    fn rollback_instructions(&self) -> String {
        // NOTE - we do not try to roll back the 'BatchRequest.errors' column, as
        // ALTER TABLE MODIFY COLUMN' cannot change an array back into a Map
        // There shouldn't be anything in the database when this runs, so this is fine -
        // re-applying the migration after a rollback will just leave the 'errors' column unchanged
        "/* Change the timing columns back to non-nullable types */\
            ALTER TABLE ModelInference MODIFY COLUMN response_time_ms UInt32;
            ALTER TABLE ChatInference MODIFY COLUMN processing_time_ms UInt32;
            ALTER TABLE JsonInference MODIFY COLUMN processing_time_ms UInt32;
            /* Drop the columns */\
            ALTER TABLE BatchRequest \
            DROP COLUMN raw_request,\
            DROP COLUMN raw_response,\
            DROP COLUMN function_name,\
            DROP COLUMN variant_name;
        "
        .to_string()
    }
}
