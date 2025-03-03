use crate::clickhouse::ClickHouseConnectionInfo;
use crate::clickhouse_migration_manager::migration_trait::Migration;
use crate::error::{Error, ErrorDetails};
use async_trait::async_trait;

use super::get_column_type;

/// This migration is used to mark the `input_tokens` and `output_tokens` columns as nullable in the `ModelInference` table.
pub struct Migration0015<'a> {
    pub clickhouse: &'a ClickHouseConnectionInfo,
}

#[async_trait]
impl Migration for Migration0015<'_> {
    /// Check if you can connect to the database
    async fn can_apply(&self) -> Result<(), Error> {
        self.clickhouse.health().await.map_err(|e| {
            Error::new(ErrorDetails::ClickHouseMigration {
                id: "0015".to_string(),
                message: e.to_string(),
            })
        })?;

        Ok(())
    }

    /// Check if the migration has already been applied by checking if
    /// the `input_tokens` and `output_tokens` columns are nullable
    async fn should_apply(&self) -> Result<bool, Error> {
        if get_column_type(self.clickhouse, "ModelInference", "input_tokens", "0015").await?
            != "Nullable(UInt32)"
        {
            return Ok(true);
        }

        if get_column_type(self.clickhouse, "ModelInference", "output_tokens", "0015").await?
            != "Nullable(UInt32)"
        {
            return Ok(true);
        }

        // Everything is in place, so we should not apply the migration
        Ok(false)
    }

    async fn apply(&self) -> Result<(), Error> {
        // Alter the `input_tokens` column of `ModelInference` to be a nullable column
        let query = r#"
            ALTER TABLE ModelInference
            MODIFY COLUMN input_tokens Nullable(UInt32)
        "#;
        let _ = self.clickhouse.run_query(query.to_string(), None).await?;

        // Alter the `output_tokens` column of `ModelInference` to be a nullable column
        let query = r#"
            ALTER TABLE ModelInference
            MODIFY COLUMN output_tokens Nullable(UInt32)
        "#;
        let _ = self.clickhouse.run_query(query.to_string(), None).await?;

        Ok(())
    }

    fn rollback_instructions(&self) -> String {
        "\
            -- Change the columns back to non-nullable types\n\
            ALTER TABLE ModelInference
            MODIFY COLUMN input_tokens UInt32;\n\
            ALTER TABLE ModelInference
            MODIFY COLUMN output_tokens UInt32;\n\
            "
        .to_string()
    }

    /// Check if the migration has succeeded (i.e. it should not be applied again)
    async fn has_succeeded(&self) -> Result<bool, Error> {
        let should_apply = self.should_apply().await?;
        Ok(!should_apply)
    }
}
