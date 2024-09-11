use serde_json::Value;

use crate::clickhouse::ClickHouseConnectionInfo;
use crate::clickhouse_migration_manager::migration_trait::Migration;
use crate::error::Error;

/// This migration modifies the `Inference` table to add a column for
/// `output_schema`.

pub struct Migration0002<'a> {
    pub clickhouse: &'a ClickHouseConnectionInfo,
}

impl<'a> Migration for Migration0002<'a> {
    /// Check if you can connect to the database
    async fn can_apply(&self) -> Result<(), Error> {
        self.clickhouse
            .health()
            .await
            .map_err(|e| Error::ClickHouseMigration {
                id: "0002".to_string(),
                message: e.to_string(),
            })?;
        // Should be a newline-delimited list of table names
        let tables = self.clickhouse.run_query("SHOW TABLES".to_string()).await?;
        // Parse into a vec of &str
        let table_vec: Vec<&str> = tables.split('\n').map(|s| s.trim()).collect();
        // Throw an error if Inference is not in the set
        if !table_vec.contains(&"Inference") {
            return Err(Error::ClickHouseMigration {
                id: "0002".to_string(),
                message: "Inference table not found".to_string(),
            });
        }
        Ok(())
    }

    /// Check if the columns exist, if not then we should apply the migration
    async fn should_apply(&self) -> Result<bool, Error> {
        let rows = self
            .clickhouse
            .run_query("DESCRIBE Inference FORMAT JSONEachRow".to_string())
            .await?;
        // rows is a JSONL format, let's iterate over each row
        for row in rows.trim().split('\n') {
            let row: Value = serde_json::from_str(row).map_err(|e| Error::ClickHouseMigration {
                id: "0002".to_string(),
                message: e.to_string(),
            })?;
            let name = row.get("name").ok_or(Error::ClickHouseMigration {
                id: "0002".to_string(),
                message: "name field not found in column of Inference".to_string(),
            })?;
            if name == "output_schema" {
                return Ok(false);
            }
        }

        Ok(true)
    }

    async fn apply(&self) -> Result<(), Error> {
        // Alter the `ModelInference` table to add the `model_name` and `model_provider_name` columns
        self.clickhouse
            .run_query("ALTER TABLE Inference ADD COLUMN output_schema String".to_string())
            .await?;
        Ok(())
    }

    fn rollback_instructions(&self) -> String {
        "\
            **CAREFUL: THIS WILL DELETE DATA**\n\
            \n\
            -- Drop the columns\n\
            ALTER TABLE Inference DROP COLUMN output_schema\n\
            \n\
            **CAREFUL: THIS WILL DELETE DATA**\n\
            "
        .to_string()
    }

    /// Check if the migration has succeeded (i.e. it should not be applied again)
    async fn has_succeeded(&self) -> Result<bool, Error> {
        let should_apply = self.should_apply().await?;
        Ok(!should_apply)
    }
}
