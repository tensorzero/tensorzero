use serde_json::Value;

use crate::clickhouse::ClickHouseConnectionInfo;
use crate::clickhouse_migration_manager::migration_trait::Migration;
use crate::error::Error;

/// This migration modifies the `ModelInference` table to add columns for
/// `model_name` and `model_provider_name`.

pub struct Migration0001<'a> {
    pub clickhouse: &'a ClickHouseConnectionInfo,
}

impl<'a> Migration for Migration0001<'a> {
    /// Check if you can connect to the database
    async fn can_apply(&self) -> Result<(), Error> {
        self.clickhouse
            .health()
            .await
            .map_err(|e| Error::ClickHouseMigration {
                id: "0001".to_string(),
                message: e.to_string(),
            })?;
        // Should be a newline-delimited list of table names
        let tables = self.clickhouse.run_query("SHOW TABLES".to_string()).await?;
        // Parse into a vec of &str
        let table_vec: Vec<&str> = tables.split('\n').map(|s| s.trim()).collect();
        // Throw an error if ModelInference is not in the set
        if !table_vec.contains(&"ModelInference") {
            return Err(Error::ClickHouseMigration {
                id: "0001".to_string(),
                message: "ModelInference table not found".to_string(),
            });
        }
        Ok(())
    }

    /// Check if the columns exist, if not then we should apply the migration
    async fn should_apply(&self) -> Result<bool, Error> {
        let rows = self
            .clickhouse
            .run_query("DESCRIBE ModelInference FORMAT JSONEachRow".to_string())
            .await?;
        // rows is a JSONL format, let's iterate over each row
        for row in rows.trim().split('\n') {
            let row: Value = serde_json::from_str(row).map_err(|e| Error::ClickHouseMigration {
                id: "0001".to_string(),
                message: e.to_string(),
            })?;
            let name = row.get("name").ok_or(Error::ClickHouseMigration {
                id: "0001".to_string(),
                message: "name field not found in column of ModelInference".to_string(),
            })?;
            if name == "model_name" || name == "model_provider_name" {
                return Ok(false);
            }
        }

        Ok(true)
    }

    async fn apply(&self) -> Result<(), Error> {
        // Alter the `ModelInference` table to add the `model_name` and `model_provider_name` columns
        self.clickhouse
            .run_query("ALTER TABLE ModelInference ADD COLUMN model_name String, ADD COLUMN model_provider_name String".to_string())
            .await?;
        Ok(())
    }

    fn rollback_instructions(&self) -> String {
        "\
            **CAREFUL: THIS WILL DELETE ALL DATA**\n\
            \n\
            -- Drop the database\n\
            ALTER TABLE ModelInference DROP COLUMN model_name, DROP COLUMN model_provider_name\n\
            \n\
            **CAREFUL: THIS WILL DELETE ALL DATA**\n\
            "
        .to_string()
    }

    /// Check if the migration has succeeded (i.e. it should not be applied again)
    async fn has_succeeded(&self) -> Result<bool, Error> {
        let should_apply = self.should_apply().await?;
        Ok(!should_apply)
    }
}
