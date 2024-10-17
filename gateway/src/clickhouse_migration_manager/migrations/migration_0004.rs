use crate::clickhouse::ClickHouseConnectionInfo;
use crate::clickhouse_migration_manager::migration_trait::Migration;
use crate::error::Error;

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

impl<'a> Migration for Migration0004<'a> {
    /// Check if you can connect to the database
    /// Then check if the four feedback tables exist as the sources for the materialized views
    /// If all of this is OK, then we can apply the migration
    async fn can_apply(&self) -> Result<(), Error> {
        self.clickhouse
            .health()
            .await
            .map_err(|e| Error::ClickHouseMigration {
                id: "0004".to_string(),
                message: e.to_string(),
            })?;
        let database = self.clickhouse.database();
        let query = format!(
            r#"SELECT EXISTS(
                SELECT 1
                FROM system.tables
                WHERE database = '{database}' AND name = 'ModelInference'
            )"#
        );

        match self.clickhouse.run_query(query).await {
            Ok(response) => {
                if response.trim() != "1" {
                    return Err(Error::ClickHouseMigration {
                        id: "0004".to_string(),
                        message: "ModelInference table does not exist".to_string(),
                    });
                }
            }
            Err(e) => {
                return Err(Error::ClickHouseMigration {
                    id: "0004".to_string(),
                    message: format!("Failed to check ModelInference table: {}", e),
                });
            }
        }

        Ok(())
    }

    /// Check if the migration has already been applied
    /// This should be equivalent to checking if `FeedbackTag` exists
    async fn should_apply(&self) -> Result<bool, Error> {
        let database = self.clickhouse.database();
        let query = format!(
            "SELECT name FROM system.columns WHERE database = '{}' AND table = 'ModelInference'",
            database
        );
        let response =
            self.clickhouse
                .run_query(query)
                .await
                .map_err(|e| Error::ClickHouseMigration {
                    id: "0004".to_string(),
                    message: format!("Failed to fetch columns for ModelInference: {}", e),
                })?;
        let present_columns: Vec<&str> = response.lines().map(|line| line.trim()).collect();
        if present_columns.contains(&"system")
            && present_columns.contains(&"input_messages")
            && present_columns.contains(&"output")
        {
            Ok(false)
        } else {
            Ok(true)
        }
    }

    async fn apply(&self) -> Result<(), Error> {
        // Add a column `system` to the `ModelInference` table
        let query = r#"
            ALTER TABLE ModelInference
            ADD COLUMN IF NOT EXISTS system Nullable(String)
        "#;
        let _ = self.clickhouse.run_query(query.to_string()).await?;

        // Add a column `input_messages` to the `ModelInference` table
        let query = r#"
            ALTER TABLE ModelInference
            ADD COLUMN IF NOT EXISTS input_messages String
        "#;
        let _ = self.clickhouse.run_query(query.to_string()).await?;

        // Add a column `output` to the `ModelInference` table
        let query = r#"
            ALTER TABLE ModelInference
            ADD COLUMN IF NOT EXISTS output String
        "#;
        let _ = self.clickhouse.run_query(query.to_string()).await?;

        Ok(())
    }

    fn rollback_instructions(&self) -> String {
        "\
            -- Drop the columns\n\
            ALTER TABLE ModelInference DROP COLUMN system;\n\
            ALTER TABLE ModelInference DROP COLUMN input_messages;\n\
            ALTER TABLE ModelInference DROP COLUMN output;\n\
        "
        .to_string()
    }

    /// Check if the migration has succeeded (i.e. it should not be applied again)
    async fn has_succeeded(&self) -> Result<bool, Error> {
        let should_apply = self.should_apply().await?;
        Ok(!should_apply)
    }
}
