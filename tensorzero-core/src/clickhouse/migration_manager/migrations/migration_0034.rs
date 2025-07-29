use async_trait::async_trait;

use super::{create_table_engine, create_replacing_table_engine};
use crate::clickhouse::migration_manager::migration_trait::Migration;
use crate::clickhouse::ClickHouseConnectionInfo;
use crate::config_parser::Config;
use crate::error::{Error, ErrorDetails};

pub struct Migration0034<'a> {
    pub clickhouse: &'a ClickHouseConnectionInfo,
    pub config: &'a Config,
}

#[async_trait]
impl Migration for Migration0034<'_> {
    fn name(&self) -> String {
        "Migration0034".to_string()
    }

    async fn can_apply(&self) -> Result<(), Error> {
        // Only apply if replication is enabled
        if !self.config.clickhouse.replication_enabled {
            return Ok(());
        }

        // Check that we can connect to ClickHouse
        self.clickhouse.health().await?;
        Ok(())
    }

    async fn should_apply(&self) -> Result<bool, Error> {
        // Only apply if replication is enabled
        if !self.config.clickhouse.replication_enabled {
            return Ok(false);
        }

        // Check if there are any non-replicated MergeTree tables that need conversion
        let database = self.clickhouse.database();
        let tables_query = format!(
            "SELECT COUNT(*) FROM system.tables WHERE database = '{database}' AND engine NOT LIKE 'Replicated%' AND engine LIKE 'MergeTree%'"
        );
        
        let response = self.clickhouse
            .run_query_synchronous_no_params(tables_query)
            .await
            .map_err(|e| {
                Error::new(ErrorDetails::ClickHouseMigration {
                    id: "0034".to_string(),
                    message: format!("Failed to check for non-replicated tables: {e}"),
                })
            })?;

        let count: u64 = response.response.trim().parse().unwrap_or(0);
        Ok(count > 0)
    }

    async fn apply(&self, _clean_start: bool) -> Result<(), Error> {
        let cluster_name = &self.config.clickhouse.cluster_name;

        tracing::info!("Migration0034: Converting tables to replicated engines for cluster: {cluster_name}");

        // Get list of all tables in the database that might need conversion
        let database = self.clickhouse.database();
        let tables_query = format!(
            "SELECT name, engine FROM system.tables WHERE database = '{database}' AND engine NOT LIKE 'Replicated%'"
        );
        let response = self.clickhouse
            .run_query_synchronous_no_params(tables_query)
            .await
            .map_err(|e| {
                Error::new(ErrorDetails::ClickHouseMigration {
                    id: "0034".to_string(),
                    message: format!("Failed to get list of non-replicated tables: {e}"),
                })
            })?;

        // Parse the response to get table names and engines
        let mut tables_to_convert = Vec::new();
        for line in response.response.lines() {
            let parts: Vec<&str> = line.split('\t').collect();
            if parts.len() >= 2 {
                let table_name = parts[0].trim();
                let engine = parts[1].trim();
                
                // Skip system tables and materialized views, only convert MergeTree family engines
                if !table_name.starts_with('.') && 
                   (engine.starts_with("MergeTree") || engine.starts_with("ReplacingMergeTree")) {
                    tables_to_convert.push((table_name.to_string(), engine.to_string()));
                }
            }
        }

        if tables_to_convert.is_empty() {
            tracing::info!("Migration0034: All tables are already replicated or no MergeTree tables found");
            return Ok(());
        }

        // Convert each table
        for (table_name, current_engine) in tables_to_convert {
            tracing::info!("Migration0034: Converting table {table_name} from {current_engine} to replicated engine");

            // Determine the replicated engine using helper functions
            let replicated_engine = if current_engine.starts_with("ReplacingMergeTree") {
                // Extract parameters from ReplacingMergeTree(params)
                let params = if let Some(params_start) = current_engine.find('(') {
                    if let Some(params_end) = current_engine.find(')') {
                        let inner = &current_engine[params_start + 1..params_end];
                        if inner.is_empty() { None } else { Some(inner) }
                    } else { None }
                } else { None };
                
                create_replacing_table_engine(
                    true, // Always replicated in this context
                    cluster_name,
                    &table_name,
                    params
                )
            } else {
                create_table_engine(
                    true, // Always replicated in this context
                    cluster_name,
                    &table_name
                )
            };

            tracing::info!("Migration0034: Original engine: {current_engine}");
            tracing::info!("Migration0034: Final engine: {replicated_engine}");

            // Get the full CREATE TABLE statement for the existing table
            let show_create_query = format!("SHOW CREATE TABLE {table_name}");
            let create_response = self.clickhouse
                .run_query_synchronous_no_params(show_create_query)
                .await
                .map_err(|e| {
                    Error::new(ErrorDetails::ClickHouseMigration {
                        id: "0034".to_string(),
                        message: format!("Failed to get CREATE TABLE statement for {table_name}: {e}"),
                    })
                })?;

            // Parse the CREATE TABLE statement and replace the engine
            let create_statement = create_response.response.trim();
            let temp_table_name = format!("{table_name}_replicated_temp");
            
            // Replace literal \n with actual newlines and \' with actual quotes (ClickHouse SHOW CREATE TABLE returns escaped strings)
            let create_statement = create_statement
                .replace("\\n", "\n")
                .replace("\\'", "'");
            
            // Replace table name first - handle both qualified and unqualified table names
            let database = self.clickhouse.database();
            let mut new_create_statement = create_statement
                .replace(&format!("CREATE TABLE {database}.{table_name}"), &format!("CREATE TABLE {database}.{temp_table_name}"))
                .replace(&format!("CREATE TABLE {table_name}"), &format!("CREATE TABLE {temp_table_name}"));
            
            // Find and replace the ENGINE clause more carefully
            // Look for "ENGINE = " followed by anything until "ORDER BY" or end of statement
            if let Some(engine_start) = new_create_statement.find("ENGINE = ") {
                let before_engine = &new_create_statement[..engine_start + "ENGINE = ".len()];
                let after_engine_start = &new_create_statement[engine_start + "ENGINE = ".len()..];
                
                // Find the end of the engine clause (look for ORDER BY or end of string)
                let engine_end = after_engine_start.find("ORDER BY")
                    .or_else(|| after_engine_start.find("PARTITION BY"))
                    .or_else(|| after_engine_start.find("SETTINGS"))
                    .unwrap_or(after_engine_start.len());
                
                let after_engine = &after_engine_start[engine_end..];
                new_create_statement = format!("{before_engine}{replicated_engine}\n{after_engine}");
            }

            tracing::info!("Migration0034: Creating replicated table with statement: {new_create_statement}");

            // Clean up any existing temp table from previous failed runs
            let cleanup_query = format!("DROP TABLE IF EXISTS {temp_table_name}");
            let _ = self.clickhouse
                .run_query_synchronous_no_params(cleanup_query)
                .await; // Ignore errors for cleanup

            // Create temporary replicated table
            self.clickhouse
                .run_query_synchronous_no_params(new_create_statement)
                .await
                .map_err(|e| {
                    Error::new(ErrorDetails::ClickHouseMigration {
                        id: "0034".to_string(),
                        message: format!("Failed to create temporary replicated table {temp_table_name}: {e}"),
                    })
                })?;

            // Copy data from original table to temporary table
            let copy_query = format!("INSERT INTO {temp_table_name} SELECT * FROM {table_name}");
            self.clickhouse
                .run_query_synchronous_no_params(copy_query)
                .await
                .map_err(|e| {
                    Error::new(ErrorDetails::ClickHouseMigration {
                        id: "0034".to_string(),
                        message: format!("Failed to copy data from {table_name} to {temp_table_name}: {e}"),
                    })
                })?;

            // Drop original table
            let drop_query = format!("DROP TABLE {table_name}");
            self.clickhouse
                .run_query_synchronous_no_params(drop_query)
                .await
                .map_err(|e| {
                    Error::new(ErrorDetails::ClickHouseMigration {
                        id: "0034".to_string(),
                        message: format!("Failed to drop original table {table_name}: {e}"),
                    })
                })?;

            // Rename temporary table to original name
            let rename_query = format!("RENAME TABLE {temp_table_name} TO {table_name}");
            self.clickhouse
                .run_query_synchronous_no_params(rename_query)
                .await
                .map_err(|e| {
                    Error::new(ErrorDetails::ClickHouseMigration {
                        id: "0034".to_string(),
                        message: format!("Failed to rename {temp_table_name} to {table_name}: {e}"),
                    })
                })?;

            tracing::info!("Migration0034: Successfully converted table {table_name} to replicated engine");
        }

        tracing::info!("Migration0034: Completed conversion of all tables to replicated engines");
        Ok(())
    }

    async fn has_succeeded(&self) -> Result<bool, Error> {
        // If replication is not enabled, consider this migration successful (no-op)
        if !self.config.clickhouse.replication_enabled {
            return Ok(true);
        }

        // Check that there are no more non-replicated MergeTree tables
        let database = self.clickhouse.database();
        let tables_query = format!(
            "SELECT COUNT(*) FROM system.tables WHERE database = '{database}' AND engine NOT LIKE 'Replicated%' AND engine LIKE 'MergeTree%'"
        );
        
        let response = self.clickhouse
            .run_query_synchronous_no_params(tables_query)
            .await
            .map_err(|e| {
                Error::new(ErrorDetails::ClickHouseMigration {
                    id: "0034".to_string(),
                    message: format!("Failed to verify migration success: {e}"),
                })
            })?;

        let count: u64 = response.response.trim().parse().unwrap_or(0);
        Ok(count == 0)
    }

    fn rollback_instructions(&self) -> String {
        r#"
Migration0034 converted non-replicated MergeTree tables to replicated engines.

To rollback:
1. For each table that was converted, you would need to:
   - Create a new table with the original non-replicated engine
   - Copy data from the replicated table to the new table
   - Drop the replicated table
   - Rename the new table to the original name

However, this rollback should generally not be necessary as:
- The data is preserved during conversion
- Replicated tables are functionally equivalent to non-replicated tables
- Rolling back would lose the benefits of replication

If you must rollback, manually inspect each table's structure and recreate with
MergeTree or ReplacingMergeTree engines instead of their Replicated* equivalents.
"#.to_string()
    }
}
