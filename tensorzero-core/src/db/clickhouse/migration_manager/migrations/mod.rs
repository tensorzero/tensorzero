use crate::{
    db::clickhouse::ClickHouseConnectionInfo,
    error::{Error, ErrorDetails},
};

pub mod migration_0000;
pub mod migration_0002;
pub mod migration_0003;
pub mod migration_0004;
pub mod migration_0005;
pub mod migration_0006;
pub mod migration_0008;
pub mod migration_0009;
pub mod migration_0011;
pub mod migration_0013;
pub mod migration_0015;
pub mod migration_0016;
pub mod migration_0017;
pub mod migration_0018;
pub mod migration_0019;
pub mod migration_0020;
pub mod migration_0021;
pub mod migration_0022;
pub mod migration_0024;
pub mod migration_0025;
pub mod migration_0026;
pub mod migration_0027;
pub mod migration_0028;
pub mod migration_0029;
pub mod migration_0030;
pub mod migration_0031;
pub mod migration_0032;
pub mod migration_0033;
pub mod migration_0034;
pub mod migration_0035;
pub mod migration_0036;
pub mod migration_0037;
pub mod migration_0038;

/// Returns true if the table exists, false if it does not
/// Errors if the query fails
/// This function also works to check for materialized views
pub async fn check_table_exists(
    clickhouse: &ClickHouseConnectionInfo,
    table: &str,
    migration_id: &str,
) -> Result<bool, Error> {
    let query = format!(
        "SELECT 1 FROM system.tables WHERE database = '{}' AND name = '{}'",
        clickhouse.database(),
        table
    );
    match clickhouse.run_query_synchronous_no_params(query).await {
        Err(e) => {
            return Err(ErrorDetails::ClickHouseMigration {
                id: migration_id.to_string(),
                message: e.to_string(),
            }
            .into())
        }
        Ok(response) => {
            if response.response.trim() != "1" {
                return Ok(false);
            }
        }
    }
    Ok(true)
}

/// Returns true if the table exists in the detached state, false if it does not
/// Errors if the query fails
/// This function also works to check for materialized views
pub async fn check_detached_table_exists(
    clickhouse: &ClickHouseConnectionInfo,
    table: &str,
    migration_id: &str,
) -> Result<bool, Error> {
    let query = format!(
        "SELECT 1 FROM system.detached_tables WHERE database = '{}' AND table = '{}'",
        clickhouse.database(),
        table
    );
    match clickhouse.run_query_synchronous_no_params(query).await {
        Err(e) => {
            return Err(ErrorDetails::ClickHouseMigration {
                id: migration_id.to_string(),
                message: e.to_string(),
            }
            .into())
        }
        Ok(response) => {
            if response.response.trim() != "1" {
                return Ok(false);
            }
        }
    }
    Ok(true)
}

/// Returns true if the column exists in the table, false if it does not
/// For sharded deployments, checks both distributed and local tables
/// Errors if the query fails
pub async fn check_column_exists(
    clickhouse: &ClickHouseConnectionInfo,
    table: &str,
    column: &str,
    migration_id: &str,
) -> Result<bool, Error> {
    let database = clickhouse.database();
    
    if clickhouse.is_sharding_enabled() {
        // Check both distributed and local tables in sharded deployments
        let local_table = format!("{}_local", table);
        
        let query = format!(
            r"
                SELECT count() as table_count
                FROM (
                    SELECT 1 FROM system.columns 
                    WHERE database = '{database}' AND table = '{table}' AND name = '{column}'
                    UNION ALL
                    SELECT 1 FROM system.columns 
                    WHERE database = '{database}' AND table = '{local_table}' AND name = '{column}'
                )
            "
        );
        
        match clickhouse.run_query_synchronous_no_params(query).await {
            Err(e) => {
                return Err(ErrorDetails::ClickHouseMigration {
                    id: migration_id.to_string(),
                    message: e.to_string(),
                }
                .into())
            }
            Ok(response) => {
                let count: u32 = response.response.trim().parse().unwrap_or(0);
                // Column should exist in both tables (count = 2) for sharded deployments
                return Ok(count == 2);
            }
        }
    } else {
        // Check only the main table in non-sharded deployments
        let query = format!(
            r"
                SELECT 1
                FROM system.columns
                WHERE database = '{database}'
                  AND table = '{table}'
                  AND name = '{column}'
                LIMIT 1
            "
        );
        
        match clickhouse.run_query_synchronous_no_params(query).await {
            Err(e) => {
                return Err(ErrorDetails::ClickHouseMigration {
                    id: migration_id.to_string(),
                    message: e.to_string(),
                }
                .into())
            }
            Ok(response) => {
                return Ok(response.response.trim() == "1");
            }
        }
    }
}

pub async fn get_column_type(
    clickhouse: &ClickHouseConnectionInfo,
    table: &str,
    column: &str,
    migration_id: &str,
) -> Result<String, Error> {
    let database = clickhouse.database();
    
    // For sharded deployments, check the local table as it contains the actual data structure
    let table_to_check = if clickhouse.is_sharding_enabled() {
        format!("{}_local", table)
    } else {
        table.to_string()
    };
    
    let query = format!(
        "SELECT type FROM system.columns WHERE database='{database}' AND table='{table_to_check}' AND name='{column}'"
    );
    match clickhouse.run_query_synchronous_no_params(query).await {
        Err(e) => Err(ErrorDetails::ClickHouseMigration {
            id: migration_id.to_string(),
            message: e.to_string(),
        }
        .into()),
        Ok(response) => Ok(response.response.trim().to_string()),
    }
}

pub async fn get_default_expression(
    clickhouse: &ClickHouseConnectionInfo,
    table: &str,
    column: &str,
    migration_id: &str,
) -> Result<String, Error> {
    let database = clickhouse.database();
    
    // For sharded deployments, check the local table as it contains the actual data structure
    let table_to_check = if clickhouse.is_sharding_enabled() {
        format!("{}_local", table)
    } else {
        table.to_string()
    };
    
    let query = format!(
        "SELECT default_expression FROM system.columns WHERE database='{database}' AND table='{table_to_check}' AND name='{column}'"
    );
    match clickhouse.run_query_synchronous_no_params(query).await {
        Err(e) => Err(ErrorDetails::ClickHouseMigration {
            id: migration_id.to_string(),
            message: e.to_string(),
        }
        .into()),
        Ok(response) => Ok(response.response.trim().to_string()),
    }
}

async fn table_is_nonempty(
    clickhouse: &ClickHouseConnectionInfo,
    table: &str,
    migration_id: &str,
) -> Result<bool, Error> {
    // For data checking, we need to check the local table in sharded deployments
    // since that's where the actual data is stored
    let table_to_check = if clickhouse.is_sharding_enabled() {
        clickhouse.get_local_table_name(table)
    } else {
        table.to_string()
    };
    
    let query = format!("SELECT COUNT() FROM {table_to_check} FORMAT CSV");
    let result = clickhouse.run_query_synchronous_no_params(query).await?;
    Ok(result.response.trim().parse::<i64>().map_err(|e| {
        Error::new(ErrorDetails::ClickHouseMigration {
            id: migration_id.to_string(),
            message: e.to_string(),
        })
    })? > 0)
}

async fn get_table_engine(
    clickhouse: &ClickHouseConnectionInfo,
    table: &str,
) -> Result<String, Error> {
    // For engine inspection, we need to check the local table in sharded deployments
    // since that's where the actual table definition/engine is stored
    let table_to_check = if clickhouse.is_sharding_enabled() {
        clickhouse.get_local_table_name(table)
    } else {
        table.to_string()
    };
    
    let query = format!(
        "SELECT engine FROM system.tables WHERE database='{}' AND name='{}'",
        clickhouse.database(),
        table_to_check
    );
    let result = clickhouse.run_query_synchronous_no_params(query).await?;
    Ok(result.response.trim().to_string())
}

async fn check_index_exists(
    clickhouse: &ClickHouseConnectionInfo,
    table: &str,
    index: &str,
) -> Result<bool, Error> {
    // For sharded deployments, indices only exist on local tables, so we need to check the local table name
    let actual_table = if clickhouse.is_sharding_enabled() {
        clickhouse.get_local_table_name(table)
    } else {
        table.to_string()
    };
    
    let query = format!("SELECT 1 FROM system.data_skipping_indices WHERE database='{}' AND table='{}' AND name='{}'", clickhouse.database(), actual_table, index);
    let result = clickhouse.run_query_synchronous_no_params(query).await?;
    Ok(result.response.trim() == "1")
}

/// Shows the CREATE TABLE statement for a table, automatically handling sharding.
/// In sharded deployments, inspects the local table structure since that's where 
/// the actual table definition is stored.
async fn show_create_table(
    clickhouse: &ClickHouseConnectionInfo,
    table: &str,
) -> Result<String, Error> {
    // For table structure inspection, we need to check the local table in sharded deployments
    // since that's where the actual table definition/structure is stored
    let table_to_inspect = if clickhouse.is_sharding_enabled() {
        clickhouse.get_local_table_name(table)
    } else {
        table.to_string()
    };
    
    let query = format!("SHOW CREATE TABLE {}", table_to_inspect);
    let result = clickhouse.run_query_synchronous_no_params(query).await?;
    Ok(result.response.trim().to_string())
}
