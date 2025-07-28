use crate::{
    clickhouse::ClickHouseConnectionInfo,
    error::{Error, ErrorDetails},
};

/// Creates the appropriate table engine based on replication configuration.
/// Returns a string ready to use in format! calls with actual replica identifiers.
pub fn create_table_engine(replication_enabled: bool, cluster_name: &str, table_name: &str) -> String {
    if replication_enabled {
        // Use {replica} macro which ClickHouse will replace with the actual replica name from config
        format!("ReplicatedMergeTree('/clickhouse/tables/{cluster_name}/{table_name}', '{{replica}}')")
    } else {
        "MergeTree()".to_string()
    }
}

/// Creates the appropriate ReplacingMergeTree engine based on replication configuration.
/// Returns a string ready to use in format! calls with actual replica identifiers.
pub fn create_replacing_table_engine(
    replication_enabled: bool, 
    cluster_name: &str, 
    table_name: &str, 
    params: Option<&str>
) -> String {
    if replication_enabled {
        if let Some(p) = params {
            format!("ReplicatedReplacingMergeTree('/clickhouse/tables/{cluster_name}/{table_name}', '{{replica}}', {p})")
        } else {
            format!("ReplicatedReplacingMergeTree('/clickhouse/tables/{cluster_name}/{table_name}', '{{replica}}')")
        }
    } else {
        if let Some(p) = params {
            format!("ReplacingMergeTree({p})")
        } else {
            "ReplacingMergeTree()".to_string()
        }
    }
}

/// Creates the appropriate ON CLUSTER clause for DDL operations when replication is enabled.
/// Note: Does not include leading space - add the space in your format string for readability.
pub fn create_cluster_clause(replication_enabled: bool, cluster_name: &str) -> String {
    if replication_enabled {
        format!("ON CLUSTER `{cluster_name}`")
    } else {
        String::new()
    }
}

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

/// Returns true if the table exists, false if it does not
/// Errors if the query fails
/// This function also works to check for materialized views
async fn check_table_exists(
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

/// Returns true if the column exists in the table, false if it does not
/// Errors if the query fails
async fn check_column_exists(
    clickhouse: &ClickHouseConnectionInfo,
    table: &str,
    column: &str,
    migration_id: &str,
) -> Result<bool, Error> {
    let query = format!(
        r"SELECT EXISTS(
            SELECT 1
            FROM system.columns
            WHERE database = '{}'
              AND table = '{}'
              AND name = '{}'
        )",
        clickhouse.database(),
        table,
        column,
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

async fn get_column_type(
    clickhouse: &ClickHouseConnectionInfo,
    table: &str,
    column: &str,
    migration_id: &str,
) -> Result<String, Error> {
    let query = format!(
        "SELECT type FROM system.columns WHERE database='{}' AND table='{}' AND name='{}'",
        clickhouse.database(),
        table,
        column
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

async fn get_default_expression(
    clickhouse: &ClickHouseConnectionInfo,
    table: &str,
    column: &str,
    migration_id: &str,
) -> Result<String, Error> {
    let query = format!(
        "SELECT default_expression FROM system.columns WHERE database='{}' AND table='{}' AND name='{}'",
        clickhouse.database(),
        table,
        column
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
    let query = format!("SELECT COUNT() FROM {table} FORMAT CSV");
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
    let query = format!(
        "SELECT engine FROM system.tables WHERE database='{}' AND name='{}'",
        clickhouse.database(),
        table
    );
    let result = clickhouse.run_query_synchronous_no_params(query).await?;
    Ok(result.response.trim().to_string())
}

async fn check_index_exists(
    clickhouse: &ClickHouseConnectionInfo,
    table: &str,
    index: &str,
) -> Result<bool, Error> {
    let query = format!("SELECT 1 FROM system.data_skipping_indices WHERE database='{}' AND table='{}' AND name='{}'", clickhouse.database(), table, index);
    let result = clickhouse.run_query_synchronous_no_params(query).await?;
    Ok(result.response.trim() == "1")
}
