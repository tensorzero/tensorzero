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
pub mod migration_0039;
pub mod migration_0040;
pub mod migration_0041;
pub mod migration_0042;

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
/// Errors if the query fails
async fn check_column_exists(
    clickhouse: &ClickHouseConnectionInfo,
    table: &str,
    column: &str,
    migration_id: &str,
) -> Result<bool, Error> {
    let query = format!(
        r"
            SELECT 1
            FROM system.columns
            WHERE database = '{}'
              AND table = '{}'
              AND name = '{}'
            LIMIT 1
        ",
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
