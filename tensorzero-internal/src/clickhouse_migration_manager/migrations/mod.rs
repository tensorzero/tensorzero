use crate::{
    clickhouse::ClickHouseConnectionInfo,
    error::{Error, ErrorDetails},
};

pub mod migration_0000;
// pub mod migration_0001;
pub mod migration_0002;
pub mod migration_0003;
pub mod migration_0004;
pub mod migration_0005;
pub mod migration_0006;
// pub mod migration_0007;
pub mod migration_0008;
pub mod migration_0009;
// pub mod migration_0010;
pub mod migration_0011;
pub mod migration_0012;
pub mod migration_0013;

/// Returns true if the table exists, false if it does not
/// Errors if the query fails
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
    match clickhouse.run_query(query, None).await {
        Err(e) => {
            return Err(ErrorDetails::ClickHouseMigration {
                id: migration_id.to_string(),
                message: e.to_string(),
            }
            .into())
        }
        Ok(response) => {
            if response.trim() != "1" {
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
        r#"SELECT EXISTS(
            SELECT 1
            FROM system.columns
            WHERE database = '{}'
              AND table = '{}'
              AND name = '{}'
        )"#,
        clickhouse.database(),
        table,
        column,
    );
    match clickhouse.run_query(query, None).await {
        Err(e) => {
            return Err(ErrorDetails::ClickHouseMigration {
                id: migration_id.to_string(),
                message: e.to_string(),
            }
            .into())
        }
        Ok(response) => {
            if response.trim() != "1" {
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
    match clickhouse.run_query(query, None).await {
        Err(e) => Err(ErrorDetails::ClickHouseMigration {
            id: migration_id.to_string(),
            message: e.to_string(),
        }
        .into()),
        Ok(response) => Ok(response.trim().to_string()),
    }
}
