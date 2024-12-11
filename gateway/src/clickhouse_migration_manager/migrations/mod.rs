use crate::{
    clickhouse::ClickHouseConnectionInfo,
    error::{Error, ErrorDetails},
};

pub mod migration_0000;
pub mod migration_0001;
pub mod migration_0002;
pub mod migration_0003;
pub mod migration_0004;
pub mod migration_0005;

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
    match clickhouse.run_query(query).await {
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
