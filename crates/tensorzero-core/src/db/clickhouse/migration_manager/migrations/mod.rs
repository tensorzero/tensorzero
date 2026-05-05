use std::time::Duration;

use crate::{
    db::clickhouse::ClickHouseConnectionInfo,
    error::{ErrorDetails, delayed_error::DelayedError},
};

/// Default offset for materialized view recreation during migrations.
///
/// When a migration drops and recreates a MV, there's a window where inserts
/// are not captured. The MV is created with `WHERE id >= T` (where T = now + offset),
/// and the backfill covers `WHERE id < T`. We must wait until T has passed before
/// running the backfill so that no new rows can have `id < T`.
///
/// The offset only needs to exceed the time to drop + create the MV (typically < 1s,
/// up to a few seconds on ClickHouse Cloud with replication).
const VIEW_OFFSET: Duration = Duration::from_secs(5);

/// Tracks a future deadline for MV backfill operations.
///
/// Created before the MV drop/recreate work starts. Call [`wait`](Self::wait)
/// after the MV is recreated — it only sleeps the remaining time until the
/// deadline, not the full offset duration.
pub(crate) struct ViewOffsetDeadline {
    deadline: tokio::time::Instant,
}

impl ViewOffsetDeadline {
    /// Creates a new deadline `VIEW_OFFSET` seconds from now.
    pub(crate) fn new() -> Self {
        Self {
            deadline: tokio::time::Instant::now() + VIEW_OFFSET,
        }
    }

    /// Waits until the deadline has passed. If the deadline is already in the
    /// past (because the MV work took longer than the offset), returns immediately.
    pub(crate) async fn wait(self) {
        tokio::time::sleep_until(self.deadline).await;
    }

    /// Returns the offset duration (for computing the future timestamp).
    pub(crate) fn offset() -> Duration {
        VIEW_OFFSET
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
pub mod migration_0035;
pub mod migration_0036;
pub mod migration_0037;
pub mod migration_0038;
pub mod migration_0039;
pub mod migration_0040;
pub mod migration_0041;
pub mod migration_0042;
pub mod migration_0043;
pub mod migration_0044;
pub mod migration_0045;
pub mod migration_0046;
pub mod migration_0047;
pub mod migration_0048;
pub mod migration_0049;
pub mod migration_0050;
pub mod migration_0051;
pub mod migration_0052;
pub mod migration_0053;
pub mod migration_0054;

/// Returns true if the table exists, false if it does not
/// Errors if the query fails
/// This function also works to check for materialized views
pub async fn check_table_exists(
    clickhouse: &ClickHouseConnectionInfo,
    table: &str,
    migration_id: &str,
) -> Result<bool, DelayedError> {
    let query = format!(
        "SELECT 1 FROM system.tables WHERE database = '{}' AND name = '{}'",
        clickhouse.database(),
        table
    );
    match clickhouse
        .run_query_synchronous_no_params_delayed_err(query)
        .await
    {
        Err(e) => {
            return Err(DelayedError::new(ErrorDetails::ClickHouseMigration {
                id: migration_id.to_string(),
                message: e.suppress_logging_of_error_message().to_string(),
            }));
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
) -> Result<bool, DelayedError> {
    let query = format!(
        "SELECT 1 FROM system.detached_tables WHERE database = '{}' AND table = '{}'",
        clickhouse.database(),
        table
    );
    match clickhouse
        .run_query_synchronous_no_params_delayed_err(query)
        .await
    {
        Err(e) => {
            return Err(DelayedError::new(ErrorDetails::ClickHouseMigration {
                id: migration_id.to_string(),
                message: e.suppress_logging_of_error_message().to_string(),
            }));
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
) -> Result<bool, DelayedError> {
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
    match clickhouse
        .run_query_synchronous_no_params_delayed_err(query)
        .await
    {
        Err(e) => {
            return Err(DelayedError::new(ErrorDetails::ClickHouseMigration {
                id: migration_id.to_string(),
                message: e.suppress_logging_of_error_message().to_string(),
            }));
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
) -> Result<String, DelayedError> {
    let query = format!(
        "SELECT type FROM system.columns WHERE database='{}' AND table='{}' AND name='{}'",
        clickhouse.database(),
        table,
        column
    );
    match clickhouse
        .run_query_synchronous_no_params_delayed_err(query)
        .await
    {
        Err(e) => Err(DelayedError::new(ErrorDetails::ClickHouseMigration {
            id: migration_id.to_string(),
            message: e.suppress_logging_of_error_message().to_string(),
        })),
        Ok(response) => Ok(response.response.trim().to_string()),
    }
}

async fn get_default_expression(
    clickhouse: &ClickHouseConnectionInfo,
    table: &str,
    column: &str,
    migration_id: &str,
) -> Result<String, DelayedError> {
    let query = format!(
        "SELECT default_expression FROM system.columns WHERE database='{}' AND table='{}' AND name='{}'",
        clickhouse.database(),
        table,
        column
    );
    match clickhouse
        .run_query_synchronous_no_params_delayed_err(query)
        .await
    {
        Err(e) => Err(DelayedError::new(ErrorDetails::ClickHouseMigration {
            id: migration_id.to_string(),
            message: e.suppress_logging_of_error_message().to_string(),
        })),
        Ok(response) => Ok(response.response.trim().to_string()),
    }
}

async fn table_is_nonempty(
    clickhouse: &ClickHouseConnectionInfo,
    table: &str,
    migration_id: &str,
) -> Result<bool, DelayedError> {
    let query = format!("SELECT COUNT() FROM {table} FORMAT CSV");
    let result = clickhouse
        .run_query_synchronous_no_params_delayed_err(query)
        .await?;
    Ok(result.response.trim().parse::<i64>().map_err(|e| {
        DelayedError::new(ErrorDetails::ClickHouseMigration {
            id: migration_id.to_string(),
            message: e.to_string(),
        })
    })? > 0)
}

async fn get_table_engine(
    clickhouse: &ClickHouseConnectionInfo,
    table: &str,
) -> Result<String, DelayedError> {
    let query = format!(
        "SELECT engine FROM system.tables WHERE database='{}' AND name='{}'",
        clickhouse.database(),
        table
    );
    let result = clickhouse
        .run_query_synchronous_no_params_delayed_err(query)
        .await?;
    Ok(result.response.trim().to_string())
}

async fn check_index_exists(
    clickhouse: &ClickHouseConnectionInfo,
    table: &str,
    index: &str,
) -> Result<bool, DelayedError> {
    let query = format!(
        "SELECT 1 FROM system.data_skipping_indices WHERE database='{}' AND table='{}' AND name='{}'",
        clickhouse.database(),
        table,
        index
    );
    let result = clickhouse
        .run_query_synchronous_no_params_delayed_err(query)
        .await?;
    Ok(result.response.trim() == "1")
}

/// Submits a `MATERIALIZE INDEX` mutation asynchronously (with `mutations_sync=0` to override
/// the connection-level `mutations_sync=2`), then polls `system.mutations` until the table
/// has no pending mutations.
///
/// This avoids holding a blocking HTTP connection for the entire duration of the mutation,
/// which can cause timeouts and lock contention when multiple gateway instances start concurrently.
pub async fn materialize_index(
    clickhouse: &ClickHouseConnectionInfo,
    table: &str,
    index: &str,
) -> Result<(), DelayedError> {
    // Submit the mutation asynchronously by overriding mutations_sync at the query level.
    // The connection-level mutations_sync=2 is overridden by the SETTINGS clause.
    let query =
        format!("ALTER TABLE {table} MATERIALIZE INDEX {index} SETTINGS mutations_sync = 0");
    clickhouse
        .run_query_synchronous_no_params_delayed_err(query)
        .await?;

    // Poll system.mutations until no pending mutations remain for this table.
    let poll_interval = Duration::from_millis(100);
    let timeout = Duration::from_secs(300);
    let start = std::time::Instant::now();

    loop {
        let query = format!(
            "SELECT count() FROM system.mutations WHERE database = currentDatabase() AND table = '{table}' AND is_done = 0 FORMAT CSV"
        );
        let result = clickhouse
            .run_query_synchronous_no_params_delayed_err(query)
            .await?;

        let pending: i64 = result.response.trim().parse().map_err(|e| {
            DelayedError::new(ErrorDetails::ClickHouseMigration {
                id: "materialize_index".to_string(),
                message: format!("Failed to parse pending mutation count for table `{table}`: {e}"),
            })
        })?;

        if pending == 0 {
            return Ok(());
        }

        if start.elapsed() > timeout {
            return Err(DelayedError::new(ErrorDetails::ClickHouseMigration {
                id: "materialize_index".to_string(),
                message: format!(
                    "Timed out waiting for MATERIALIZE INDEX `{index}` on table `{table}` to complete ({pending} mutations still pending after {}s)",
                    timeout.as_secs()
                ),
            }));
        }

        tokio::time::sleep(poll_interval).await;
    }
}
