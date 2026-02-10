use sqlx::PgPool;

use crate::error::{DelayedError, Error, ErrorDetails};

/// The SQL script for setting up pg_cron. This is embedded at compile time.
const PGCRON_SETUP_SQL: &str = include_str!("pgcron_setup.sql");

/// Attempts to set up pg_cron extension and schedule partition management jobs.
/// This is idempotent and safe to run multiple times.
///
/// The function logs warnings if pg_cron is not available, but does not fail.
/// The gateway will later validate pg_cron availability and error if required.
pub async fn setup_pgcron(pool: &PgPool) -> Result<(), Error> {
    // Use raw_sql for multi-statement execution without prepared statements
    sqlx::raw_sql(PGCRON_SETUP_SQL).execute(pool).await?;

    Ok(())
}

/// Checks whether pg_cron extension is installed and TensorZero's jobs are scheduled.
///
/// Returns a `DelayedError` so the caller can control logging level.
pub async fn check_pgcron_configured_correctly(pool: &PgPool) -> Result<(), DelayedError> {
    let extension_exists: bool =
        sqlx::query_scalar("SELECT EXISTS(SELECT 1 FROM pg_extension WHERE extname = 'pg_cron')")
            .fetch_one(pool)
            .await
            .map_err(|e| {
                DelayedError::new(ErrorDetails::PostgresQuery {
                    message: format!("Failed to check pg_cron extension: {e}"),
                })
            })?;

    if !extension_exists {
        return Err(DelayedError::new(ErrorDetails::PostgresMigration {
            message: "pg_cron extension is not installed.".to_string(),
        }));
    }

    // Verify the materialized view refresh job is scheduled, as an example of a TensorZero-specific job.
    let refresh_job_exists: bool = sqlx::query_scalar(
        "SELECT EXISTS(SELECT 1 FROM cron.job WHERE jobname = 'tensorzero_refresh_materialized_views')",
    )
    .fetch_one(pool)
    .await
    .map_err(|e| {
        DelayedError::new(ErrorDetails::PostgresQuery {
            message: format!("Failed to check pg_cron jobs: {e}"),
        })
    })?;

    if !refresh_job_exists {
        return Err(DelayedError::new(ErrorDetails::PostgresMigration {
            message:
                "pg_cron extension is installed but TensorZero's pg_cron jobs are not scheduled."
                    .to_string(),
        }));
    }

    Ok(())
}
