//! Background data migrations that run post-startup in Rust.
//!
//! These migrations rewrite historical rows to canonical forms (e.g., deprecating
//! variant types). They use Postgres advisory locks to ensure only one gateway
//! instance runs each migration, and a `background_migrations` table to track
//! completion so each migration runs exactly once.
//!
//! To add a new background migration:
//! 1. Add a constant name and advisory lock key below.
//! 2. Implement the migration as an `async fn(&PgPool) -> Result<i64, Error>` that
//!    returns the number of rows affected.
//! 3. Register it in `run_all`.

use sqlx::PgPool;

use crate::error::{Error, ErrorDetails};

/// Run all registered background migrations.
/// Call this once after gateway startup (non-blocking — skips if another instance is running).
#[expect(clippy::unused_async)]
pub async fn run_all(_pool: &PgPool) -> Result<(), Error> {
    // All background migrations have been completed and removed.
    // Register new migrations here as needed.
    Ok(())
}

/// Generic runner: checks completion, acquires advisory lock, runs migration, records result.
#[expect(dead_code)] // Will be used when a new background migration is added
async fn run_if_needed<F>(
    pool: &PgPool,
    name: &str,
    lock_key: i64,
    migration_fn: F,
) -> Result<(), Error>
where
    F: FnOnce(
        &PgPool,
    ) -> std::pin::Pin<
        Box<dyn std::future::Future<Output = Result<i64, Error>> + Send + '_>,
    >,
{
    // Check if already completed
    let completed = sqlx::query_scalar::<_, bool>(
        "SELECT completed_at IS NOT NULL FROM tensorzero.background_migrations WHERE name = $1",
    )
    .bind(name)
    .fetch_optional(pool)
    .await
    .map_err(|e| {
        Error::new(ErrorDetails::InternalError {
            message: format!("Failed to check background migration `{name}`: {e}"),
        })
    })?;

    if completed == Some(true) {
        tracing::debug!(
            migration = name,
            "Background migration already completed, skipping"
        );
        return Ok(());
    }

    // Try to acquire advisory lock (non-blocking, session-scoped)
    let acquired = sqlx::query_scalar::<_, bool>("SELECT pg_try_advisory_lock($1)")
        .bind(lock_key)
        .fetch_one(pool)
        .await
        .map_err(|e| {
            Error::new(ErrorDetails::InternalError {
                message: format!(
                    "Failed to acquire advisory lock for background migration `{name}`: {e}"
                ),
            })
        })?;

    if !acquired {
        tracing::info!(
            migration = name,
            "Another instance is running this background migration, skipping"
        );
        return Ok(());
    }

    // Double-check after acquiring lock (another instance may have completed between our check and lock)
    let completed = sqlx::query_scalar::<_, bool>(
        "SELECT completed_at IS NOT NULL FROM tensorzero.background_migrations WHERE name = $1",
    )
    .bind(name)
    .fetch_optional(pool)
    .await
    .map_err(|e| {
        Error::new(ErrorDetails::InternalError {
            message: format!("Failed to check background migration `{name}`: {e}"),
        })
    })?;

    if completed == Some(true) {
        release_advisory_lock(pool, lock_key).await;
        return Ok(());
    }

    // Record that we started (upsert in case a previous attempt crashed)
    sqlx::query(
        "INSERT INTO tensorzero.background_migrations (name, started_at)
         VALUES ($1, NOW())
         ON CONFLICT (name) DO UPDATE SET started_at = NOW()",
    )
    .bind(name)
    .execute(pool)
    .await
    .map_err(|e| {
        Error::new(ErrorDetails::InternalError {
            message: format!("Failed to record background migration start `{name}`: {e}"),
        })
    })?;

    tracing::info!(migration = name, "Starting background migration");

    // Run the migration
    let rows_affected = match migration_fn(pool).await {
        Ok(rows) => rows,
        Err(e) => {
            tracing::error!(migration = name, error = %e, "Background migration failed");
            release_advisory_lock(pool, lock_key).await;
            return Err(e);
        }
    };

    // Mark as completed
    sqlx::query(
        "UPDATE tensorzero.background_migrations
         SET completed_at = NOW(), rows_affected = $2
         WHERE name = $1",
    )
    .bind(name)
    .bind(rows_affected)
    .execute(pool)
    .await
    .map_err(|e| {
        Error::new(ErrorDetails::InternalError {
            message: format!("Failed to record background migration completion `{name}`: {e}"),
        })
    })?;

    tracing::info!(
        migration = name,
        rows_affected,
        "Background migration completed"
    );

    release_advisory_lock(pool, lock_key).await;
    Ok(())
}

async fn release_advisory_lock(pool: &PgPool, lock_key: i64) {
    let _ = sqlx::query("SELECT pg_advisory_unlock($1)")
        .bind(lock_key)
        .execute(pool)
        .await;
}
