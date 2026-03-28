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

/// Advisory lock key for the deprecate_chain_of_thought migration.
/// Chosen as a fixed constant — must be unique across all background migrations.
const DEPRECATE_COT_LOCK_KEY: i64 = 0x545A_4243_4F54_0001; // "TZBCoT" + 0001

const DEPRECATE_COT_NAME: &str = "deprecate_chain_of_thought_v1";

/// Batch size for processing rows in background migrations.
const BATCH_SIZE: i64 = 100;

/// Run all registered background migrations.
/// Call this once after gateway startup (non-blocking — skips if another instance is running).
pub async fn run_all(pool: &PgPool) -> Result<(), Error> {
    run_if_needed(pool, DEPRECATE_COT_NAME, DEPRECATE_COT_LOCK_KEY, |pool| {
        Box::pin(migrate_chain_of_thought(pool))
    })
    .await?;

    Ok(())
}

/// Generic runner: checks completion, acquires advisory lock, runs migration, records result.
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

// ─── deprecate_chain_of_thought ──────────────────────────────────────────────

/// Rewrite `chain_of_thought` variant version rows to `chat_completion`.
///
/// For each row where `variant_type = 'chain_of_thought'`:
/// 1. Set `reasoning_effort = 'medium'` on the config row if neither `reasoning_effort`
///    nor `thinking_budget_tokens` is already set.
/// 2. Flip `variant_type` to `chat_completion`.
///
/// Processes in batches to avoid long transactions.
async fn migrate_chain_of_thought(pool: &PgPool) -> Result<i64, Error> {
    let mut total_rows: i64 = 0;

    loop {
        let rows: Vec<(uuid::Uuid,)> = sqlx::query_as(
            "SELECT id FROM tensorzero.variant_versions
             WHERE variant_type = 'chain_of_thought'
             LIMIT $1",
        )
        .bind(BATCH_SIZE)
        .fetch_all(pool)
        .await
        .map_err(|e| {
            Error::new(ErrorDetails::InternalError {
                message: format!("Failed to fetch chain_of_thought rows: {e}"),
            })
        })?;

        if rows.is_empty() {
            break;
        }

        let batch_size = rows.len() as i64;

        for (id,) in &rows {
            let mut tx = pool.begin().await?;

            // Set reasoning_effort = 'medium' if neither reasoning param is already set.
            sqlx::query(
                "UPDATE tensorzero.variant_chat_completion_configs
                 SET reasoning_effort = 'medium'
                 WHERE variant_version_id = $1
                   AND reasoning_effort IS NULL
                   AND thinking_budget_tokens IS NULL",
            )
            .bind(id)
            .execute(&mut *tx)
            .await?;

            // Flip the type.
            sqlx::query(
                "UPDATE tensorzero.variant_versions
                 SET variant_type = 'chat_completion'
                 WHERE id = $1 AND variant_type = 'chain_of_thought'",
            )
            .bind(id)
            .execute(&mut *tx)
            .await?;

            tx.commit().await?;
        }

        total_rows += batch_size;
        tracing::debug!(
            batch_size,
            total_rows,
            "Processed batch of chain_of_thought rows"
        );
    }

    Ok(total_rows)
}
