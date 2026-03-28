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

use sqlx::{PgPool, Row};

use crate::config::variant_versions::{StoredVariantConfig, deserialize_stored_variant_version};
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
        // Release lock and return
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
/// Loads all variant_versions rows in batches, deserializes each via serde,
/// and rewrites any `ChainOfThought` variants as `ChatCompletion` with
/// `reasoning_effort` defaulted to `"medium"`. Rows that are already
/// `ChatCompletion` (or any other type) are skipped without modification.
///
/// All filtering happens in typed Rust — no raw JSONB queries.
async fn migrate_chain_of_thought(pool: &PgPool) -> Result<i64, Error> {
    let mut total_migrated: i64 = 0;
    let mut last_id: Option<uuid::Uuid> = None;

    loop {
        // Fetch a batch of rows, paginated by id (UUIDv7 = time-ordered)
        let rows = if let Some(cursor) = last_id {
            sqlx::query(
                "SELECT id, schema_version, config
                 FROM tensorzero.variant_versions
                 WHERE id > $1
                 ORDER BY id
                 LIMIT $2",
            )
            .bind(cursor)
            .bind(BATCH_SIZE)
            .fetch_all(pool)
            .await
        } else {
            sqlx::query(
                "SELECT id, schema_version, config
                 FROM tensorzero.variant_versions
                 ORDER BY id
                 LIMIT $1",
            )
            .bind(BATCH_SIZE)
            .fetch_all(pool)
            .await
        }
        .map_err(|e| {
            Error::new(ErrorDetails::InternalError {
                message: format!("Failed to fetch variant_versions rows: {e}"),
            })
        })?;

        if rows.is_empty() {
            break;
        }

        for row in &rows {
            let id: uuid::Uuid = row.get("id");
            last_id = Some(id);

            let schema_version: i32 = row.get("schema_version");
            let config: serde_json::Value = row
                .get::<sqlx::types::Json<serde_json::Value>, _>("config")
                .0;

            // Deserialize using the standard serde path
            let mut stored = deserialize_stored_variant_version(schema_version, config)?;

            // Only transform ChainOfThought rows — skip everything else
            let StoredVariantConfig::ChainOfThought(cc_config) = stored.config else {
                continue;
            };

            let mut new_config = cc_config;
            if new_config.reasoning_effort.is_none() {
                new_config.reasoning_effort = Some("medium".to_string());
            }
            stored.config = StoredVariantConfig::ChatCompletion(new_config);

            // Serialize back via serde and write
            let new_json = serde_json::to_value(&stored).map_err(|e| {
                Error::new(ErrorDetails::Serialization {
                    message: format!("Failed to serialize migrated variant version `{id}`: {e}"),
                })
            })?;

            sqlx::query(
                "UPDATE tensorzero.variant_versions
                 SET config = $2
                 WHERE id = $1",
            )
            .bind(id)
            .bind(sqlx::types::Json(&new_json))
            .execute(pool)
            .await
            .map_err(|e| {
                Error::new(ErrorDetails::InternalError {
                    message: format!("Failed to update migrated variant version `{id}`: {e}"),
                })
            })?;

            total_migrated += 1;
        }

        tracing::debug!(total_migrated, "Processed batch of variant_versions rows");
    }

    Ok(total_migrated)
}
