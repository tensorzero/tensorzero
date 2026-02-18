use sqlx::AssertSqlSafe;
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
            message: "pg_cron extension is not installed".to_string(),
        }));
    }

    // Verify TensorZero's minute and hour latency refresh jobs are scheduled.
    let refresh_minute_job_exists: bool = sqlx::query_scalar(
        "SELECT EXISTS(SELECT 1 FROM cron.job WHERE jobname = 'tensorzero_refresh_model_latency_histogram_minute_incremental')",
    )
    .fetch_one(pool)
    .await
    .map_err(|e| {
        DelayedError::new(ErrorDetails::PostgresQuery {
            message: format!("Failed to check pg_cron jobs: {e}"),
        })
    })?;

    let refresh_hour_job_exists: bool = sqlx::query_scalar(
        "SELECT EXISTS(SELECT 1 FROM cron.job WHERE jobname = 'tensorzero_refresh_model_latency_histogram_hour_incremental')",
    )
    .fetch_one(pool)
    .await
    .map_err(|e| {
        DelayedError::new(ErrorDetails::PostgresQuery {
            message: format!("Failed to check pg_cron jobs: {e}"),
        })
    })?;

    if !(refresh_minute_job_exists && refresh_hour_job_exists) {
        return Err(DelayedError::new(ErrorDetails::PostgresMigration {
            message:
                "pg_cron extension is installed but TensorZero's pg_cron jobs are not scheduled."
                    .to_string(),
        }));
    }

    Ok(())
}

/// Checks whether pg_trgm extension is installed and trigram indexes exist on all target tables.
///
/// Returns a `DelayedError` so the caller can control logging level.
pub async fn check_trigram_indexes_configured_correctly(pool: &PgPool) -> Result<(), DelayedError> {
    let extension_exists: bool =
        sqlx::query_scalar("SELECT EXISTS(SELECT 1 FROM pg_extension WHERE extname = 'pg_trgm')")
            .fetch_one(pool)
            .await
            .map_err(|e| {
                DelayedError::new(ErrorDetails::PostgresQuery {
                    message: format!("Failed to check pg_trgm extension: {e}"),
                })
            })?;

    if !extension_exists {
        return Err(DelayedError::new(ErrorDetails::PostgresMigration {
            message: "pg_trgm extension is not installed".to_string(),
        }));
    }

    // Verify that a parent index exists for each table/column pair.
    for (table, columns) in TRIGRAM_INDEX_TARGETS {
        for column in *columns {
            let index_name = format!("idx_{table}_{column}_trgm");
            let exists: bool = sqlx::query_scalar(
                r"
                SELECT EXISTS(
                    SELECT 1 FROM pg_class c
                    JOIN pg_namespace ns ON c.relnamespace = ns.oid
                    WHERE c.relname = $1
                      AND ns.nspname = 'tensorzero'
                      AND c.relkind IN ('i', 'I')
                )
                ",
            )
            .bind(&index_name)
            .fetch_one(pool)
            .await
            .map_err(|e| {
                DelayedError::new(ErrorDetails::PostgresQuery {
                    message: format!("Failed to check trigram index `{index_name}`: {e}"),
                })
            })?;

            if !exists {
                return Err(DelayedError::new(ErrorDetails::PostgresMigration {
                    message: format!(
                        "Trigram index `{index_name}` is missing on `{table}.{column}`"
                    ),
                }));
            }
        }
    }

    Ok(())
}

// --- Trigram index setup ---
//
// Tables and columns that should have trigram GIN indexes for efficient `ILIKE '%substring%'` queries.
// The index expression `(column)::text` matches queries using either `CAST(col AS TEXT)` or `col::TEXT`
// since Postgres normalizes both to the same internal representation.
//
// Table and column names below are trusted constants used in dynamic SQL via `format!`.

/// Tables and the JSONB columns on each that should have trigram indexes.
const TRIGRAM_INDEX_TARGETS: &[(&str, &[&str])] = &[
    ("chat_inference_data", &["input", "output"]),
    ("json_inference_data", &["input", "output"]),
    ("chat_datapoints", &["input", "output"]),
    ("json_datapoints", &["input", "output"]),
];

/// Attempts to set up pg_trgm extension and create trigram GIN indexes.
/// This is idempotent and safe to run multiple times.
///
/// For partitioned tables, indexes are created `CONCURRENTLY` on each partition
/// (no table lock), then attached to a parent index. For non-partitioned tables,
/// indexes are created `CONCURRENTLY` directly.
///
/// The function logs warnings if pg_trgm is not available, but does not fail.
pub async fn setup_trigram_indexes(pool: &PgPool) -> Result<(), Error> {
    if let Err(e) = sqlx::raw_sql("CREATE EXTENSION IF NOT EXISTS pg_trgm")
        .execute(pool)
        .await
    {
        tracing::warn!("pg_trgm extension could not be created: {e}");
        return Ok(());
    }

    let available: bool =
        sqlx::query_scalar("SELECT EXISTS(SELECT 1 FROM pg_extension WHERE extname = 'pg_trgm')")
            .fetch_one(pool)
            .await
            .map_err(|e| {
                Error::new(ErrorDetails::PostgresQuery {
                    message: format!("Failed to check pg_trgm extension: {e}"),
                })
            })?;

    if !available {
        tracing::warn!("pg_trgm extension not available — trigram indexes not created");
        return Ok(());
    }

    for (table, columns) in TRIGRAM_INDEX_TARGETS {
        for column in *columns {
            create_trigram_index_for_table(pool, table, column).await?;
        }
    }

    tracing::info!("Trigram index setup complete");
    Ok(())
}

/// Returns the names of all direct child partitions of `table` in the `tensorzero` schema.
async fn get_partitions(pool: &PgPool, table: &str) -> Result<Vec<String>, Error> {
    sqlx::query_scalar(
        r"
        SELECT child.relname::TEXT
        FROM pg_inherits
        JOIN pg_class child ON pg_inherits.inhrelid = child.oid
        JOIN pg_class parent ON pg_inherits.inhparent = parent.oid
        JOIN pg_namespace ns ON parent.relnamespace = ns.oid
        WHERE parent.relname = $1
          AND ns.nspname = 'tensorzero'
        ORDER BY child.relname
        ",
    )
    .bind(table)
    .fetch_all(pool)
    .await
    .map_err(|e| {
        Error::new(ErrorDetails::PostgresQuery {
            message: format!("Failed to query partitions for `{table}`: {e}"),
        })
    })
}

/// Returns whether a given index (by name in `tensorzero` schema) is already
/// attached as a partition of some parent index via `pg_inherits`.
async fn is_index_attached(pool: &PgPool, index_name: &str) -> Result<bool, Error> {
    sqlx::query_scalar(
        r"
        SELECT EXISTS(
            SELECT 1 FROM pg_inherits
            JOIN pg_class child ON pg_inherits.inhrelid = child.oid
            JOIN pg_namespace ns ON child.relnamespace = ns.oid
            WHERE child.relname = $1
              AND ns.nspname = 'tensorzero'
        )
        ",
    )
    .bind(index_name)
    .fetch_one(pool)
    .await
    .map_err(|e| {
        Error::new(ErrorDetails::PostgresQuery {
            message: format!("Failed to check index attachment for `{index_name}`: {e}"),
        })
    })
}

/// Creates a trigram GIN index on `table.column`, handling partitioned tables by
/// creating indexes concurrently on each partition and attaching them to a parent index.
async fn create_trigram_index_for_table(
    pool: &PgPool,
    table: &str,
    column: &str,
) -> Result<(), Error> {
    let partitions = get_partitions(pool, table).await?;

    if partitions.is_empty() {
        // Non-partitioned table: create index concurrently
        let index_name = format!("idx_{table}_{column}_trgm");
        let sql = format!(
            "CREATE INDEX CONCURRENTLY IF NOT EXISTS {index_name} \
             ON tensorzero.{table} USING GIN (CAST({column} AS TEXT) gin_trgm_ops)"
        );
        sqlx::raw_sql(AssertSqlSafe(sql))
            .execute(pool)
            .await
            .map_err(|e| {
                Error::new(ErrorDetails::PostgresQuery {
                    message: format!(
                        "Failed to create trigram index `{index_name}` on `{table}`: {e}"
                    ),
                })
            })?;
        return Ok(());
    }

    // Partitioned table: create indexes on each partition concurrently (no lock)
    create_and_attach_partition_indexes(pool, table, column, &partitions).await?;

    // Re-check for partitions that may have been created while we were working.
    // `ON ONLY` doesn't recurse to existing partitions, so any partition created
    // between our initial get_partitions and CREATE INDEX ON ONLY would be missed.
    // Once the parent index exists, new partitions created *after* this point get
    // auto-indexed by Postgres.
    let current_partitions = get_partitions(pool, table).await?;
    let new_partitions: Vec<&String> = current_partitions
        .iter()
        .filter(|p| !partitions.contains(p))
        .collect();

    if !new_partitions.is_empty() {
        tracing::info!(
            "Found {} new partition(s) for `{table}` created during index setup, indexing them",
            new_partitions.len()
        );
        for partition in &new_partitions {
            create_partition_index_concurrently(pool, table, column, partition).await?;
        }
        attach_partition_indexes(pool, table, column, current_partitions.as_ref()).await?;
    }

    Ok(())
}

/// Creates indexes concurrently on each partition, creates the parent index with
/// `ON ONLY`, and attaches partition indexes to the parent.
async fn create_and_attach_partition_indexes(
    pool: &PgPool,
    table: &str,
    column: &str,
    partitions: &[String],
) -> Result<(), Error> {
    for partition in partitions {
        create_partition_index_concurrently(pool, table, column, partition).await?;
    }

    // Create parent index with ONLY (no recursion to partitions, no data to scan)
    // TODO(shuyangli): this is not quoted, so special characters (uppercase, hyphens, etc) could break it;
    // we control these names so it's not a big deal today, but consider quoting this in the future.
    let parent_index = format!("idx_{table}_{column}_trgm");
    let sql = format!(
        "CREATE INDEX IF NOT EXISTS {parent_index} \
         ON ONLY tensorzero.{table} USING GIN (CAST({column} AS TEXT) gin_trgm_ops)"
    );
    sqlx::raw_sql(AssertSqlSafe(sql))
        .execute(pool)
        .await
        .map_err(|e| {
            Error::new(ErrorDetails::PostgresQuery {
                message: format!(
                    "Failed to create parent trigram index `{parent_index}` on `{table}`: {e}"
                ),
            })
        })?;

    attach_partition_indexes(pool, table, column, partitions).await
}

/// Creates a trigram index concurrently on a single partition.
async fn create_partition_index_concurrently(
    pool: &PgPool,
    table: &str,
    column: &str,
    partition: &str,
) -> Result<(), Error> {
    // TODO(shuyangli): this is not quoted, so special characters (uppercase, hyphens, etc) could break it;
    // we control these names so it's not a big deal today, but consider quoting this in the future.
    let index_name = format!("idx_{partition}_{column}_trgm");
    let sql = format!(
        "CREATE INDEX CONCURRENTLY IF NOT EXISTS {index_name} \
         ON tensorzero.{partition} USING GIN (CAST({column} AS TEXT) gin_trgm_ops)"
    );
    sqlx::raw_sql(AssertSqlSafe(sql))
        .execute(pool)
        .await
        .map_err(|e| {
            Error::new(ErrorDetails::PostgresQuery {
                message: format!(
                    "Failed to create trigram index `{index_name}` on partition `{partition}` of `{table}`: {e}"
                ),
            })
        })?;
    Ok(())
}

/// Attaches any partition indexes that aren't yet attached to the parent index.
async fn attach_partition_indexes(
    pool: &PgPool,
    table: &str,
    column: &str,
    partitions: &[String],
) -> Result<(), Error> {
    let parent_index = format!("idx_{table}_{column}_trgm");
    for partition in partitions {
        let partition_index = format!("idx_{partition}_{column}_trgm");
        if is_index_attached(pool, &partition_index).await? {
            continue;
        }
        let sql = format!(
            "ALTER INDEX tensorzero.{parent_index} \
             ATTACH PARTITION tensorzero.{partition_index}"
        );
        sqlx::raw_sql(AssertSqlSafe(sql))
            .execute(pool)
            .await
            .map_err(|e| {
                Error::new(ErrorDetails::PostgresQuery {
                    message: format!(
                        "Failed to attach `{partition_index}` to `{parent_index}`: {e}"
                    ),
                })
            })?;
    }
    Ok(())
}
