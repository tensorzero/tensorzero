use std::sync::Arc;
use std::{collections::HashSet, time::Duration};

use async_trait::async_trait;
use durable;
use futures::TryStreamExt;
use sqlx::{PgPool, Row, migrate, postgres::PgPoolOptions};
use tokio::time::timeout;

use crate::error::{Error, ErrorDetails};

use self::batching::{PostgresBatchSender, PostgresBatchWriterHandle};
use super::HealthCheckable;

pub mod batch_inference;
pub mod batching;
pub mod config_queries;
pub mod dataset_queries;
pub mod deployment_queries;
pub mod dicl_queries;
pub mod evaluation_queries;
pub mod experimentation;
pub mod feedback;
mod howdy_queries;
pub mod inference_queries;
pub mod model_inferences;
pub mod pgcron;
pub mod rate_limiting;
mod resolve_uuid;
pub mod workflow_evaluation_queries;

mod episode_queries;
mod inference_filter_helpers;

#[cfg(any(test, feature = "e2e_tests"))]
pub mod test_helpers;

const RUN_MIGRATIONS_COMMAND: &str = "You likely need to apply migrations to your Postgres database with `--run-postgres-migrations`. Please see our documentation to learn more: https://www.tensorzero.com/docs/deployment/postgres";

#[derive(Debug, Clone)]
pub enum PostgresConnectionInfo {
    Enabled {
        pool: PgPool,
        batch_sender: Option<Arc<PostgresBatchSender>>,
    },
    #[cfg(test)]
    Mock {
        healthy: bool,
    },
    Disabled,
}

impl PostgresConnectionInfo {
    pub fn new_with_pool(pool: PgPool) -> Self {
        Self::Enabled {
            pool,
            batch_sender: None,
        }
    }

    pub fn new_with_pool_and_batcher(pool: PgPool, batch_sender: Arc<PostgresBatchSender>) -> Self {
        Self::Enabled {
            pool,
            batch_sender: Some(batch_sender),
        }
    }

    #[cfg(test)]
    pub fn new_mock(healthy: bool) -> Self {
        Self::Mock { healthy }
    }

    pub fn new_disabled() -> Self {
        Self::Disabled
    }

    pub fn get_pool(&self) -> Option<&PgPool> {
        match self {
            Self::Enabled { pool, .. } => Some(pool),
            #[cfg(test)]
            Self::Mock { .. } => None,
            Self::Disabled => None,
        }
    }

    pub fn get_pool_result(&self) -> Result<&PgPool, Error> {
        match self {
            Self::Enabled { pool, .. } => Ok(pool),
            #[cfg(test)]
            Self::Mock { .. } => Err(Error::new(ErrorDetails::PostgresConnectionInitialization {
                message: "Mock database is not supported".to_string(),
            })),
            Self::Disabled => Err(Error::new(ErrorDetails::PostgresConnectionInitialization {
                message: "Database is disabled".to_string(),
            })),
        }
    }

    pub fn batch_sender(&self) -> Option<&Arc<PostgresBatchSender>> {
        match self {
            Self::Enabled { batch_sender, .. } => batch_sender.as_ref(),
            #[cfg(test)]
            Self::Mock { .. } => None,
            Self::Disabled => None,
        }
    }

    pub fn batcher_join_handle(&self) -> Option<PostgresBatchWriterHandle> {
        match self {
            Self::Enabled { batch_sender, .. } => {
                batch_sender.as_ref().map(|s| s.writer_handle.clone())
            }
            #[cfg(test)]
            Self::Mock { .. } => None,
            Self::Disabled => None,
        }
    }

    /// If the connection is active, check that the set of migrations that have succeeded matches the expected set of migrations.
    /// If the connection is not active, return Ok(()).
    pub async fn check_migrations(&self) -> Result<(), Error> {
        if let Some(pool) = self.get_pool() {
            let migrator = make_migrator();
            let expected_migrations: HashSet<i64> = migrator.iter().map(|m| m.version).collect();
            // Query the database for all successfully applied migration versions.
            let applied_migrations = get_applied_migrations(pool).await.map_err(|e| {
                Error::new(ErrorDetails::PostgresConnectionInitialization {
                    message: format!(
                        "Failed to retrieve applied `tensorzero-core` migrations: {e}. {RUN_MIGRATIONS_COMMAND}"
                    ),
                })
            })?;

            Self::check_applied_expected(
                "tensorzero-core",
                &applied_migrations,
                &expected_migrations,
            )?;

            let tensorzero_auth_migrations_data =
                tensorzero_auth::postgres::get_migrations_data(pool)
                    .await
                    .map_err(|e| {
                        Error::new(ErrorDetails::PostgresConnectionInitialization {
                            message: format!(
                                "Failed to retrieve applied `tensorzero-auth` migrations: {e}. {RUN_MIGRATIONS_COMMAND}"
                            ),
                        })
                    })?;
            Self::check_applied_expected(
                "tensorzero-auth",
                &tensorzero_auth_migrations_data.applied,
                &tensorzero_auth_migrations_data.expected,
            )?;
        }

        Ok(())
    }

    /// Checks that the set of applied migrations is acceptable for the current gateway version (which contains an expected set of migrations).
    ///
    /// During rolling upgrades, the older gateway version will see new database schema, which is expected (so far everything has been backwards
    /// compatible). However, if the database doesn't contain some migrations the current gateway version requires, we should fail the startup.
    ///
    /// TODO(shuyangli): When we need to make backwards incompatible schema changes, properly support expand-contract.
    fn check_applied_expected(
        name: &str,
        applied_migrations: &HashSet<i64>,
        expected_migrations: &HashSet<i64>,
    ) -> Result<(), Error> {
        if expected_migrations.is_subset(applied_migrations) {
            // If expected migrations (what this gateway version expects) is a subset of applied migrations, we are okay - during rolling upgrades
            // or with optional features, this is expected.
            return Ok(());
        }

        let missing_migrations = expected_migrations
            .difference(applied_migrations)
            .collect::<Vec<_>>();
        Err(Error::new(ErrorDetails::PostgresConnectionInitialization {
            message: format!(
                "Applied `{name}` migrations do not match expected migrations: {missing_migrations:?} are missing from the database. {RUN_MIGRATIONS_COMMAND}"
            ),
        }))
    }

    /// Writes retention configuration to the `tensorzero.retention_config` table.
    /// This is called on gateway startup to sync config from tensorzero.toml to Postgres.
    pub async fn write_retention_config(
        &self,
        inference_metadata_retention_days: Option<u32>,
        inference_data_retention_days: Option<u32>,
    ) -> Result<(), Error> {
        let Some(pool) = self.get_pool() else {
            return Ok(());
        };

        // Clean up the legacy key (replaced by the two keys below)
        sqlx::query!(
            "DELETE FROM tensorzero.retention_config WHERE key = 'inference_retention_days'"
        )
        .execute(pool)
        .await
        .map_err(|e| {
            Error::new(ErrorDetails::PostgresQuery {
                message: format!("Failed to delete legacy `inference_retention_days` config: {e}"),
            })
        })?;

        Self::upsert_retention_key(
            pool,
            "inference_metadata_retention_days",
            inference_metadata_retention_days,
        )
        .await?;
        Self::upsert_retention_key(
            pool,
            "inference_data_retention_days",
            inference_data_retention_days,
        )
        .await?;

        tracing::info!(
            inference_metadata_retention_days,
            inference_data_retention_days,
            "Configured inference retention policy"
        );

        Ok(())
    }

    async fn upsert_retention_key(
        pool: &sqlx::PgPool,
        key: &str,
        value: Option<u32>,
    ) -> Result<(), Error> {
        match value {
            Some(days) => {
                sqlx::query!(
                    r"
                    INSERT INTO tensorzero.retention_config (key, value, updated_at)
                    VALUES ($1, $2, NOW())
                    ON CONFLICT (key) DO UPDATE SET value = $2, updated_at = NOW()
                    ",
                    key,
                    days.to_string(),
                )
                .execute(pool)
                .await
                .map_err(|e| {
                    Error::new(ErrorDetails::PostgresQuery {
                        message: format!("Failed to write `{key}` config: {e}"),
                    })
                })?;
            }
            None => {
                sqlx::query!(
                    "DELETE FROM tensorzero.retention_config WHERE key = $1",
                    key
                )
                .execute(pool)
                .await
                .map_err(|e| {
                    Error::new(ErrorDetails::PostgresQuery {
                        message: format!("Failed to clear `{key}` config: {e}"),
                    })
                })?;
            }
        }
        Ok(())
    }
}

#[async_trait]
impl HealthCheckable for PostgresConnectionInfo {
    async fn health(&self) -> Result<(), Error> {
        match self {
            Self::Disabled => Ok(()),
            #[cfg(test)]
            Self::Mock { healthy } => {
                if *healthy {
                    Ok(())
                } else {
                    Err(Error::new(ErrorDetails::PostgresConnection {
                        message: "Unhealthy mock postgres connection".to_string(),
                    }))
                }
            }
            Self::Enabled { pool, .. } => {
                let check = async {
                    let _result = sqlx::query("SELECT 1").fetch_one(pool).await.map_err(|e| {
                        Error::new(ErrorDetails::PostgresConnection {
                            message: e.to_string(),
                        })
                    })?;

                    Ok(())
                };

                // TODO(shuyang): customize postgres timeout
                match timeout(Duration::from_millis(1000), check).await {
                    Ok(Ok(())) => Ok(()),
                    Ok(Err(e)) => Err(e),
                    Err(_) => Err(Error::new(ErrorDetails::PostgresConnection {
                        message: "Postgres healthcheck query timed out.".to_string(),
                    })),
                }
            }
        }
    }
}

/// Runs the migrations defined in this crate, and in `tensorzero-auth`
/// This uses two independent `sqlx::Migrator` instances, which write to separate migration tables.
/// The `tensorzero-core` migrator in this crate uses the default `_sqlx_migrations` table,
/// while the `tensorzero-auth` migrator uses `tensorzero_auth__sqlx_migrations`.
pub async fn manual_run_postgres_migrations() -> Result<(), Error> {
    let postgres_url = std::env::var("TENSORZERO_POSTGRES_URL").map_err(|_| {
        Error::new(ErrorDetails::PostgresConnectionInitialization {
            message: "Failed to read TENSORZERO_POSTGRES_URL environment variable".to_string(),
        })
    })?;
    manual_run_postgres_migrations_with_url(&postgres_url).await
}

pub async fn manual_run_postgres_migrations_with_url(postgres_url: &str) -> Result<(), Error> {
    let pool = PgPoolOptions::new()
        .connect(postgres_url)
        .await
        .map_err(|err| {
            Error::new(ErrorDetails::PostgresConnectionInitialization {
                message: err.to_string(),
            })
        })?;
    // Run tensorzero-auth migrations
    tensorzero_auth::postgres::make_migrator()
        .run(&pool)
        .await
        .map_err(|e| {
            Error::new(ErrorDetails::PostgresMigration {
                message: format!("Failed to run tensorzero-auth migrations: {e}"),
            })
        })?;

    // Run durable migrations to create the durable schema,
    // which is required by some tensorzero-core migrations.
    durable::MIGRATOR.run(&pool).await.map_err(|e| {
        Error::new(ErrorDetails::PostgresMigration {
            message: format!("Failed to run durable migrations: {e}"),
        })
    })?;
    make_migrator().run(&pool).await.map_err(|e| {
        Error::new(ErrorDetails::PostgresMigration {
            message: e.to_string(),
        })
    })?;

    // Try to set up pg_cron extension and schedule partition management jobs.
    // This is idempotent and runs every time.
    pgcron::setup_pgcron(&pool).await?;

    // Verify pg_cron is available
    // TODO(#6176): Once we promote pgcron_setup.sql to a migration, we can remove this check.
    if let Err(e) = pgcron::check_pgcron_configured_correctly(&pool).await {
        let msg = e.suppress_logging_of_error_message();
        tracing::warn!(
            "pg_cron extension is not configured correctly for your Postgres setup: {msg}. TensorZero will start requiring pg_cron soon. Please see our documentation to learn more about deploying Postgres: https://www.tensorzero.com/docs/deployment/postgres",
        );
    }

    Ok(())
}

/// Helper function to retrieve the set of applied migrations from the database.
/// We pull this out so that the error can be mapped in one place.
async fn get_applied_migrations(pool: &PgPool) -> Result<HashSet<i64>, sqlx::Error> {
    let mut applied_migrations: HashSet<i64> = HashSet::new();
    let mut rows =
        sqlx::query("SELECT version FROM _sqlx_migrations WHERE success = true ORDER BY version")
            .fetch(pool);
    while let Some(row) = rows.try_next().await? {
        let id: i64 = row.try_get("version")?;
        applied_migrations.insert(id);
    }
    Ok(applied_migrations)
}

pub fn make_migrator() -> sqlx::migrate::Migrator {
    migrate!("src/db/postgres/migrations")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_check_applied_expected_exact_match() {
        let applied: HashSet<i64> = [1, 2, 3].into();
        let expected: HashSet<i64> = [1, 2, 3].into();
        PostgresConnectionInfo::check_applied_expected("test", &applied, &expected)
            .expect("exact match should succeed");
    }

    #[test]
    fn test_check_applied_expected_applied_superset() {
        // During rolling upgrades, the database may have newer migrations than this gateway version expects.
        let applied: HashSet<i64> = [1, 2, 3, 4, 5].into();
        let expected: HashSet<i64> = [1, 2, 3].into();
        PostgresConnectionInfo::check_applied_expected("test", &applied, &expected)
            .expect("applied superset of expected should succeed");
    }

    #[test]
    fn test_check_applied_expected_missing_migrations() {
        // The database is missing migrations this gateway version requires.
        let applied: HashSet<i64> = [1, 2].into();
        let expected: HashSet<i64> = [1, 2, 3].into();
        let err = PostgresConnectionInfo::check_applied_expected("test", &applied, &expected)
            .expect_err("missing required migrations should fail");
        let message = err.to_string();
        assert!(
            message.contains("[3]"),
            "error should report migration 3 as missing, got: {message}"
        );
    }

    #[test]
    fn test_check_applied_expected_disjoint_sets() {
        let applied: HashSet<i64> = [1, 2].into();
        let expected: HashSet<i64> = [3, 4].into();
        PostgresConnectionInfo::check_applied_expected("test", &applied, &expected)
            .expect_err("completely disjoint sets should fail");
    }

    #[test]
    fn test_check_applied_expected_both_empty() {
        let applied: HashSet<i64> = HashSet::new();
        let expected: HashSet<i64> = HashSet::new();
        PostgresConnectionInfo::check_applied_expected("test", &applied, &expected)
            .expect("both empty should succeed");
    }

    #[test]
    fn test_check_applied_expected_empty_expected() {
        // Gateway expects no migrations (e.g. feature not enabled).
        let applied: HashSet<i64> = [1, 2, 3].into();
        let expected: HashSet<i64> = HashSet::new();
        PostgresConnectionInfo::check_applied_expected("test", &applied, &expected)
            .expect("empty expected should succeed");
    }

    #[test]
    fn test_check_applied_expected_empty_applied() {
        // Database has no migrations but gateway expects some.
        let applied: HashSet<i64> = HashSet::new();
        let expected: HashSet<i64> = [1, 2].into();
        PostgresConnectionInfo::check_applied_expected("test", &applied, &expected)
            .expect_err("empty applied with expected migrations should fail");
    }
}
