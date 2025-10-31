use async_trait::async_trait;
use futures::future::try_join;
use futures::TryStreamExt;
use std::{collections::HashSet, time::Duration};
use tokio::time::timeout;

use sqlx::{migrate, postgres::PgPoolOptions, PgPool, Row};

use crate::error::{Error, ErrorDetails};

use super::HealthCheckable;

pub mod experimentation;
pub mod rate_limiting;

fn get_run_migrations_command() -> String {
    let version = env!("CARGO_PKG_VERSION");
    format!("docker run --rm -e TENSORZERO_POSTGRES_URL=$TENSORZERO_POSTGRES_URL tensorzero/gateway:{version} --run-postgres-migrations")
}

#[derive(Debug, Clone)]
pub enum PostgresConnectionInfo {
    Enabled {
        pool: PgPool,
        /// The `tensorzero-auth` crate currently uses an alpha release of `sqlx`, so the `PgPool` type is different.
        /// As a result, we need to create a separate pool for it.
        alpha_pool: Option<sqlx_alpha::PgPool>,
    },
    #[cfg(test)]
    Mock {
        healthy: bool,
    },
    Disabled,
}

impl PostgresConnectionInfo {
    pub fn new_with_pool(pool: PgPool, alpha_pool: Option<sqlx_alpha::PgPool>) -> Self {
        Self::Enabled { pool, alpha_pool }
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
            Self::Enabled {
                pool,
                alpha_pool: _,
            } => Some(pool),
            #[cfg(test)]
            Self::Mock { .. } => None,
            Self::Disabled => None,
        }
    }

    pub fn get_alpha_pool(&self) -> Option<&sqlx_alpha::PgPool> {
        match self {
            Self::Enabled {
                pool: _,
                alpha_pool,
            } => alpha_pool.as_ref(),
            #[cfg(test)]
            Self::Mock { .. } => None,
            Self::Disabled => None,
        }
    }

    pub fn get_pool_result(&self) -> Result<&PgPool, Error> {
        match self {
            Self::Enabled {
                pool,
                alpha_pool: _,
            } => Ok(pool),
            #[cfg(test)]
            Self::Mock { .. } => Err(Error::new(ErrorDetails::PostgresConnectionInitialization {
                message: "Mock database is not supported".to_string(),
            })),
            Self::Disabled => Err(Error::new(ErrorDetails::PostgresConnectionInitialization {
                message: "Database is disabled".to_string(),
            })),
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
                message: format!("Failed to retrieve applied `tensorzero-core` migrations: {e}. You may need to run the migrations with `{}`", get_run_migrations_command()),
            })
        })?;

            Self::check_applied_expected(
                "tensorzero-core",
                &applied_migrations,
                &expected_migrations,
            )?;
        }

        if let Some(alpha_pool) = self.get_alpha_pool() {
            let tensorzero_auth_migrations_data =
                tensorzero_auth::postgres::get_migrations_data(alpha_pool).await.map_err(|e| {
                    Error::new(ErrorDetails::PostgresConnectionInitialization {
                        message: format!("Failed to retrieve applied `tensorzero-auth` migrations: {e}. You may need to run the migrations with `{}`", get_run_migrations_command()),
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

    fn check_applied_expected(
        name: &str,
        applied_migrations: &HashSet<i64>,
        expected_migrations: &HashSet<i64>,
    ) -> Result<(), Error> {
        // NOTE: this will break old versions of the gateway once new migrations are applied.
        // We should revisit this behavior prior to releasing a new version of the gateway.
        if applied_migrations != expected_migrations {
            return Err(Error::new(ErrorDetails::PostgresConnectionInitialization {
                message: format!(
                    "Applied `{name}` migrations do not match expected migrations. Applied: {applied_migrations:?}, Expected: {expected_migrations:?}. Please run the migrations with `{}`",
                    get_run_migrations_command()
                ),
            }));
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
            Self::Enabled { pool, alpha_pool } => {
                let check = async {
                    let _result = sqlx::query("SELECT 1")
                        .fetch_one(pool)
                        .await
                        .map_err(|_e| {
                            Error::new(ErrorDetails::PostgresConnection {
                                message: _e.to_string(),
                            })
                        })?;

                    Ok(())
                };
                let alpha_check = async {
                    if let Some(alpha_pool) = alpha_pool {
                        let _result = sqlx_alpha::query("SELECT 1")
                            .fetch_one(alpha_pool)
                            .await
                            .map_err(|_e| {
                                Error::new(ErrorDetails::PostgresConnection {
                                    message: _e.to_string(),
                                })
                            })?;
                        Ok(())
                    } else {
                        Ok(())
                    }
                };

                // TODO(shuyang): customize postgres timeout
                match timeout(Duration::from_millis(1000), try_join(check, alpha_check)).await {
                    Ok(Ok(((), ()))) => Ok(()),
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
    let pool = PgPoolOptions::new()
        .connect(&postgres_url)
        .await
        .map_err(|err| {
            Error::new(ErrorDetails::PostgresConnectionInitialization {
                message: err.to_string(),
            })
        })?;
    make_migrator().run(&pool).await.map_err(|e| {
        Error::new(ErrorDetails::PostgresMigration {
            message: e.to_string(),
        })
    })?;

    // Our 'tensorzero-auth' crate currently uses an alpha release of 'sqlx', so the `PgPool` type is different.
    let sqlx_alpha_pool = sqlx_alpha::PgPool::connect(&postgres_url)
        .await
        .map_err(|err| {
            Error::new(ErrorDetails::PostgresConnectionInitialization {
                message: err.to_string(),
            })
        })?;
    tensorzero_auth::postgres::make_migrator()
        .run(&sqlx_alpha_pool)
        .await
        .map_err(|e| {
            Error::new(ErrorDetails::PostgresMigration {
                message: format!("Failed to run tensorzero-auth migrations: {e}"),
            })
        })?;
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

fn make_migrator() -> sqlx::migrate::Migrator {
    migrate!("src/db/postgres/migrations")
}
