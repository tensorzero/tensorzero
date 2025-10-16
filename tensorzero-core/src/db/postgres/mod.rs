use async_trait::async_trait;
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
    },
    #[cfg(test)]
    Mock {
        healthy: bool,
    },
    Disabled,
}

impl PostgresConnectionInfo {
    pub fn new_with_pool(pool: PgPool) -> Self {
        Self::Enabled { pool }
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
            Self::Enabled { pool } => Some(pool),
            #[cfg(test)]
            Self::Mock { .. } => None,
            Self::Disabled => None,
        }
    }

    pub fn get_pool_result(&self) -> Result<&PgPool, Error> {
        match self {
            Self::Enabled { pool } => Ok(pool),
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
        let Some(pool) = self.get_pool() else {
            return Ok(());
        };
        let migrator = make_migrator();
        let expected_migrations: HashSet<i64> = migrator.iter().map(|m| m.version).collect();
        // Query the database for all successfully applied migration versions.
        let applied_migrations = get_applied_migrations(pool).await.map_err(|e| {
            Error::new(ErrorDetails::PostgresConnectionInitialization {
                message: format!("Failed to retrieve applied migrations: {e}. You may need to run the migrations with `{}`", get_run_migrations_command()),
            })
        })?;
        // NOTE: this will break old versions of the gateway once new migrations are applied.
        // We should revisit this behavior prior to releasing a new version of the gateway.
        if applied_migrations != expected_migrations {
            return Err(Error::new(ErrorDetails::PostgresConnectionInitialization {
                message: format!(
                    "Applied migrations do not match expected migrations. Applied: {applied_migrations:?}, Expected: {expected_migrations:?}. Please run the migrations with `{}`",
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
            Self::Enabled { pool } => {
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
                // TODO(shuyang): customize postgres timeout
                match timeout(Duration::from_millis(500), check).await {
                    Ok(healthcheck_status) => healthcheck_status,
                    Err(_) => Err(Error::new(ErrorDetails::PostgresConnection {
                        message: "Postgres healthcheck query timed out.".to_string(),
                    })),
                }
            }
        }
    }
}

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
    })
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
