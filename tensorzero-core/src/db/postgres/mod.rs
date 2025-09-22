use futures::TryStreamExt;
use std::collections::HashSet;

use sqlx::{migrate, postgres::PgPoolOptions, PgPool, Row};

use crate::error::{Error, ErrorDetails};

pub mod rate_limiting;

#[derive(Debug, Clone)]
pub enum PostgresConnectionInfo {
    Enabled { pool: PgPool },
    Disabled,
}

impl PostgresConnectionInfo {
    pub fn new_with_pool(pool: PgPool) -> Self {
        Self::Enabled { pool }
    }

    pub fn new_disabled() -> Self {
        Self::Disabled
    }

    pub fn get_pool(&self) -> Option<&PgPool> {
        match self {
            Self::Enabled { pool } => Some(pool),
            Self::Disabled => None,
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
                message: format!("Failed to retrieve applied migrations: {e}"),
            })
        })?;
        // NOTE: this will break old versions of the gateway once new migrations are applied.
        // We should revisit this behavior prior to releasing a new version of the gateway.
        if applied_migrations != expected_migrations {
            return Err(Error::new(ErrorDetails::PostgresConnectionInitialization {
                message: format!(
                    "Applied migrations do not match expected migrations. Applied: {applied_migrations:?}, Expected: {expected_migrations:?}",
                ),
            }));
        }
        Ok(())
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
