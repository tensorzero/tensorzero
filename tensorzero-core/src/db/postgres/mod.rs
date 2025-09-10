use sqlx::{migrate, postgres::PgPoolOptions, PgPool};

use crate::error::{Error, ErrorDetails};

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
    migrate!("src/db/postgres/migrations")
        .run(&pool)
        .await
        .map_err(|e| {
            Error::new(ErrorDetails::PostgresMigration {
                message: e.to_string(),
            })
        })
}
