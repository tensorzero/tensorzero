use std::collections::HashSet;

use futures::TryStreamExt;
use sqlx::{PgPool, Row};

pub static MIGRATOR: sqlx::migrate::Migrator = sqlx::migrate!("src/postgres/migrations");

pub fn make_migrator() -> sqlx::migrate::Migrator {
    sqlx::migrate!("src/postgres/migrations")
}

pub struct MigrationsData {
    pub applied: HashSet<i64>,
    pub expected: HashSet<i64>,
}

async fn get_applied_migrations(pool: &PgPool) -> Result<HashSet<i64>, sqlx::Error> {
    let mut applied_migrations: HashSet<i64> = HashSet::new();
    let mut rows = sqlx::query(
        "SELECT version FROM tensorzero_stored_config__sqlx_migrations WHERE success = true ORDER BY version",
    )
    .fetch(pool);
    while let Some(row) = rows.try_next().await? {
        let id: i64 = row.try_get("version")?;
        applied_migrations.insert(id);
    }
    Ok(applied_migrations)
}

pub async fn get_migrations_data(pool: &PgPool) -> Result<MigrationsData, sqlx::Error> {
    let migrator = make_migrator();
    let expected_migrations: HashSet<i64> = migrator.iter().map(|m| m.version).collect();
    let applied_migrations = get_applied_migrations(pool).await?;
    Ok(MigrationsData {
        applied: applied_migrations,
        expected: expected_migrations,
    })
}
