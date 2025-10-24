use std::collections::HashSet;

use chrono::{DateTime, Utc};
use futures::TryStreamExt;
use secrecy::{ExposeSecret, SecretString};
use sqlx::PgPool;
use sqlx::Row;

use crate::key::{TensorZeroApiKey, TensorZeroAuthError, secure_fresh_api_key};

pub fn make_migrator() -> sqlx::migrate::Migrator {
    sqlx::migrate!("src/postgres/migrations")
}

pub struct MigrationsData {
    pub applied: HashSet<i64>,
    pub expected: HashSet<i64>,
}

/// Helper function to retrieve the set of applied migrations from the database.
/// We pull this out so that the error can be mapped in one place.
/// This is almost the same as the corresponding `get_applied_migrations` function in 'tensorzero_core', but with a different table name,
/// and using sqlx_alpha
/// TODO - consolidate these functions into a single `get_applied_migrations`function  once we're using a single sqlx version.
async fn get_applied_migrations(pool: &PgPool) -> Result<HashSet<i64>, sqlx::Error> {
    let mut applied_migrations: HashSet<i64> = HashSet::new();
    let mut rows =
        sqlx::query("SELECT version FROM tensorzero_auth__sqlx_migrations WHERE success = true ORDER BY version")
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
    // Query the database for all successfully applied migration versions.
    let applied_migrations = get_applied_migrations(pool).await?;
    Ok(MigrationsData {
        applied: applied_migrations,
        expected: expected_migrations,
    })
}

/// Create a new API key, and store in the database.
/// Returns the generated API key
pub async fn create_key(
    organization: &str,
    workspace: &str,
    description: Option<&str>,
    pool: &PgPool,
) -> Result<SecretString, TensorZeroAuthError> {
    let key = secure_fresh_api_key();
    let parsed_key = TensorZeroApiKey::parse(key.expose_secret())?;
    sqlx::query!(
        "INSERT INTO tensorzero_auth_api_key (organization, workspace, description, short_id, hash) VALUES ($1, $2, $3, $4, $5)",
        organization,
        workspace,
        description,
        parsed_key.short_id,
        parsed_key.hashed_long_key.expose_secret()
    ).execute(pool).await?;
    Ok(key)
}

#[derive(Debug)]
pub enum AuthResult {
    /// The API key exists and is not disabled.
    Success(KeyInfo),
    /// The API key exists, but was disabled at the specified time.
    Disabled(DateTime<Utc>),
    /// The API key does not exist.
    MissingKey,
}

#[derive(sqlx::FromRow, Debug, PartialEq, Eq, Clone)]
pub struct KeyInfo {
    pub id: i64,
    pub organization: String,
    pub workspace: String,
    pub description: Option<String>,
    pub disabled_at: Option<DateTime<Utc>>,
}

/// Looks up an API key in the database, and checks that it was not disabled.
pub async fn check_key(
    key: &TensorZeroApiKey,
    pool: &PgPool,
) -> Result<AuthResult, TensorZeroAuthError> {
    let key = sqlx::query_as!(
        KeyInfo,
        "SELECT id, organization, workspace, description, disabled_at from tensorzero_auth_api_key WHERE short_id = $1 AND hash = $2",
        key.short_id,
        key.hashed_long_key.expose_secret()
    ).fetch_optional(pool).await?;
    match key {
        Some(key) => {
            if let Some(disabled_at) = key.disabled_at {
                Ok(AuthResult::Disabled(disabled_at))
            } else {
                Ok(AuthResult::Success(key))
            }
        }
        None => Ok(AuthResult::MissingKey),
    }
}

/// Marks an API key as disabled in the database
pub async fn disable_key(key: &TensorZeroApiKey, pool: &PgPool) -> Result<(), TensorZeroAuthError> {
    sqlx::query!(
        "UPDATE tensorzero_auth_api_key SET disabled_at = $1, updated_at = $1 WHERE short_id = $2 AND hash = $3",
        Utc::now(),
        key.short_id,
        key.hashed_long_key.expose_secret()
    )
    .execute(pool)
    .await?;
    Ok(())
}

/// Lists all API keys in the database, optionally filtered by organization,
/// with an optional limit and offset.
pub async fn list_key_info(
    organization: Option<String>,
    limit: Option<u32>,
    offset: Option<u32>,
    pool: &PgPool,
) -> Result<Vec<KeyInfo>, TensorZeroAuthError> {
    let keys = sqlx::query_as!(
        KeyInfo,
        "SELECT id, organization, workspace, description, disabled_at from tensorzero_auth_api_key WHERE (organization = $1 OR $1 is NULL) ORDER BY ID ASC LIMIT $2 OFFSET $3",
        organization,
        // We take in a 'u32' and convert to 'i64' to avoid any weirdness around negative values
        // Postgres does the right thing when the LIMIT or OFFSET is null (it gets ignored)
        limit.map(i64::from),
        offset.map(i64::from)
    )
    .fetch_all(pool)
    .await?;
    Ok(keys)
}
