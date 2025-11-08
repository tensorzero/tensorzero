use std::collections::HashSet;

use chrono::SubsecRound;
use chrono::{DateTime, Utc};
use futures::TryStreamExt;
use secrecy::{ExposeSecret, SecretString};
use serde::Serialize;
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
        "INSERT INTO tensorzero_auth_api_key (organization, workspace, description, public_id, hash) VALUES ($1, $2, $3, $4, $5)",
        organization,
        workspace,
        description,
        parsed_key.public_id,
        parsed_key.hashed_long_key.expose_secret()
    ).execute(pool).await?;
    Ok(key)
}

#[derive(Debug, Clone)]
pub enum AuthResult {
    /// The API key exists and is not disabled.
    Success(KeyInfo),
    /// The API key exists, but was disabled at the specified time.
    Disabled(DateTime<Utc>),
    /// The API key does not exist.
    MissingKey,
}

#[derive(sqlx::FromRow, Debug, PartialEq, Eq, Clone, Serialize, ts_rs::TS)]
#[ts(export)]
pub struct KeyInfo {
    pub public_id: String,
    pub organization: String,
    pub workspace: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    #[cfg_attr(test, ts(optional))]
    pub description: Option<String>,
    pub created_at: DateTime<Utc>,
    #[serde(skip_serializing_if = "Option::is_none")]
    #[cfg_attr(test, ts(optional))]
    pub disabled_at: Option<DateTime<Utc>>,
}

/// Looks up an API key in the database, and checks that it was not disabled.
pub async fn check_key(
    key: &TensorZeroApiKey,
    pool: &PgPool,
) -> Result<AuthResult, TensorZeroAuthError> {
    let key = sqlx::query_as!(
        KeyInfo,
        "SELECT public_id, organization, workspace, description, created_at, disabled_at from tensorzero_auth_api_key WHERE public_id = $1 AND hash = $2",
        key.public_id,
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

/// Marks an API key as disabled in the database by its public_id
/// Returns the `disabled_at` timestamp that was set in the database.
pub async fn disable_key(
    public_id: &str,
    pool: &PgPool,
) -> Result<DateTime<Utc>, TensorZeroAuthError> {
    // Round to microseconds, since postgres only has microsecond precision
    // This ensures that the value we return matches the value we set in the database.
    let now = Utc::now().round_subsecs(6);
    sqlx::query!(
        "UPDATE tensorzero_auth_api_key SET disabled_at = $1, updated_at = $1 WHERE public_id = $2",
        now,
        public_id
    )
    .execute(pool)
    .await?;
    Ok(now)
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
        "SELECT public_id, organization, workspace, description, created_at, disabled_at FROM tensorzero_auth_api_key WHERE (organization = $1 OR $1 is NULL) ORDER BY created_at DESC LIMIT $2 OFFSET $3",
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
