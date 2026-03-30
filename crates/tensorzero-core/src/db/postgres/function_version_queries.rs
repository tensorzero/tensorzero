//! Postgres queries for the `function_versions` table.

use sqlx::{PgPool, Row, types::Json};
use uuid::Uuid;

use crate::config::function_versions::{
    CURRENT_SCHEMA_VERSION, StoredFunctionConfig, deserialize_stored_function_config,
};
use crate::error::{Error, ErrorDetails};

/// Insert a function version into Postgres.
/// The `StoredFunctionConfig` is serialized as a single JSONB blob.
pub async fn insert_function_version(
    pool: &PgPool,
    id: Uuid,
    function_id: Uuid,
    function_type: &str,
    stored: &StoredFunctionConfig,
    creation_source: &str,
) -> Result<(), Error> {
    let config_json = serde_json::to_value(stored).map_err(|e| {
        Error::new(ErrorDetails::Serialization {
            message: format!("Failed to serialize function version: {e}"),
        })
    })?;

    sqlx::query(
        r"INSERT INTO tensorzero.function_versions
            (id, function_id, function_type, schema_version, config, creation_source)
          VALUES ($1, $2, $3, $4, $5, $6)",
    )
    .bind(id)
    .bind(function_id)
    .bind(function_type)
    .bind(CURRENT_SCHEMA_VERSION)
    .bind(Json(&config_json))
    .bind(creation_source)
    .execute(pool)
    .await
    .map_err(|e| {
        Error::new(ErrorDetails::InternalError {
            message: format!("Failed to insert function version: {e}"),
        })
    })?;

    Ok(())
}

/// Load a function version by ID.
/// Returns (function_type, StoredFunctionConfig).
pub async fn load_function_version(
    pool: &PgPool,
    function_version_id: Uuid,
) -> Result<(String, StoredFunctionConfig), Error> {
    let row = sqlx::query(
        r"SELECT function_type, schema_version, config
          FROM tensorzero.function_versions
          WHERE id = $1",
    )
    .bind(function_version_id)
    .fetch_optional(pool)
    .await
    .map_err(|e| {
        Error::new(ErrorDetails::InternalError {
            message: format!("Failed to load function version: {e}"),
        })
    })?
    .ok_or_else(|| {
        Error::new(ErrorDetails::Config {
            message: format!("Function version `{function_version_id}` not found"),
        })
    })?;

    let function_type: String = row.get("function_type");
    let schema_version: i32 = row.get("schema_version");
    let config: serde_json::Value = row.get::<Json<serde_json::Value>, _>("config").0;

    let stored = deserialize_stored_function_config(schema_version, config)?;

    Ok((function_type, stored))
}
