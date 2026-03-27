//! Postgres queries for `prompt_template_versions` and `variant_versions` tables.

use std::collections::HashMap;

use sqlx::{PgPool, Row, types::Json};
use uuid::Uuid;

use crate::config::variant_versions::{
    CURRENT_SCHEMA_VERSION, PromptTemplateVersionRow, StoredVariantVersion,
    deserialize_stored_variant_version,
};
use crate::error::{Error, ErrorDetails};

/// Insert a batch of prompt template versions into Postgres.
pub async fn insert_prompt_template_versions(
    pool: &PgPool,
    templates: &[(Uuid, String, String)], // (id, template_key, source_body)
    creation_source: &str,
) -> Result<(), Error> {
    if templates.is_empty() {
        return Ok(());
    }

    let mut qb = sqlx::QueryBuilder::new(
        r"INSERT INTO tensorzero.prompt_template_versions (id, template_key, source_body, creation_source) ",
    );

    qb.push_values(templates.iter(), |mut b, (id, key, body)| {
        b.push_bind(id)
            .push_bind(key)
            .push_bind(body)
            .push_bind(creation_source);
    });

    qb.build().execute(pool).await.map_err(|e| {
        Error::new(ErrorDetails::InternalError {
            message: format!("Failed to insert prompt template versions: {e}"),
        })
    })?;

    Ok(())
}

/// Insert a variant version into Postgres.
/// The `StoredVariantVersion` is serialized as a single JSONB blob (including the variant
/// type tag, config, timeouts, and namespace).
pub async fn insert_variant_version(
    pool: &PgPool,
    id: Uuid,
    variant_name: &str,
    stored: &StoredVariantVersion,
    creation_source: &str,
) -> Result<(), Error> {
    let config_json = serde_json::to_value(stored).map_err(|e| {
        Error::new(ErrorDetails::Serialization {
            message: format!("Failed to serialize variant version: {e}"),
        })
    })?;

    sqlx::query(
        r"INSERT INTO tensorzero.variant_versions
            (id, variant_name, schema_version, config, creation_source)
          VALUES ($1, $2, $3, $4, $5)",
    )
    .bind(id)
    .bind(variant_name)
    .bind(CURRENT_SCHEMA_VERSION)
    .bind(Json(&config_json))
    .bind(creation_source)
    .execute(pool)
    .await
    .map_err(|e| {
        Error::new(ErrorDetails::InternalError {
            message: format!("Failed to insert variant version: {e}"),
        })
    })?;

    Ok(())
}

/// Load a variant version by ID.
pub async fn load_variant_version(
    pool: &PgPool,
    variant_version_id: Uuid,
) -> Result<(String, StoredVariantVersion), Error> {
    let row = sqlx::query(
        r"SELECT variant_name, schema_version, config
          FROM tensorzero.variant_versions
          WHERE id = $1",
    )
    .bind(variant_version_id)
    .fetch_optional(pool)
    .await
    .map_err(|e| {
        Error::new(ErrorDetails::InternalError {
            message: format!("Failed to load variant version: {e}"),
        })
    })?
    .ok_or_else(|| {
        Error::new(ErrorDetails::Config {
            message: format!("Variant version `{variant_version_id}` not found"),
        })
    })?;

    let variant_name: String = row.get("variant_name");
    let schema_version: i32 = row.get("schema_version");
    let config: serde_json::Value = row.get::<Json<serde_json::Value>, _>("config").0;

    let stored = deserialize_stored_variant_version(schema_version, config)?;

    Ok((variant_name, stored))
}

/// Load prompt template versions by their IDs.
pub async fn load_prompt_template_versions(
    pool: &PgPool,
    ids: &[Uuid],
) -> Result<HashMap<Uuid, PromptTemplateVersionRow>, Error> {
    if ids.is_empty() {
        return Ok(HashMap::new());
    }

    let rows = sqlx::query(
        r"SELECT id, template_key, source_body
          FROM tensorzero.prompt_template_versions
          WHERE id = ANY($1)",
    )
    .bind(ids)
    .fetch_all(pool)
    .await
    .map_err(|e| {
        Error::new(ErrorDetails::InternalError {
            message: format!("Failed to load prompt template versions: {e}"),
        })
    })?;

    let mut result = HashMap::with_capacity(rows.len());
    for row in rows {
        let id: Uuid = row.get("id");
        result.insert(
            id,
            PromptTemplateVersionRow {
                id,
                template_key: row.get("template_key"),
                source_body: row.get("source_body"),
            },
        );
    }

    Ok(result)
}
