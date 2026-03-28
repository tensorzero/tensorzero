//! Postgres queries for `prompt_template_versions` and `variant_versions` tables.
//!
//! ## Startup rehydration flow
//!
//! Function config stays on disk (TOML). Variants are stored in the database.
//! On startup:
//!
//! 1. Parse function config from TOML — each function references variant names.
//! 2. Call `load_latest_variant_versions()` with all variant names across all functions.
//! 3. For each loaded `StoredVariantVersion`, call `referenced_prompt_template_ids()`
//!    to collect all prompt template IDs.
//! 4. Call `load_prompt_template_versions()` with the collected IDs.
//! 5. For each variant, call `rehydrate_variant()` with the prompt rows to get
//!    `UninitializedVariantInfo`.
//! 6. Plug the rehydrated variants into the TOML-loaded function config and proceed
//!    with the normal `load()` pipeline.
//!
//! This is implemented in `load_and_rehydrate_variants()` below.

use std::collections::HashMap;

use sqlx::{PgPool, Row, types::Json};
use uuid::Uuid;

use crate::config::UninitializedVariantInfo;
use crate::config::variant_versions::{
    CURRENT_SCHEMA_VERSION, PromptTemplateVersionRow, StoredVariantVersion,
    deserialize_stored_variant_version, rehydrate_variant,
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
    function_name: &str,
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
            (id, function_name, variant_name, schema_version, config, creation_source)
          VALUES ($1, $2, $3, $4, $5, $6)",
    )
    .bind(id)
    .bind(function_name)
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

/// Load the latest variant version for each of the given variant names within a function.
/// "Latest" = highest UUIDv7 (most recently created) for that (function_name, variant_name).
/// Returns a map from variant_name → StoredVariantVersion.
pub async fn load_latest_variant_versions(
    pool: &PgPool,
    function_name: &str,
    variant_names: &[&str],
) -> Result<HashMap<String, StoredVariantVersion>, Error> {
    if variant_names.is_empty() {
        return Ok(HashMap::new());
    }

    // Use DISTINCT ON to get the latest row per variant_name, ordered by id DESC (UUIDv7 = time-ordered)
    let rows = sqlx::query(
        r"SELECT DISTINCT ON (variant_name) variant_name, schema_version, config
          FROM tensorzero.variant_versions
          WHERE function_name = $1 AND variant_name = ANY($2)
          ORDER BY variant_name, id DESC",
    )
    .bind(function_name)
    .bind(variant_names)
    .fetch_all(pool)
    .await
    .map_err(|e| {
        Error::new(ErrorDetails::InternalError {
            message: format!("Failed to load latest variant versions: {e}"),
        })
    })?;

    let mut result = HashMap::with_capacity(rows.len());
    for row in rows {
        let variant_name: String = row.get("variant_name");
        let schema_version: i32 = row.get("schema_version");
        let config: serde_json::Value = row.get::<Json<serde_json::Value>, _>("config").0;

        let stored = deserialize_stored_variant_version(schema_version, config)?;
        result.insert(variant_name, stored);
    }

    Ok(result)
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

// ─── Startup rehydration ─────────────────────────────────────────────────────

/// Load the latest variant versions for the given names from the database,
/// resolve all referenced prompt templates, and rehydrate to `UninitializedVariantInfo`.
///
/// Returns a map from variant_name → `UninitializedVariantInfo`, ready to be plugged
/// into the TOML-loaded function config.
///
/// Variant names not found in the database are silently omitted from the result
/// (the caller should check for missing variants and decide whether to error).
pub async fn load_and_rehydrate_variants(
    pool: &PgPool,
    function_name: &str,
    variant_names: &[&str],
) -> Result<HashMap<String, UninitializedVariantInfo>, Error> {
    // Step 1: Load latest variant versions by name
    let stored_variants = load_latest_variant_versions(pool, function_name, variant_names).await?;

    if stored_variants.is_empty() {
        return Ok(HashMap::new());
    }

    // Step 2: Collect all referenced prompt template IDs across all variants
    let all_prompt_ids: Vec<Uuid> = stored_variants
        .values()
        .flat_map(|sv| sv.referenced_prompt_template_ids())
        .collect();

    // Step 3: Batch-load all prompt templates
    let prompt_rows = load_prompt_template_versions(pool, &all_prompt_ids).await?;

    // Step 4: Rehydrate each variant
    let mut result = HashMap::with_capacity(stored_variants.len());
    for (name, stored_variant) in &stored_variants {
        let variant_info = rehydrate_variant(stored_variant, &prompt_rows)?;
        result.insert(name.clone(), variant_info);
    }

    Ok(result)
}
