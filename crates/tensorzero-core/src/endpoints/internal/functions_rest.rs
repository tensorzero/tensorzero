//! Narrow per-object REST endpoints for **functions** and **variants**.
//!
//! See `crates/tensorzero-stored-config/AGENTS.md` for the conceptual model.
//! Briefly:
//!
//! - The per-object tables are append-only edit history (the DB equivalent
//!   of `git log` over `tensorzero.toml`).
//! - At any moment, the live system is the latest non-deleted row per name.
//! - These endpoints only operate on "the current shape" — they never expose
//!   older rows for routing. `function_version_id` (a UUID) and
//!   `expected_current_function_version_id` are the CAS handles used to
//!   serialize concurrent edits; they are NOT part of the URL space.
//!
//! Eight endpoints:
//!
//! - `POST   /internal/functions`                                 — create new function
//! - `GET    /internal/functions`                                 — list active functions
//! - `GET    /internal/functions/{name}`                          — read current function
//! - `PATCH  /internal/functions/{name}`                          — update (CAS, version auto-bumps)
//! - `DELETE /internal/functions/{name}`                          — tombstone (CAS)
//! - `POST   /internal/functions/{name}/variants`                 — add new variant
//! - `GET    /internal/functions/{name}/variants`                 — list variants
//! - `PATCH  /internal/functions/{name}/variants/{variant}`       — update variant (CAS, version auto-bumps)
//! - `DELETE /internal/functions/{name}/variants/{variant}`       — remove variant (CAS)
//!
//! **Version auto-increment.** On PATCH, the server reads the current
//! version of the object being patched and writes `current + 1`. The
//! `version` field in the request body is **ignored** — clients shouldn't
//! pretend to know what the next number should be, and the auto-bump is
//! invisible to the UI ("Update" button just works). Authors who want to
//! pin an explicit version can do so on `POST` (creating a new
//! function/variant), where the supplied value is honored.
//!
//! Each mutating endpoint follows the same pattern:
//!
//! 1. Reload the latest assembled config from the per-object DB tables.
//! 2. Patch it with the requested change.
//! 3. Validate via `Config::load_from_uninitialized` (same pipeline as boot;
//!    surfaces missing template references, broken JSON schemas, etc.).
//! 4. Persist the per-object row(s) via `write_function_config` (CAS-protected).
//! 5. Reload + write a fresh snapshot + atomically hot-swap the runtime.
//!
//! Variant endpoints are a thin layer over function endpoints: each variant
//! mutation is a read-function → patch-one-variant → write-function dance.
//! The CAS slot is the function's `version_id` (one per function), so
//! function-level edits and variant-level edits compete for the same CAS
//! and therefore can't silently clobber each other.
//!
//! **DB routing rule.** This module deliberately reaches into
//! `app_state.postgres_connection_info` for reads/writes against
//! `tensorzero.function_configs` / `variant_configs` — the **raw config
//! DB**, which is Postgres-only by architecture. Snapshot writes always
//! go through `swap_state.prepare_config_swap` → `db.write_config_snapshot`
//! via `DelegatingDatabase`, which dispatches to whichever observability
//! backend (Postgres or ClickHouse) is configured. Do not move snapshot
//! work into the `sqlx::query(...)` paths below — that would silently
//! break ClickHouse-observability deployments.
//!
//! These types intentionally do NOT derive `ts_rs::TS`. Their typed
//! Rust shape pulls in dozens of transitive `UninitializedFunctionConfig`
//! sub-types that are themselves untyped over the wire today; exporting
//! TS bindings for the full surface is its own project. UI callers
//! construct request bodies as untyped JSON and parse responses by
//! shape — acceptable for the V0 internal API.

use std::collections::HashMap;

use axum::Json;
use axum::extract::{Path, State};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use sqlx::Row;
use tracing::instrument;
use uuid::Uuid;

use crate::config::snapshot::SnapshotHash;
use crate::config::{Config, UninitializedFunctionConfig, UninitializedVariantInfo};
use crate::db::postgres::function_config_writes::WriteFunctionConfigParams;
use crate::db::postgres::stored_config_queries::{load_config_from_db, merge_load_config_errors};
use crate::error::{Error, ErrorDetails};
#[expect(
    clippy::disallowed_types,
    reason = "narrow REST handlers need SwappableAppStateData to hot-swap the live config after each per-object write"
)]
use crate::utils::gateway::{
    AppStateData, PreparedConfigSwap, StructuredJson, SwappableAppStateData,
};
use std::collections::BTreeMap;
use tensorzero_stored_config::ConfigObjectMetadata;

// ─── Request / response shapes ────────────────────────────────────────────

/// Identifier returned with every successful mutating response so the
/// caller can pass it back as `expected_current_function_version_id` on
/// the next edit (CAS).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionVersionRef {
    pub function_name: String,
    pub function_version_id: Uuid,
}

/// Result envelope shared by the mutating endpoints.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConfigEditResult {
    pub function: FunctionVersionRef,
    /// Variant rows written/refreshed in this edit, keyed by variant name.
    pub variant_version_ids: HashMap<String, Uuid>,
    /// Hash of the snapshot freshly taken after applying the edit.
    pub snapshot_hash: SnapshotHash,
}

#[derive(Debug, Deserialize)]
pub struct CreateFunctionRequest {
    pub name: String,
    pub config: UninitializedFunctionConfig,
    #[serde(default)]
    pub metadata: Option<ConfigObjectMetadata>,
}

#[derive(Debug, Deserialize)]
pub struct UpdateFunctionRequest {
    pub config: UninitializedFunctionConfig,
    pub expected_current_function_version_id: Uuid,
    #[serde(default)]
    pub metadata: Option<ConfigObjectMetadata>,
}

#[derive(Debug, Deserialize)]
pub struct DeleteFunctionRequest {
    pub expected_current_function_version_id: Uuid,
}

#[derive(Debug, Clone, Serialize)]
pub struct FunctionSummary {
    pub name: String,
    pub function_type: String,
    pub function_version_id: Uuid,
    pub created_at: DateTime<Utc>,
    pub creation_source: String,
    pub metadata: ConfigObjectMetadata,
}

#[derive(Debug, Clone, Serialize)]
pub struct ListFunctionsResponse {
    pub functions: Vec<FunctionSummary>,
}

#[derive(Debug, Clone, Serialize)]
pub struct GetFunctionResponse {
    pub name: String,
    pub config: UninitializedFunctionConfig,
    pub function_version_id: Uuid,
    pub created_at: DateTime<Utc>,
    pub creation_source: String,
    pub metadata: ConfigObjectMetadata,
}

#[derive(Debug, Deserialize)]
pub struct CreateVariantRequest {
    pub variant_name: String,
    pub config: UninitializedVariantInfo,
    pub expected_current_function_version_id: Uuid,
    #[serde(default)]
    pub metadata: Option<ConfigObjectMetadata>,
}

#[derive(Debug, Deserialize)]
pub struct UpdateVariantRequest {
    pub config: UninitializedVariantInfo,
    pub expected_current_function_version_id: Uuid,
    #[serde(default)]
    pub metadata: Option<ConfigObjectMetadata>,
}

#[derive(Debug, Deserialize)]
pub struct DeleteVariantRequest {
    pub expected_current_function_version_id: Uuid,
}

#[derive(Debug, Clone, Serialize)]
pub struct VariantSummary {
    pub name: String,
    pub variant_type: String,
    pub variant_version_id: Uuid,
    pub created_at: DateTime<Utc>,
    pub creation_source: String,
    pub version: u32,
}

#[derive(Debug, Clone, Serialize)]
pub struct ListVariantsResponse {
    pub variants: Vec<VariantSummary>,
}

// ─── Mode gate ────────────────────────────────────────────────────────────

/// These endpoints only make sense when the gateway is in
/// config-in-database mode. In file-authoritative mode, the per-object
/// tables are not populated and editing must happen against the TOML
/// file in git, not via REST. Refuse early with `400 InvalidRequest`
/// rather than producing confusing empty responses or partial writes.
fn require_config_in_database(app_state: &AppStateData) -> Result<(), Error> {
    if app_state.config_in_database {
        Ok(())
    } else {
        Err(Error::new(ErrorDetails::InvalidRequest {
            message: "Per-object config endpoints require config-in-database mode. Boot the gateway without `--config-file` (and with TENSORZERO_INTERNAL_FLAG_ENABLE_CONFIG_IN_DATABASE=true) to enable them.".to_string(),
        }))
    }
}

// ─── Function handlers ────────────────────────────────────────────────────

/// `POST /internal/functions` — create a new function.
///
/// Errors with 409 if a function with this name already has an active row.
/// Use `PATCH` to update an existing function instead.
#[expect(
    clippy::disallowed_types,
    reason = "needs SwappableAppStateData to hot-swap"
)]
#[instrument(name = "functions_rest.create", skip_all, fields(function_name))]
pub async fn create_function_handler(
    State(swap_state): State<SwappableAppStateData>,
    StructuredJson(request): StructuredJson<CreateFunctionRequest>,
) -> Result<Json<ConfigEditResult>, Error> {
    tracing::Span::current().record("function_name", &request.name);
    let app_state = swap_state.load_latest();
    require_config_in_database(&app_state)?;

    // Check live config first — friendlier 4xx than waiting for the
    // CAS-conflict surface inside `write_function_config`.
    if app_state.config.functions.contains_key(&request.name) {
        return Err(Error::new(ErrorDetails::Config {
            message: format!(
                "A function named `{}` already exists. Pick a different name.",
                request.name,
            ),
        }));
    }

    let result = Box::pin(apply_function_edit(
        &swap_state,
        &app_state,
        &request.name,
        request.config,
        None,
        request.metadata.unwrap_or_default(),
        BTreeMap::new(),
    ))
    .await?;
    Ok(Json(result))
}

/// `GET /internal/functions` — list active functions.
#[instrument(name = "functions_rest.list", skip_all)]
pub async fn list_functions_handler(
    State(app_state): State<AppStateData>,
) -> Result<Json<ListFunctionsResponse>, Error> {
    require_config_in_database(&app_state)?;
    let pool = app_state
        .postgres_connection_info
        .get_pool_result()
        .map_err(|e| e.log())?;
    let rows = sqlx::query(
        r"SELECT DISTINCT ON (name)
             name, id, function_type, created_at, creation_source, metadata
           FROM tensorzero.function_configs
           WHERE deleted_at IS NULL
           ORDER BY name, created_at DESC, id DESC",
    )
    .fetch_all(pool)
    .await
    .map_err(|e| {
        Error::new(ErrorDetails::PostgresQuery {
            message: format!("list_functions query failed: {e}"),
        })
    })?;

    let mut functions = Vec::with_capacity(rows.len());
    for row in rows {
        let metadata: serde_json::Value = row.try_get("metadata")?;
        let metadata: ConfigObjectMetadata = serde_json::from_value(metadata).unwrap_or_default();
        functions.push(FunctionSummary {
            name: row.try_get("name")?,
            function_type: row.try_get("function_type")?,
            function_version_id: row.try_get("id")?,
            created_at: row.try_get("created_at")?,
            creation_source: row.try_get("creation_source")?,
            metadata,
        });
    }

    Ok(Json(ListFunctionsResponse { functions }))
}

/// `GET /internal/functions/{name}` — read the current function.
#[instrument(name = "functions_rest.get", skip_all, fields(function_name = %name))]
pub async fn get_function_handler(
    State(app_state): State<AppStateData>,
    Path(name): Path<String>,
) -> Result<Json<GetFunctionResponse>, Error> {
    require_config_in_database(&app_state)?;
    let pool = app_state
        .postgres_connection_info
        .get_pool_result()
        .map_err(|e| e.log())?;
    let row = sqlx::query(
        r"SELECT id, created_at, creation_source, metadata
           FROM tensorzero.function_configs
           WHERE name = $1 AND deleted_at IS NULL
           ORDER BY created_at DESC, id DESC
           LIMIT 1",
    )
    .bind(&name)
    .fetch_optional(pool)
    .await
    .map_err(|e| {
        Error::new(ErrorDetails::PostgresQuery {
            message: format!("get_function query failed: {e}"),
        })
    })?
    .ok_or_else(|| {
        Error::new(ErrorDetails::Config {
            message: format!("function `{name}` not found"),
        })
    })?;
    let metadata: serde_json::Value = row.try_get("metadata")?;
    let metadata: ConfigObjectMetadata = serde_json::from_value(metadata).unwrap_or_default();

    // Pull the function's current shape from a fresh DB rehydration so
    // we're returning exactly what the snapshot pipeline would produce.
    let loaded = load_config_from_db(pool)
        .await
        .map_err(merge_load_config_errors)?;
    let function_config = loaded
        .config
        .functions
        .as_ref()
        .and_then(|m| m.get(&name))
        .cloned()
        .ok_or_else(|| {
            Error::new(ErrorDetails::Config {
                message: format!("function `{name}` not present in latest DB load"),
            })
        })?;

    Ok(Json(GetFunctionResponse {
        name,
        config: function_config,
        function_version_id: row.try_get("id")?,
        created_at: row.try_get("created_at")?,
        creation_source: row.try_get("creation_source")?,
        metadata,
    }))
}

/// `PATCH /internal/functions/{name}` — update an existing function.
///
/// Version is auto-incremented: the server reads the current function's
/// `version` and writes `current + 1`. Any value supplied by the client
/// is ignored.
#[expect(
    clippy::disallowed_types,
    reason = "needs SwappableAppStateData to hot-swap"
)]
#[instrument(name = "functions_rest.update", skip_all, fields(function_name = %name))]
pub async fn update_function_handler(
    State(swap_state): State<SwappableAppStateData>,
    Path(name): Path<String>,
    StructuredJson(request): StructuredJson<UpdateFunctionRequest>,
) -> Result<Json<ConfigEditResult>, Error> {
    let app_state = swap_state.load_latest();
    require_config_in_database(&app_state)?;

    // Auto-increment the function's `version`. Read the current value
    // from the live config and override whatever the client sent.
    let current_version = current_function_version(&app_state, &name).await?;
    let mut new_config = request.config;
    set_function_version(&mut new_config, current_version + 1);

    let result = Box::pin(apply_function_edit(
        &swap_state,
        &app_state,
        &name,
        new_config,
        Some(request.expected_current_function_version_id),
        request.metadata.unwrap_or_default(),
        BTreeMap::new(),
    ))
    .await?;
    Ok(Json(result))
}

/// `DELETE /internal/functions/{name}` — tombstone the function.
///
/// Tombstones every non-deleted `function_configs` row for the name so the
/// function disappears from the active list, not just its latest revision.
/// The CAS guard matches `expected_current_function_version_id` against the
/// latest live row's id under `FOR UPDATE` to serialize against concurrent
/// edits. Variant rows are not touched — they're addressed by id from the
/// function config, so once no live function references them they are
/// naturally orphaned.
#[expect(
    clippy::disallowed_types,
    reason = "needs SwappableAppStateData to hot-swap"
)]
#[instrument(name = "functions_rest.delete", skip_all, fields(function_name = %name))]
pub async fn delete_function_handler(
    State(swap_state): State<SwappableAppStateData>,
    Path(name): Path<String>,
    StructuredJson(request): StructuredJson<DeleteFunctionRequest>,
) -> Result<Json<ConfigEditResult>, Error> {
    let app_state = swap_state.load_latest();
    require_config_in_database(&app_state)?;
    let pool = app_state
        .postgres_connection_info
        .get_pool_result()
        .map_err(|e| e.log())?;

    let mut tx = pool.begin().await.map_err(|e| {
        Error::new(ErrorDetails::PostgresQuery {
            message: format!("delete_function begin tx failed: {e}"),
        })
    })?;

    let latest_id: Option<Uuid> = sqlx::query_scalar(
        r"SELECT id FROM tensorzero.function_configs
           WHERE name = $1 AND deleted_at IS NULL
           ORDER BY created_at DESC, id DESC
           LIMIT 1
           FOR UPDATE",
    )
    .bind(&name)
    .fetch_optional(&mut *tx)
    .await
    .map_err(|e| {
        Error::new(ErrorDetails::PostgresQuery {
            message: format!("delete_function CAS read failed: {e}"),
        })
    })?;

    let latest_id = latest_id.ok_or_else(|| {
        Error::new(ErrorDetails::Config {
            message: format!("function `{name}` not found or already deleted"),
        })
    })?;

    if latest_id != request.expected_current_function_version_id {
        return Err(Error::new(ErrorDetails::InvalidRequest {
            message: format!(
                "Function `{name}` has been edited concurrently. Refresh and try again.",
            ),
        }));
    }

    sqlx::query(
        r"UPDATE tensorzero.function_configs
           SET deleted_at = NOW()
           WHERE name = $1 AND deleted_at IS NULL",
    )
    .bind(&name)
    .execute(&mut *tx)
    .await
    .map_err(|e| {
        Error::new(ErrorDetails::PostgresQuery {
            message: format!("delete_function tombstone failed: {e}"),
        })
    })?;

    tx.commit().await.map_err(|e| {
        Error::new(ErrorDetails::PostgresQuery {
            message: format!("delete_function commit failed: {e}"),
        })
    })?;

    let snapshot_hash = refresh_snapshot_and_hot_swap(&swap_state).await?;

    Ok(Json(ConfigEditResult {
        function: FunctionVersionRef {
            function_name: name,
            function_version_id: latest_id,
        },
        variant_version_ids: HashMap::new(),
        snapshot_hash,
    }))
}

// ─── Variant handlers ─────────────────────────────────────────────────────

/// `POST /internal/functions/{name}/variants` — add a new variant.
#[expect(
    clippy::disallowed_types,
    reason = "needs SwappableAppStateData to hot-swap"
)]
#[instrument(
    name = "variants_rest.create",
    skip_all,
    fields(function_name = %function_name, variant_name)
)]
pub async fn create_variant_handler(
    State(swap_state): State<SwappableAppStateData>,
    Path(function_name): Path<String>,
    StructuredJson(request): StructuredJson<CreateVariantRequest>,
) -> Result<Json<ConfigEditResult>, Error> {
    tracing::Span::current().record("variant_name", &request.variant_name);
    let app_state = swap_state.load_latest();
    require_config_in_database(&app_state)?;

    let mut function = current_function_uninitialized(&app_state, &function_name).await?;
    if function_has_variant(&function, &request.variant_name) {
        return Err(Error::new(ErrorDetails::Config {
            message: format!(
                "A variant named `{}` already exists on function `{function_name}`. Pick a different name.",
                request.variant_name,
            ),
        }));
    }
    insert_or_replace_variant(&mut function, &request.variant_name, request.config);

    let mut variant_metadata = BTreeMap::new();
    if let Some(meta) = request.metadata {
        variant_metadata.insert(request.variant_name, meta);
    }

    let result = Box::pin(apply_function_edit(
        &swap_state,
        &app_state,
        &function_name,
        function,
        Some(request.expected_current_function_version_id),
        ConfigObjectMetadata::default(),
        variant_metadata,
    ))
    .await?;
    Ok(Json(result))
}

/// `GET /internal/functions/{name}/variants` — list variants of a function.
#[instrument(name = "variants_rest.list", skip_all, fields(function_name = %function_name))]
pub async fn list_variants_handler(
    State(app_state): State<AppStateData>,
    Path(function_name): Path<String>,
) -> Result<Json<ListVariantsResponse>, Error> {
    require_config_in_database(&app_state)?;
    let pool = app_state
        .postgres_connection_info
        .get_pool_result()
        .map_err(|e| e.log())?;
    let rows = sqlx::query(
        r"SELECT DISTINCT ON (name)
             name, id, variant_type, created_at, creation_source, config
           FROM tensorzero.variant_configs
           WHERE function_name = $1
           ORDER BY name, created_at DESC, id DESC",
    )
    .bind(&function_name)
    .fetch_all(pool)
    .await
    .map_err(|e| {
        Error::new(ErrorDetails::PostgresQuery {
            message: format!("list_variants query failed: {e}"),
        })
    })?;

    let mut variants = Vec::with_capacity(rows.len());
    for row in rows {
        let config: serde_json::Value = row.try_get("config")?;
        let version = config
            .get("version")
            .and_then(|v| v.as_u64())
            .and_then(|n| u32::try_from(n).ok())
            .unwrap_or(0);
        variants.push(VariantSummary {
            name: row.try_get("name")?,
            variant_type: row.try_get("variant_type")?,
            variant_version_id: row.try_get("id")?,
            created_at: row.try_get("created_at")?,
            creation_source: row.try_get("creation_source")?,
            version,
        });
    }

    Ok(Json(ListVariantsResponse { variants }))
}

/// `PATCH /internal/functions/{name}/variants/{variant}` — update a variant.
///
/// Version is auto-incremented: the server reads the current variant's
/// `version` and writes `current + 1`. Any value supplied by the client
/// is ignored.
#[expect(
    clippy::disallowed_types,
    reason = "needs SwappableAppStateData to hot-swap"
)]
#[instrument(
    name = "variants_rest.update",
    skip_all,
    fields(function_name = %function_name, variant_name = %variant_name)
)]
pub async fn update_variant_handler(
    State(swap_state): State<SwappableAppStateData>,
    Path((function_name, variant_name)): Path<(String, String)>,
    StructuredJson(request): StructuredJson<UpdateVariantRequest>,
) -> Result<Json<ConfigEditResult>, Error> {
    let app_state = swap_state.load_latest();
    require_config_in_database(&app_state)?;
    let mut function = current_function_uninitialized(&app_state, &function_name).await?;
    if !function_has_variant(&function, &variant_name) {
        return Err(Error::new(ErrorDetails::Config {
            message: format!(
                "Variant `{variant_name}` does not exist on function `{function_name}`."
            ),
        }));
    }

    // Auto-increment the variant's `version`. Read the existing variant
    // from the live function (we know it exists from the check above),
    // bump by 1, and apply that to the incoming config.
    let current_version = current_variant_version(&function, &variant_name);
    let mut new_variant = request.config;
    new_variant.version = current_version + 1;
    insert_or_replace_variant(&mut function, &variant_name, new_variant);

    let mut variant_metadata = BTreeMap::new();
    if let Some(meta) = request.metadata {
        variant_metadata.insert(variant_name.clone(), meta);
    }

    let result = Box::pin(apply_function_edit(
        &swap_state,
        &app_state,
        &function_name,
        function,
        Some(request.expected_current_function_version_id),
        ConfigObjectMetadata::default(),
        variant_metadata,
    ))
    .await?;
    Ok(Json(result))
}

/// `DELETE /internal/functions/{name}/variants/{variant}` — remove a variant.
#[expect(
    clippy::disallowed_types,
    reason = "needs SwappableAppStateData to hot-swap"
)]
#[instrument(
    name = "variants_rest.delete",
    skip_all,
    fields(function_name = %function_name, variant_name = %variant_name)
)]
pub async fn delete_variant_handler(
    State(swap_state): State<SwappableAppStateData>,
    Path((function_name, variant_name)): Path<(String, String)>,
    StructuredJson(request): StructuredJson<DeleteVariantRequest>,
) -> Result<Json<ConfigEditResult>, Error> {
    let app_state = swap_state.load_latest();
    require_config_in_database(&app_state)?;
    let mut function = current_function_uninitialized(&app_state, &function_name).await?;
    if !function_has_variant(&function, &variant_name) {
        return Err(Error::new(ErrorDetails::Config {
            message: format!(
                "Variant `{variant_name}` does not exist on function `{function_name}`",
            ),
        }));
    }
    remove_variant(&mut function, &variant_name);

    let result = Box::pin(apply_function_edit(
        &swap_state,
        &app_state,
        &function_name,
        function,
        Some(request.expected_current_function_version_id),
        ConfigObjectMetadata::default(),
        BTreeMap::new(),
    ))
    .await?;
    Ok(Json(result))
}

// ─── Shared edit pipeline ─────────────────────────────────────────────────

/// Apply a function-level edit: validate the candidate config, write the
/// per-object rows under CAS, hot-swap the runtime, return the new
/// version IDs.
#[expect(
    clippy::disallowed_types,
    reason = "needs SwappableAppStateData to hot-swap after persistence"
)]
async fn apply_function_edit(
    swap_state: &SwappableAppStateData,
    app_state: &AppStateData,
    function_name: &str,
    new_function_config: UninitializedFunctionConfig,
    expected_current_function_version_id: Option<Uuid>,
    function_metadata: ConfigObjectMetadata,
    variant_metadata: BTreeMap<String, ConfigObjectMetadata>,
) -> Result<ConfigEditResult, Error> {
    // 1. Build a candidate `UninitializedConfig` by patching the latest
    //    DB-assembled config with the requested change. We rehydrate from
    //    the DB rather than from `live_config.functions` because the
    //    runtime `FunctionConfig::as_uninitialized` is lossy (drops
    //    schemas, experimentation, etc.).
    let pool = app_state
        .postgres_connection_info
        .get_pool_result()
        .map_err(|e| e.log())?;
    let loaded = load_config_from_db(pool)
        .await
        .map_err(merge_load_config_errors)?;
    let mut candidate = loaded.config;
    candidate
        .functions
        .get_or_insert_with(HashMap::new)
        .insert(function_name.to_string(), new_function_config.clone());

    // 2. Validate via the same pipeline as boot — same surfaces for
    //    template references, JSON schemas, `tensorzero::` prefix
    //    rules, etc. Failures here produce a 4xx before we touch the DB.
    Config::load_from_uninitialized(candidate, false)
        .await
        .map_err(|e| {
            Error::new(ErrorDetails::Config {
                message: format!("Validation failed: {e}"),
            })
        })?;

    // 3. Persist the per-object row(s) under CAS.
    let pg = &app_state.postgres_connection_info;
    let result = pg
        .write_function_config(WriteFunctionConfigParams {
            function_name,
            config: &new_function_config,
            expected_current_version_id: expected_current_function_version_id,
            creation_source: "rest/internal-api",
            source_autopilot_session_id: None,
            extra_templates: &HashMap::new(),
            function_metadata: &function_metadata,
            variant_metadata: &variant_metadata,
        })
        .await?;

    // 4. Reload, snapshot, hot-swap.
    let snapshot_hash = refresh_snapshot_and_hot_swap(swap_state).await?;

    Ok(ConfigEditResult {
        function: FunctionVersionRef {
            function_name: function_name.to_string(),
            function_version_id: result.function_version_id,
        },
        variant_version_ids: result.variant_version_ids,
        snapshot_hash,
    })
}

/// Reload the full config from the per-object tables, validate, write a
/// fresh snapshot, and atomically swap the runtime.
#[expect(
    clippy::disallowed_types,
    reason = "needs SwappableAppStateData to hot-swap after persistence"
)]
async fn refresh_snapshot_and_hot_swap(
    swap_state: &SwappableAppStateData,
) -> Result<SnapshotHash, Error> {
    let app_state = swap_state.load_latest();
    let pool = app_state
        .postgres_connection_info
        .get_pool_result()
        .map_err(|e| e.log())?;
    let loaded = load_config_from_db(pool)
        .await
        .map_err(merge_load_config_errors)?;
    let unwritten = Config::load_from_uninitialized(loaded.config, false).await?;

    let db = app_state.get_delegating_database();
    let prepared: PreparedConfigSwap = Box::pin(swap_state.prepare_config_swap(unwritten, &db))
        .await
        .map_err(|e| e.log())?;
    let hash = prepared.config().hash.clone();
    swap_state.swap_config(prepared);
    Ok(hash)
}

// ─── UninitializedConfig manipulation helpers ─────────────────────────────

/// Returns an `UninitializedFunctionConfig` reflecting the live shape of
/// `function_name`, sourced from a fresh DB rehydration. Errors if the
/// function isn't present.
async fn current_function_uninitialized(
    app_state: &AppStateData,
    function_name: &str,
) -> Result<UninitializedFunctionConfig, Error> {
    let pool = app_state
        .postgres_connection_info
        .get_pool_result()
        .map_err(|e| e.log())?;
    let loaded = load_config_from_db(pool)
        .await
        .map_err(merge_load_config_errors)?;
    loaded
        .config
        .functions
        .as_ref()
        .and_then(|m| m.get(function_name))
        .cloned()
        .ok_or_else(|| {
            Error::new(ErrorDetails::Config {
                message: format!("function `{function_name}` not found"),
            })
        })
}

fn function_has_variant(function: &UninitializedFunctionConfig, variant_name: &str) -> bool {
    match function {
        UninitializedFunctionConfig::Chat(c) => c.variants.contains_key(variant_name),
        UninitializedFunctionConfig::Json(j) => j.variants.contains_key(variant_name),
    }
}

/// Returns the variant's current `version` (0 if the variant exists but
/// is unversioned, 0 if absent — caller should `function_has_variant`
/// first if absence matters).
fn current_variant_version(function: &UninitializedFunctionConfig, variant_name: &str) -> u32 {
    let variants = match function {
        UninitializedFunctionConfig::Chat(c) => &c.variants,
        UninitializedFunctionConfig::Json(j) => &j.variants,
    };
    variants.get(variant_name).map(|v| v.version).unwrap_or(0)
}

/// Reads the function's current `version` from the live in-memory
/// config. Returns 0 if the function isn't present (defensive — the
/// caller should validate existence before bumping).
async fn current_function_version(
    app_state: &AppStateData,
    function_name: &str,
) -> Result<u32, Error> {
    let function = current_function_uninitialized(app_state, function_name).await?;
    Ok(match &function {
        UninitializedFunctionConfig::Chat(c) => c.version,
        UninitializedFunctionConfig::Json(j) => j.version,
    })
}

/// Sets the `version` on whichever function variant we have.
fn set_function_version(function: &mut UninitializedFunctionConfig, version: u32) {
    match function {
        UninitializedFunctionConfig::Chat(c) => c.version = version,
        UninitializedFunctionConfig::Json(j) => j.version = version,
    }
}

fn insert_or_replace_variant(
    function: &mut UninitializedFunctionConfig,
    variant_name: &str,
    variant: UninitializedVariantInfo,
) {
    match function {
        UninitializedFunctionConfig::Chat(c) => {
            c.variants.insert(variant_name.to_string(), variant);
        }
        UninitializedFunctionConfig::Json(j) => {
            j.variants.insert(variant_name.to_string(), variant);
        }
    }
}

fn remove_variant(function: &mut UninitializedFunctionConfig, variant_name: &str) {
    match function {
        UninitializedFunctionConfig::Chat(c) => {
            c.variants.remove(variant_name);
        }
        UninitializedFunctionConfig::Json(j) => {
            j.variants.remove(variant_name);
        }
    }
}
