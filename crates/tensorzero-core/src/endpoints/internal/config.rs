//! Config snapshot endpoints.
//!
//! These endpoints allow retrieving config snapshots by hash, or the live config,
//! and writing new config snapshots.

use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;

use axum::Json;
use axum::extract::{Path, State};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use sqlx::Postgres;
use tracing::instrument;

use crate::config::editable::config_to_toml;
use crate::config::editable::toml_to_config;
use crate::config::snapshot::{ConfigSnapshot, SnapshotHash};
use crate::config::{Config, UninitializedConfig};
use crate::db::ConfigQueries;
use crate::db::postgres::stored_config_queries::{load_config_from_db, load_config_from_db_in_tx};
use crate::db::postgres::stored_config_writes::{
    WriteStoredConfigParams, write_stored_config_in_tx,
};
use crate::error::{Error, ErrorDetails};
use crate::feature_flags;
use crate::utils::gateway::{
    AppState, ResolvedAppStateData, StructuredJson, SwappableAppStateData,
};

const CONFIG_EDITOR_ADVISORY_LOCK_KEY: i64 = 0x434F_4E46_4947_544D;

/// Response containing a config snapshot.
#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Debug, Serialize, Deserialize)]
#[cfg_attr(feature = "ts-bindings", ts(export, optional_fields))]
pub struct GetConfigResponse {
    /// The config as a JSON value.
    /// Important: This should not be a strongly typed UninitializedConfig.
    /// Nothing outside of the gateway should attempt to deserialize it into UninitializedConfig.
    pub config: Value,
    /// The hash identifying this config version.
    pub hash: String,
    /// Templates that were loaded from the filesystem.
    pub extra_templates: HashMap<String, String>,
    /// User-defined tags for categorizing or labeling this config snapshot.
    pub tags: HashMap<String, String>,
}

impl GetConfigResponse {
    fn from_uninitialized(
        config: UninitializedConfig,
        hash: String,
        extra_templates: HashMap<String, String>,
        tags: HashMap<String, String>,
    ) -> Result<Self, Error> {
        let config = serde_json::to_value(&config).map_err(|e| {
            Error::new(ErrorDetails::Config {
                message: format!("Failed to serialize config: {e}"),
            })
        })?;
        Ok(Self {
            config,
            hash,
            extra_templates,
            tags,
        })
    }

    fn from_snapshot(snapshot: ConfigSnapshot) -> Result<Self, Error> {
        let uninitialized: UninitializedConfig =
            snapshot.config.try_into().map_err(|e: &'static str| {
                Error::new(ErrorDetails::Config {
                    message: e.to_string(),
                })
            })?;
        Self::from_uninitialized(
            uninitialized,
            snapshot.hash.to_string(),
            snapshot.extra_templates,
            snapshot.tags,
        )
    }
}

/// Response containing a TOML-editable config snapshot.
#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Debug, Serialize, Deserialize)]
#[cfg_attr(feature = "ts-bindings", ts(export, optional_fields))]
pub struct GetConfigTomlResponse {
    /// Human-readable TOML config with path strings instead of inlined file contents.
    pub toml: String,
    /// File-backed content referenced by the TOML, keyed by the path string in the TOML body.
    pub path_contents: HashMap<String, String>,
    /// The hash identifying this config version.
    pub hash: String,
    /// Compare-and-swap signature for the full editable document.
    pub base_signature: String,
    /// User-defined tags for categorizing or labeling this config snapshot.
    pub tags: HashMap<String, String>,
}

impl GetConfigTomlResponse {
    fn from_uninitialized(
        config: UninitializedConfig,
        hash: String,
        tags: HashMap<String, String>,
    ) -> Result<Self, Error> {
        let (toml, path_contents) = config_to_toml(&config)?;
        let base_signature = editable_config_signature(&toml, &path_contents)?;
        Ok(Self {
            toml,
            path_contents,
            hash,
            base_signature,
            tags,
        })
    }

    fn from_snapshot(snapshot: ConfigSnapshot) -> Result<Self, Error> {
        let uninitialized: UninitializedConfig =
            snapshot.config.try_into().map_err(|e: &'static str| {
                Error::new(ErrorDetails::Config {
                    message: e.to_string(),
                })
            })?;
        Self::from_uninitialized(uninitialized, snapshot.hash.to_string(), snapshot.tags)
    }
}

#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Debug, Deserialize, Serialize)]
#[cfg_attr(feature = "ts-bindings", ts(export, optional_fields))]
pub struct ApplyConfigTomlRequest {
    pub base_signature: String,
    pub toml: String,
    #[serde(default)]
    pub path_contents: HashMap<String, String>,
}

#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Debug, Deserialize, Serialize)]
#[cfg_attr(feature = "ts-bindings", ts(export, optional_fields))]
pub struct ApplyConfigTomlResponse {
    pub toml: String,
    pub path_contents: HashMap<String, String>,
    pub hash: String,
    pub base_signature: String,
}

fn editable_config_signature(
    toml: &str,
    path_contents: &HashMap<String, String>,
) -> Result<String, Error> {
    let sorted_path_contents = path_contents
        .iter()
        .map(|(path, contents)| (path.clone(), contents.clone()))
        .collect::<BTreeMap<_, _>>();
    let payload = serde_json::to_vec(&json!({
        "toml": toml,
        "path_contents": sorted_path_contents,
    }))
    .map_err(|e| {
        Error::new(ErrorDetails::Serialization {
            message: format!("Failed to serialize editable config signature payload: {e}"),
        })
    })?;
    Ok(hex::encode(Sha256::digest(payload)))
}

fn merge_db_config_errors(errors: Vec<Error>) -> Error {
    let mut iter = errors.into_iter();
    let Some(first) = iter.next() else {
        return Error::new(ErrorDetails::Config {
            message: "Failed to load config from database".to_string(),
        });
    };
    if iter.len() == 0 {
        return first;
    }

    let message = std::iter::once(first)
        .chain(iter)
        .map(|error| error.to_string())
        .collect::<Vec<_>>()
        .join("; ");
    Error::new(ErrorDetails::Config { message })
}

async fn load_db_authoritative_uninitialized_config(
    app_state: &ResolvedAppStateData,
) -> Result<(UninitializedConfig, String), Error> {
    let pool = app_state
        .postgres_connection_info
        .get_pool_result()
        .map_err(|e| e.log())?;
    let uninitialized = load_config_from_db(pool)
        .await
        .map_err(merge_db_config_errors)?;
    let validated = Config::load_from_uninitialized(uninitialized.clone(), false).await?;
    Ok((uninitialized, validated.hash.to_string()))
}

async fn load_db_authoritative_config_toml(
    app_state: &ResolvedAppStateData,
) -> Result<GetConfigTomlResponse, Error> {
    let (uninitialized, hash) = load_db_authoritative_uninitialized_config(app_state).await?;
    GetConfigTomlResponse::from_uninitialized(uninitialized, hash, HashMap::new())
}

async fn load_db_authoritative_config_json(
    app_state: &ResolvedAppStateData,
) -> Result<GetConfigResponse, Error> {
    let (uninitialized, hash) = load_db_authoritative_uninitialized_config(app_state).await?;
    GetConfigResponse::from_uninitialized(uninitialized, hash, HashMap::new(), HashMap::new())
}

async fn acquire_config_editor_lock(tx: &mut sqlx::Transaction<'_, Postgres>) -> Result<(), Error> {
    sqlx::query("SELECT pg_advisory_xact_lock($1)")
        .bind(CONFIG_EDITOR_ADVISORY_LOCK_KEY)
        .execute(&mut **tx)
        .await
        .map_err(|e| {
            Error::new(ErrorDetails::PostgresQuery {
                message: format!("Failed to acquire config editor advisory lock: {e}"),
            })
        })?;
    Ok(())
}

/// Handler for `GET /internal/config`
///
/// Returns the live config snapshot.
#[axum::debug_handler(state = SwappableAppStateData)]
#[instrument(name = "config.get_live", skip_all)]
pub async fn get_live_config_handler(
    State(app_state): AppState,
) -> Result<Json<GetConfigResponse>, Error> {
    if feature_flags::ENABLE_CONFIG_IN_DATABASE.get() {
        return Ok(Json(load_db_authoritative_config_json(&app_state).await?));
    }

    let hash = app_state.config.hash.clone();
    let db = app_state.get_delegating_database();
    let snapshot = db.get_config_snapshot(hash).await?;

    Ok(Json(GetConfigResponse::from_snapshot(snapshot)?))
}

/// Handler for `GET /internal/config/{hash}`
///
/// Returns a config snapshot by hash.
#[axum::debug_handler(state = SwappableAppStateData)]
#[instrument(name = "config.get_by_hash", skip_all, fields(hash = %hash))]
pub async fn get_config_by_hash_handler(
    State(app_state): AppState,
    Path(hash): Path<String>,
) -> Result<Json<GetConfigResponse>, Error> {
    let snapshot_hash: SnapshotHash = hash.parse().map_err(|_| {
        Error::new(ErrorDetails::ConfigSnapshotNotFound {
            snapshot_hash: hash.clone(),
        })
    })?;

    let db = app_state.get_delegating_database();
    let snapshot = db.get_config_snapshot(snapshot_hash).await?;

    Ok(Json(GetConfigResponse::from_snapshot(snapshot)?))
}

/// Handler for `GET /internal/config_toml`
///
/// Returns the live config snapshot in editable TOML form.
#[axum::debug_handler(state = SwappableAppStateData)]
#[instrument(name = "config.get_live_toml", skip_all)]
pub async fn get_live_config_toml_handler(
    State(app_state): AppState,
) -> Result<Json<GetConfigTomlResponse>, Error> {
    if feature_flags::ENABLE_CONFIG_IN_DATABASE.get() {
        return Ok(Json(load_db_authoritative_config_toml(&app_state).await?));
    }

    let hash = app_state.config.hash.clone();
    let db = app_state.get_delegating_database();
    let snapshot = db.get_config_snapshot(hash).await?;

    Ok(Json(GetConfigTomlResponse::from_snapshot(snapshot)?))
}

/// Handler for `GET /internal/config_toml/{hash}`
///
/// Returns a config snapshot by hash in editable TOML form.
#[axum::debug_handler(state = SwappableAppStateData)]
#[instrument(name = "config.get_by_hash_toml", skip_all, fields(hash = %hash))]
pub async fn get_config_toml_by_hash_handler(
    State(app_state): AppState,
    Path(hash): Path<String>,
) -> Result<Json<GetConfigTomlResponse>, Error> {
    let snapshot_hash: SnapshotHash = hash.parse().map_err(|_| {
        Error::new(ErrorDetails::ConfigSnapshotNotFound {
            snapshot_hash: hash.clone(),
        })
    })?;

    let db = app_state.get_delegating_database();
    let snapshot = db.get_config_snapshot(snapshot_hash).await?;

    Ok(Json(GetConfigTomlResponse::from_snapshot(snapshot)?))
}

/// Request body for writing a config snapshot.
#[derive(Debug, Deserialize, Serialize)]
pub struct WriteConfigRequest {
    /// The config to write.
    pub config: UninitializedConfig,
    /// Templates that should be stored with the config.
    #[serde(default)]
    pub extra_templates: HashMap<String, String>,
    /// User-defined tags for categorizing this config snapshot.
    /// Tags are merged with any existing tags for the same config hash.
    #[serde(default)]
    pub tags: HashMap<String, String>,
}

/// Request body for validating editable config TOML.
#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Debug, Deserialize, Serialize)]
#[cfg_attr(feature = "ts-bindings", ts(export, optional_fields))]
pub struct ValidateConfigTomlRequest {
    /// Human-readable TOML config with path strings.
    pub toml: String,
    /// File-backed content referenced by the TOML, keyed by the path string in the TOML body.
    #[serde(default)]
    pub path_contents: HashMap<String, String>,
}

/// Response from validating editable config TOML.
#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Debug, Deserialize, Serialize)]
#[cfg_attr(feature = "ts-bindings", ts(export, optional_fields))]
pub struct ValidateConfigTomlResponse {
    pub valid: bool,
}

/// Response from writing a config snapshot.
#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Debug, Deserialize, Serialize)]
#[cfg_attr(feature = "ts-bindings", ts(export, optional_fields))]
pub struct WriteConfigResponse {
    /// The hash identifying this config version.
    pub hash: String,
}

/// Handler for `POST /internal/config`
///
/// Writes a config snapshot to the database and returns its hash.
/// If a config with the same hash already exists, tags are merged
/// (new tags override existing keys) and created_at is preserved.
///
/// The config is validated by running the full config loading pipeline
/// (with credential validation disabled) before writing. This catches
/// issues like invalid model references, missing templates, and
/// cross-reference errors that serde deserialization alone would miss.
#[axum::debug_handler(state = SwappableAppStateData)]
#[instrument(name = "config.write", skip_all)]
pub async fn write_config_handler(
    State(app_state): AppState,
    StructuredJson(request): StructuredJson<WriteConfigRequest>,
) -> Result<Json<WriteConfigResponse>, Error> {
    let mut snapshot = ConfigSnapshot::new(request.config, request.extra_templates)?;
    snapshot.tags = request.tags;

    let hash = snapshot.hash.to_string();

    app_state
        .validate_and_write_config_snapshot(&snapshot)
        .await?;

    Ok(Json(WriteConfigResponse { hash }))
}

/// Handler for `POST /internal/config_toml/validate`
///
/// Validates editable TOML by parsing it back into `UninitializedConfig` and
/// running the shared config-loading pipeline without persisting anything.
#[axum::debug_handler(state = SwappableAppStateData)]
#[instrument(name = "config.validate_toml", skip_all)]
pub async fn validate_config_toml_handler(
    StructuredJson(request): StructuredJson<ValidateConfigTomlRequest>,
) -> Result<Json<ValidateConfigTomlResponse>, Error> {
    validate_config_toml_request(&request).await?;
    Ok(Json(ValidateConfigTomlResponse { valid: true }))
}

async fn validate_config_toml_request(request: &ValidateConfigTomlRequest) -> Result<(), Error> {
    let config = toml_to_config(&request.toml, &request.path_contents)?;
    let _validated = Config::load_from_uninitialized(config, false).await?;
    Ok(())
}

/// Handler for `POST /internal/config_toml/apply`
///
/// Applies a full editable TOML document back to the stored-config tables using
/// a compare-and-swap against the current DB-authoritative editable snapshot.
#[axum::debug_handler(state = SwappableAppStateData)]
#[instrument(name = "config.apply_toml", skip_all)]
pub async fn apply_config_toml_handler(
    State(swap_state): State<SwappableAppStateData>,
    StructuredJson(request): StructuredJson<ApplyConfigTomlRequest>,
) -> Result<Json<ApplyConfigTomlResponse>, Error> {
    let app_state = swap_state.load_latest();
    let edited_config = toml_to_config(&request.toml, &request.path_contents)?;
    // Validate up front — the returned `UnwrittenConfig` is discarded because
    // after we commit we reload+hot-swap via `load_config_from_db` below.
    Config::load_from_uninitialized(edited_config.clone(), false).await?;
    let (canonical_toml, canonical_path_contents) = config_to_toml(&edited_config)?;
    let canonical_signature = editable_config_signature(&canonical_toml, &canonical_path_contents)?;

    let pool = app_state
        .postgres_connection_info
        .get_pool_result()
        .map_err(|e| e.log())?;
    let mut tx = pool.begin().await.map_err(|e| {
        Error::new(ErrorDetails::PostgresQuery {
            message: format!("Failed to start config TOML apply transaction: {e}"),
        })
    })?;
    acquire_config_editor_lock(&mut tx).await?;

    let current_uninitialized = load_config_from_db_in_tx(&mut tx)
        .await
        .map_err(merge_db_config_errors)?;
    let current_validated =
        Config::load_from_uninitialized(current_uninitialized.clone(), false).await?;
    let (current_toml, current_path_contents) = config_to_toml(&current_uninitialized)?;
    let current_signature = editable_config_signature(&current_toml, &current_path_contents)?;

    if current_signature != request.base_signature && current_signature != canonical_signature {
        return Err(Error::new(ErrorDetails::ConfigCompareAndSwapConflict {
            message: "Config changed underneath you, reload the latest snapshot before applying your edits.".to_string(),
        }));
    }

    if current_signature == canonical_signature {
        tx.rollback().await.map_err(|e| {
            Error::new(ErrorDetails::PostgresQuery {
                message: format!("Failed to roll back no-op config TOML apply transaction: {e}"),
            })
        })?;
        return Ok(Json(ApplyConfigTomlResponse {
            toml: current_toml,
            path_contents: current_path_contents,
            hash: current_validated.hash.to_string(),
            base_signature: current_signature,
        }));
    }

    let extra_templates = HashMap::new();
    write_stored_config_in_tx(
        &mut tx,
        WriteStoredConfigParams {
            config: &edited_config,
            creation_source: "ui-config-editor",
            source_autopilot_session_id: None,
            extra_templates: &extra_templates,
        },
    )
    .await?;

    tx.commit().await.map_err(|e| {
        Error::new(ErrorDetails::PostgresQuery {
            message: format!("Failed to commit config TOML apply transaction: {e}"),
        })
    })?;

    // Rehydrate the new config from the database and hot-swap it into the
    // gateway so subsequent requests see the edit without a restart. We
    // re-read from Postgres (rather than reusing `edited_validated`) to
    // exercise the same read path the gateway would use on a cold start,
    // catching any write/read round-trip issues immediately.
    let reloaded_uninitialized = load_config_from_db(pool)
        .await
        .map_err(merge_db_config_errors)?;
    let reloaded_unwritten =
        Config::load_from_uninitialized(reloaded_uninitialized.clone(), false).await?;
    // The snapshot has already been persisted via `write_stored_config_in_tx`
    // above, so we consume the `UnwrittenConfig` without going through the
    // `write_config_snapshot` path again.
    let reloaded_config = reloaded_unwritten.dangerous_into_config_without_writing();
    let reloaded_hash = reloaded_config.hash.to_string();
    swap_state.config.store(Arc::new(reloaded_config));
    swap_state
        .uninitialized_config
        .store(Arc::new(reloaded_uninitialized));

    Ok(Json(ApplyConfigTomlResponse {
        toml: canonical_toml,
        path_contents: canonical_path_contents,
        hash: reloaded_hash,
        base_signature: canonical_signature,
    }))
}

#[cfg(test)]
mod tests {
    use googletest::prelude::*;
    use tokio::runtime::Runtime;

    use crate::config::UninitializedToolConfig;
    use crate::config::gateway::UninitializedGatewayConfig;
    use crate::config::path::ResolvedTomlPathData;

    use super::*;

    #[gtest]
    fn config_toml_response_uses_plain_path_strings() {
        let config = UninitializedConfig {
            gateway: Some(UninitializedGatewayConfig {
                bind_address: Some(
                    "127.0.0.1:4000"
                        .parse()
                        .expect("bind address should parse for TOML response test"),
                ),
                ..Default::default()
            }),
            clickhouse: Default::default(),
            postgres: Default::default(),
            rate_limiting: Default::default(),
            object_storage: None,
            models: Some(HashMap::new()),
            embedding_models: Some(HashMap::new()),
            functions: Some(HashMap::new()),
            metrics: Some(HashMap::new()),
            tools: Some(HashMap::from([(
                "weather".to_string(),
                UninitializedToolConfig {
                    description: "Weather tool".to_string(),
                    parameters: ResolvedTomlPathData::new_fake_path(
                        "tools/weather.json".to_string(),
                        "{\"type\":\"object\"}".to_string(),
                    ),
                    name: None,
                    strict: false,
                },
            )])),
            evaluations: Some(HashMap::new()),
            provider_types: Default::default(),
            optimizers: Some(HashMap::new()),
            autopilot: Default::default(),
        };
        let snapshot =
            ConfigSnapshot::new(config, HashMap::new()).expect("snapshot creation should succeed");

        let response = GetConfigTomlResponse::from_snapshot(snapshot)
            .expect("TOML response generation should succeed");

        expect_that!(
            response.toml.as_str(),
            contains_substring("bind_address = \"127.0.0.1:4000\"")
        );
        expect_that!(
            response.toml.as_str(),
            contains_substring("parameters = \"tools/weather.json\"")
        );
        expect_that!(
            response.path_contents.get("tools/weather.json"),
            some(eq(&"{\"type\":\"object\"}".to_string()))
        );
    }

    #[gtest]
    fn validate_config_toml_request_accepts_round_trip_output() {
        let config = UninitializedConfig {
            gateway: Some(UninitializedGatewayConfig {
                bind_address: Some(
                    "127.0.0.1:4000"
                        .parse()
                        .expect("bind address should parse for validation test"),
                ),
                ..Default::default()
            }),
            clickhouse: Default::default(),
            postgres: Default::default(),
            rate_limiting: Default::default(),
            object_storage: None,
            models: Some(HashMap::new()),
            embedding_models: Some(HashMap::new()),
            functions: Some(HashMap::new()),
            metrics: Some(HashMap::new()),
            tools: Some(HashMap::from([(
                "weather".to_string(),
                UninitializedToolConfig {
                    description: "Weather tool".to_string(),
                    parameters: ResolvedTomlPathData::new_fake_path(
                        "tools/weather.json".to_string(),
                        "{\"type\":\"object\"}".to_string(),
                    ),
                    name: None,
                    strict: false,
                },
            )])),
            evaluations: Some(HashMap::new()),
            provider_types: Default::default(),
            optimizers: Some(HashMap::new()),
            autopilot: Default::default(),
        };
        let (toml, path_contents) =
            config_to_toml(&config).expect("editable TOML serialization should succeed");
        let request = ValidateConfigTomlRequest {
            toml,
            path_contents,
        };

        let runtime = Runtime::new().expect("tokio runtime should build for validation test");
        expect_that!(
            runtime.block_on(validate_config_toml_request(&request)),
            ok(eq(&()))
        );
    }
}
