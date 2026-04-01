//! Config snapshot endpoints.
//!
//! These endpoints allow retrieving config snapshots by hash, or the live config,
//! and writing new config snapshots.

use std::collections::HashMap;

use axum::Json;
use axum::extract::{Path, State};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tracing::instrument;

use crate::config::editable::config_to_toml;
use crate::config::editable::toml_to_config;
use crate::config::snapshot::{ConfigSnapshot, SnapshotHash};
use crate::config::{Config, UninitializedConfig};
use crate::db::ConfigQueries;
use crate::error::{Error, ErrorDetails};
use crate::utils::gateway::{AppState, StructuredJson, SwappableAppStateData};

/// Response containing a config snapshot.
#[derive(Debug, Serialize, Deserialize)]
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
    fn from_snapshot(snapshot: ConfigSnapshot) -> Result<Self, Error> {
        let uninitialized: UninitializedConfig =
            snapshot.config.try_into().map_err(|e: &'static str| {
                Error::new(ErrorDetails::Config {
                    message: e.to_string(),
                })
            })?;
        let config = serde_json::to_value(&uninitialized).map_err(|e| {
            Error::new(ErrorDetails::Config {
                message: format!("Failed to serialize config: {e}"),
            })
        })?;
        Ok(Self {
            hash: snapshot.hash.to_string(),
            config,
            extra_templates: snapshot.extra_templates,
            tags: snapshot.tags,
        })
    }
}

/// Response containing a TOML-editable config snapshot.
#[derive(Debug, Serialize, Deserialize)]
pub struct GetConfigTomlResponse {
    /// Human-readable TOML config with path strings instead of inlined file contents.
    pub toml: String,
    /// File-backed content referenced by the TOML, keyed by the path string in the TOML body.
    pub path_contents: HashMap<String, String>,
    /// The hash identifying this config version.
    pub hash: String,
    /// User-defined tags for categorizing or labeling this config snapshot.
    pub tags: HashMap<String, String>,
}

impl GetConfigTomlResponse {
    fn from_snapshot(snapshot: ConfigSnapshot) -> Result<Self, Error> {
        let uninitialized: UninitializedConfig =
            snapshot.config.try_into().map_err(|e: &'static str| {
                Error::new(ErrorDetails::Config {
                    message: e.to_string(),
                })
            })?;
        let (toml, path_contents) = config_to_toml(&uninitialized)?;
        Ok(Self {
            toml,
            path_contents,
            hash: snapshot.hash.to_string(),
            tags: snapshot.tags,
        })
    }
}

/// Handler for `GET /internal/config`
///
/// Returns the live config snapshot.
#[axum::debug_handler(state = SwappableAppStateData)]
#[instrument(name = "config.get_live", skip_all)]
pub async fn get_live_config_handler(
    State(app_state): AppState,
) -> Result<Json<GetConfigResponse>, Error> {
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
#[derive(Debug, Deserialize, Serialize)]
pub struct ValidateConfigTomlRequest {
    /// Human-readable TOML config with path strings.
    pub toml: String,
    /// File-backed content referenced by the TOML, keyed by the path string in the TOML body.
    #[serde(default)]
    pub path_contents: HashMap<String, String>,
}

/// Response from validating editable config TOML.
#[derive(Debug, Deserialize, Serialize)]
pub struct ValidateConfigTomlResponse {
    pub valid: bool,
}

/// Response from writing a config snapshot.
#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Debug, Deserialize, Serialize)]
#[cfg_attr(feature = "ts-bindings", ts(export))]
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
