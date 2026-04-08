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

use crate::config::UninitializedConfig;
use crate::config::snapshot::{ConfigSnapshot, SnapshotHash};
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

/// Response from writing a config snapshot.
#[derive(ts_rs::TS, Debug, Deserialize, Serialize)]
#[ts(export)]
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
