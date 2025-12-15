//! Config snapshot endpoints.
//!
//! These endpoints allow retrieving config snapshots by hash, or the live config.

use std::collections::HashMap;

use axum::Json;
use axum::extract::{Path, State};
use serde::{Deserialize, Serialize};
use tracing::instrument;

use crate::config::snapshot::{ConfigSnapshot, SnapshotHash};
use crate::config::stored::StoredConfig;
use crate::db::ConfigQueries;
use crate::error::{Error, ErrorDetails};
use crate::utils::gateway::{AppState, AppStateData};

/// Response containing a config snapshot.
#[derive(Debug, Serialize, Deserialize)]
pub struct GetConfigResponse {
    /// The config in a form suitable for serialization.
    pub config: StoredConfig,
    /// The hash identifying this config version.
    pub hash: String,
    /// Templates that were loaded from the filesystem.
    pub extra_templates: HashMap<String, String>,
    /// User-defined tags for categorizing or labeling this config snapshot.
    pub tags: HashMap<String, String>,
}

impl GetConfigResponse {
    fn from_snapshot(snapshot: ConfigSnapshot) -> Self {
        Self {
            hash: snapshot.hash.to_string(),
            config: snapshot.config,
            extra_templates: snapshot.extra_templates,
            tags: snapshot.tags,
        }
    }
}

/// Handler for `GET /internal/config`
///
/// Returns the live config snapshot.
#[axum::debug_handler(state = AppStateData)]
#[instrument(name = "config.get_live", skip_all)]
pub async fn get_live_config_handler(
    State(app_state): AppState,
) -> Result<Json<GetConfigResponse>, Error> {
    let hash = app_state.config.hash.clone();
    let snapshot = app_state
        .clickhouse_connection_info
        .get_config_snapshot(hash)
        .await?;

    Ok(Json(GetConfigResponse::from_snapshot(snapshot)))
}

/// Handler for `GET /internal/config/{hash}`
///
/// Returns a config snapshot by hash.
#[axum::debug_handler(state = AppStateData)]
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

    let snapshot = app_state
        .clickhouse_connection_info
        .get_config_snapshot(snapshot_hash)
        .await?;

    Ok(Json(GetConfigResponse::from_snapshot(snapshot)))
}
