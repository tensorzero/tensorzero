//! Action endpoint types and helpers for executing inference or feedback with historical config snapshots.
//!
//! This module provides type definitions and config loading utilities for the action endpoint.
//! The action dispatch logic lives in the gateway.

use std::sync::Arc;

use serde::{Deserialize, Serialize};
use tensorzero_derive::TensorZeroDeserialize;

use crate::client::client_inference_params::ClientInferenceParams;
use crate::config::snapshot::SnapshotHash;
use crate::config::{Config, RuntimeOverlay};
use crate::db::ConfigQueries;
use crate::endpoints::feedback::{FeedbackResponse, Params as FeedbackParams};
use crate::endpoints::inference::InferenceResponse;
use crate::error::{Error, ErrorDetails};
use crate::utils::gateway::AppStateData;

/// Input for the action endpoint.
#[derive(Debug, Deserialize, Serialize)]
pub struct ActionInputInfo {
    /// The snapshot hash identifying which config version to use.
    pub snapshot_hash: SnapshotHash,
    /// The action to perform (inference or feedback).
    #[serde(flatten)]
    pub input: ActionInput,
}

/// The specific action type to execute.
#[derive(Clone, Debug, Serialize, TensorZeroDeserialize)]
#[serde(tag = "type")]
#[serde(rename_all = "snake_case")]
pub enum ActionInput {
    Inference(Box<ClientInferenceParams>),
    Feedback(Box<FeedbackParams>),
}

/// Response from the action endpoint.
#[derive(Debug, Serialize, Deserialize)]
#[serde(untagged)]
#[expect(clippy::large_enum_variant)]
pub enum ActionResponse {
    Inference(InferenceResponse),
    Feedback(FeedbackResponse),
}

/// Get config from cache or load from snapshot.
///
/// This helper is used by the gateway's action handler to load historical
/// config snapshots for reproducible inference and feedback execution.
pub async fn get_or_load_config(
    app_state: &AppStateData,
    snapshot_hash: &SnapshotHash,
) -> Result<Arc<Config>, Error> {
    let cache = app_state.config_snapshot_cache.as_ref().ok_or_else(|| {
        Error::new(ErrorDetails::InvalidRequest {
            message: "Config snapshot cache is not enabled".to_string(),
        })
    })?;

    // Cache hit
    if let Some(config) = cache.get(snapshot_hash) {
        return Ok(config);
    }

    // Cache miss: load from ClickHouse
    let snapshot = app_state
        .clickhouse_connection_info
        .get_config_snapshot(snapshot_hash.clone())
        .await?;

    let runtime_overlay = RuntimeOverlay::from_config(&app_state.config);

    let unwritten_config = Config::load_from_snapshot(
        snapshot,
        runtime_overlay,
        false, // Don't validate credentials for historical configs
    )
    .await?;

    let config = Arc::new(unwritten_config.dangerous_into_config_without_writing());

    cache.insert(snapshot_hash.clone(), config.clone());

    Ok(config)
}
