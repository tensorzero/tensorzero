//! Model inferences endpoint for getting model inference details by inference ID.

use axum::extract::{Path, State};
use axum::{Json, debug_handler};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use tracing::instrument;
use uuid::Uuid;

use crate::db::model_inferences::ModelInferenceQueries;
use crate::error::{Error, ErrorDetails};
use crate::inference::types::{ContentBlockOutput, StoredRequestMessage};
use crate::utils::gateway::{AppState, AppStateData};

/// Response containing model inferences
#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Debug, Serialize, Deserialize)]
#[cfg_attr(feature = "ts-bindings", ts(export))]
pub struct GetModelInferencesResponse {
    pub model_inferences: Vec<ModelInference>,
}

// NOTE(shuyangli): Internal-only until we sort out `input_messages` types.
/// Wire type for a single ModelInference (raw request and response sent to a model).
#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Debug, Serialize, Deserialize)]
#[cfg_attr(feature = "ts-bindings", ts(export, optional_fields))]
pub struct ModelInference {
    /// Unique identifier for the ModelInference.
    pub id: Uuid,

    /// Unique identifier for the inference.
    pub inference_id: Uuid,

    /// Raw request sent to the model.
    pub raw_request: String,

    /// Raw response received from the model.
    pub raw_response: String,

    /// Name of the model used for the inference.
    pub model_name: String,

    /// Name of the model provider used for the inference.
    pub model_provider_name: String,

    /// Number of input tokens used for the inference.
    /// This may be missing if the model inference provider does not report token usage.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub input_tokens: Option<u32>,

    /// Number of output tokens used for the inference.
    /// This may be missing if the model inference provider does not report token usage.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output_tokens: Option<u32>,

    /// Response time in milliseconds.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response_time_ms: Option<u32>,

    /// Time to first token in milliseconds.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ttft_ms: Option<u32>,

    /// Timestamp of the inference.
    pub timestamp: DateTime<Utc>,

    /// System prompt used for the inference.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system: Option<String>,

    // TODO(shuyangli): Figure out if this should be a different message type, since we should not send Stored* types in API.
    /// Input messages sent to the model.
    pub input_messages: Vec<StoredRequestMessage>,

    /// Output content blocks from the model.
    pub output: Vec<ContentBlockOutput>,

    /// Whether the inference was cached.
    pub cached: bool,
}

/// HTTP handler for getting model inferences by inference ID
#[debug_handler(state = AppStateData)]
#[instrument(
    name = "get_model_inferences_handler",
    skip_all,
    fields(
        inference_id = %inference_id,
    )
)]
pub async fn get_model_inferences_handler(
    State(app_state): AppState,
    Path(inference_id): Path<Uuid>,
) -> Result<Json<GetModelInferencesResponse>, Error> {
    let model_inferences = get_model_inferences(app_state, inference_id).await?;
    Ok(Json(GetModelInferencesResponse { model_inferences }))
}

/// Core business logic for getting model inferences
async fn get_model_inferences(
    AppStateData {
        clickhouse_connection_info,
        ..
    }: AppStateData,
    inference_id: Uuid,
) -> Result<Vec<ModelInference>, Error> {
    let rows = clickhouse_connection_info
        .get_model_inferences_by_inference_id(inference_id)
        .await?;

    // Convert StoredModelInference to wire type ModelInference
    rows.into_iter()
        .map(|row| {
            // Parse the timestamp from the materialized column
            let timestamp = row
                .timestamp
                .as_ref()
                .ok_or_else(|| {
                    Error::new(ErrorDetails::ClickHouseDeserialization {
                        message: "timestamp field is missing".to_string(),
                    })
                })
                .and_then(|ts| {
                    DateTime::parse_from_rfc3339(ts)
                        .map_err(|e| {
                            Error::new(ErrorDetails::ClickHouseDeserialization {
                                message: format!("Failed to parse timestamp: {e}"),
                            })
                        })
                        .map(|dt| dt.with_timezone(&Utc))
                })?;

            Ok(ModelInference {
                id: row.id,
                inference_id: row.inference_id,
                raw_request: row.raw_request,
                raw_response: row.raw_response,
                model_name: row.model_name,
                model_provider_name: row.model_provider_name,
                input_tokens: row.input_tokens,
                output_tokens: row.output_tokens,
                response_time_ms: row.response_time_ms,
                ttft_ms: row.ttft_ms,
                timestamp,
                system: row.system,
                input_messages: row.input_messages,
                output: row.output,
                cached: row.cached,
            })
        })
        .collect()
}
