//! Shared test utilities for autopilot-tools integration tests.

use std::sync::Arc;

use async_trait::async_trait;
use durable_tools::{InferenceClient, InferenceError};
use tensorzero::{ClientInferenceParams, InferenceResponse, Usage};
use tensorzero_core::config::snapshot::SnapshotHash;
use tensorzero_core::endpoints::inference::ChatInferenceResponse;
use tensorzero_core::inference::types::{ContentBlockChatOutput, Text};
use tokio::sync::Mutex;
use uuid::Uuid;

/// Mock InferenceClient that captures inference calls for verification.
pub struct MockInferenceClient {
    captured_inference_params: Arc<Mutex<Option<ClientInferenceParams>>>,
    captured_action_params: Arc<Mutex<Option<(SnapshotHash, ClientInferenceParams)>>>,
    response: InferenceResponse,
}

impl MockInferenceClient {
    pub fn new(response: InferenceResponse) -> Self {
        Self {
            captured_inference_params: Arc::new(Mutex::new(None)),
            captured_action_params: Arc::new(Mutex::new(None)),
            response,
        }
    }

    pub async fn get_captured_inference_params(&self) -> Option<ClientInferenceParams> {
        self.captured_inference_params.lock().await.clone()
    }

    pub async fn get_captured_action_params(
        &self,
    ) -> Option<(SnapshotHash, ClientInferenceParams)> {
        self.captured_action_params.lock().await.clone()
    }
}

#[async_trait]
impl InferenceClient for MockInferenceClient {
    async fn inference(
        &self,
        params: ClientInferenceParams,
    ) -> Result<InferenceResponse, InferenceError> {
        *self.captured_inference_params.lock().await = Some(params);
        Ok(self.response.clone())
    }

    async fn action(
        &self,
        snapshot_hash: SnapshotHash,
        params: ClientInferenceParams,
    ) -> Result<InferenceResponse, InferenceError> {
        *self.captured_action_params.lock().await = Some((snapshot_hash, params));
        Ok(self.response.clone())
    }

    async fn create_autopilot_event(
        &self,
        _session_id: Uuid,
        _request: durable_tools::CreateEventRequest,
    ) -> Result<durable_tools::CreateEventResponse, InferenceError> {
        Err(InferenceError::AutopilotUnavailable)
    }

    async fn list_autopilot_events(
        &self,
        _session_id: Uuid,
        _params: durable_tools::ListEventsParams,
    ) -> Result<durable_tools::ListEventsResponse, InferenceError> {
        Err(InferenceError::AutopilotUnavailable)
    }

    async fn list_autopilot_sessions(
        &self,
        _params: durable_tools::ListSessionsParams,
    ) -> Result<durable_tools::ListSessionsResponse, InferenceError> {
        Err(InferenceError::AutopilotUnavailable)
    }
}

/// Create a mock chat inference response with the given text content.
pub fn create_mock_chat_response(text: &str) -> InferenceResponse {
    InferenceResponse::Chat(ChatInferenceResponse {
        inference_id: Uuid::now_v7(),
        episode_id: Uuid::now_v7(),
        variant_name: "test_variant".to_string(),
        content: vec![ContentBlockChatOutput::Text(Text {
            text: text.to_string(),
        })],
        usage: Usage {
            input_tokens: Some(10),
            output_tokens: Some(5),
        },
        original_response: None,
        finish_reason: None,
    })
}
