//! Shared test utilities for autopilot-tools integration tests.

// Allow dead code at module level since different test binaries use different subsets of these utilities
#![allow(dead_code, clippy::allow_attributes)]

use std::collections::HashMap;

use durable_tools::{TensorZeroClient, TensorZeroClientError};
use mockall::mock;
use tensorzero::{
    ClientInferenceParams, CreateDatapointRequest, CreateDatapointsFromInferenceRequestParams,
    CreateDatapointsResponse, DeleteDatapointsResponse, GetConfigResponse, GetDatapointsResponse,
    InferenceResponse, ListDatapointsRequest, Role, UpdateDatapointRequest,
    UpdateDatapointsResponse, Usage, WriteConfigRequest, WriteConfigResponse,
};
use tensorzero_core::config::snapshot::SnapshotHash;
use tensorzero_core::endpoints::datasets::{ChatInferenceDatapoint, Datapoint};
use tensorzero_core::endpoints::inference::ChatInferenceResponse;
use tensorzero_core::inference::types::{ContentBlockChatOutput, Input, InputMessage, Text};
use tensorzero_core::tool::DynamicToolParams;
use uuid::Uuid;

// Generate mock using mockall's mock! macro
mock! {
    pub TensorZeroClient {}

    #[async_trait::async_trait]
    impl TensorZeroClient for TensorZeroClient {
        async fn inference(
            &self,
            params: ClientInferenceParams,
        ) -> Result<InferenceResponse, TensorZeroClientError>;

        async fn create_autopilot_event(
            &self,
            session_id: Uuid,
            request: durable_tools::CreateEventRequest,
        ) -> Result<durable_tools::CreateEventResponse, TensorZeroClientError>;

        async fn list_autopilot_events(
            &self,
            session_id: Uuid,
            params: durable_tools::ListEventsParams,
        ) -> Result<durable_tools::ListEventsResponse, TensorZeroClientError>;

        async fn list_autopilot_sessions(
            &self,
            params: durable_tools::ListSessionsParams,
        ) -> Result<durable_tools::ListSessionsResponse, TensorZeroClientError>;

        async fn action(
            &self,
            snapshot_hash: SnapshotHash,
            input: tensorzero::ActionInput,
        ) -> Result<InferenceResponse, TensorZeroClientError>;

        async fn get_config_snapshot(
            &self,
            hash: Option<String>,
        ) -> Result<GetConfigResponse, TensorZeroClientError>;

        async fn write_config(
            &self,
            request: WriteConfigRequest,
        ) -> Result<WriteConfigResponse, TensorZeroClientError>;

        async fn create_datapoints(
            &self,
            dataset_name: String,
            datapoints: Vec<CreateDatapointRequest>,
        ) -> Result<CreateDatapointsResponse, TensorZeroClientError>;

        async fn create_datapoints_from_inferences(
            &self,
            dataset_name: String,
            params: CreateDatapointsFromInferenceRequestParams,
        ) -> Result<CreateDatapointsResponse, TensorZeroClientError>;

        async fn list_datapoints(
            &self,
            dataset_name: String,
            request: ListDatapointsRequest,
        ) -> Result<GetDatapointsResponse, TensorZeroClientError>;

        async fn get_datapoints(
            &self,
            dataset_name: Option<String>,
            ids: Vec<Uuid>,
        ) -> Result<GetDatapointsResponse, TensorZeroClientError>;

        async fn update_datapoints(
            &self,
            dataset_name: String,
            datapoints: Vec<UpdateDatapointRequest>,
        ) -> Result<UpdateDatapointsResponse, TensorZeroClientError>;

        async fn delete_datapoints(
            &self,
            dataset_name: String,
            ids: Vec<Uuid>,
        ) -> Result<DeleteDatapointsResponse, TensorZeroClientError>;
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

// ===== Datapoint Mock Response Factories =====

/// Create a mock CreateDatapointsResponse with the given IDs.
pub fn create_mock_create_datapoints_response(ids: Vec<Uuid>) -> CreateDatapointsResponse {
    CreateDatapointsResponse { ids }
}

/// Create a mock GetDatapointsResponse with the given datapoints.
pub fn create_mock_get_datapoints_response(datapoints: Vec<Datapoint>) -> GetDatapointsResponse {
    GetDatapointsResponse { datapoints }
}

/// Create a mock UpdateDatapointsResponse with the given IDs.
pub fn create_mock_update_datapoints_response(ids: Vec<Uuid>) -> UpdateDatapointsResponse {
    UpdateDatapointsResponse { ids }
}

/// Create a mock DeleteDatapointsResponse with the given count.
pub fn create_mock_delete_datapoints_response(num_deleted: u64) -> DeleteDatapointsResponse {
    DeleteDatapointsResponse {
        num_deleted_datapoints: num_deleted,
    }
}

/// Create a mock chat datapoint for testing.
pub fn create_mock_chat_datapoint(id: Uuid, dataset_name: &str, function_name: &str) -> Datapoint {
    Datapoint::Chat(ChatInferenceDatapoint {
        dataset_name: dataset_name.to_string(),
        function_name: function_name.to_string(),
        id,
        episode_id: Some(Uuid::now_v7()),
        input: Input {
            system: None,
            messages: vec![InputMessage {
                role: Role::User,
                content: vec![
                    tensorzero_core::inference::types::InputMessageContent::Text(Text {
                        text: "test input".to_string(),
                    }),
                ],
            }],
        },
        output: Some(vec![ContentBlockChatOutput::Text(Text {
            text: "test output".to_string(),
        })]),
        tool_params: DynamicToolParams::default(),
        tags: Some(HashMap::new()),
        auxiliary: String::new(),
        is_deleted: false,
        is_custom: false,
        source_inference_id: None,
        staled_at: None,
        updated_at: "2024-01-01T00:00:00Z".to_string(),
        name: None,
    })
}

/// Create a simple test Input for use in tests.
pub fn create_test_input(text: &str) -> Input {
    Input {
        system: None,
        messages: vec![InputMessage {
            role: Role::User,
            content: vec![
                tensorzero_core::inference::types::InputMessageContent::Text(Text {
                    text: text.to_string(),
                }),
            ],
        }],
    }
}
