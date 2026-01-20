//! Shared test utilities for durable-tools integration tests.

// Allow dead code at module level since different test binaries use different subsets of these utilities
#![allow(dead_code, clippy::allow_attributes)]

use durable_tools::{TensorZeroClient, TensorZeroClientError};
use mockall::mock;
use tensorzero::{
    ClientInferenceParams, CreateDatapointRequest, CreateDatapointsFromInferenceRequestParams,
    CreateDatapointsResponse, DeleteDatapointsResponse, FeedbackParams, FeedbackResponse,
    GetConfigResponse, GetDatapointsResponse, GetInferencesResponse, InferenceResponse,
    ListDatapointsRequest, ListInferencesRequest, UpdateDatapointRequest, UpdateDatapointsResponse,
    Usage, WriteConfigRequest, WriteConfigResponse,
};
use tensorzero_core::config::snapshot::SnapshotHash;
use tensorzero_core::db::feedback::FeedbackByVariant;
use tensorzero_core::endpoints::feedback::internal::LatestFeedbackIdByMetricResponse;
use tensorzero_core::endpoints::inference::ChatInferenceResponse;
use tensorzero_core::inference::types::{ContentBlockChatOutput, Text};
use tensorzero_core::optimization::{OptimizationJobHandle, OptimizationJobInfo};
use tensorzero_optimizers::endpoints::LaunchOptimizationWorkflowParams;
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

        async fn feedback(
            &self,
            params: FeedbackParams,
        ) -> Result<FeedbackResponse, TensorZeroClientError>;

        async fn create_autopilot_event(
            &self,
            session_id: Uuid,
            request: durable_tools::CreateEventGatewayRequest,
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

        async fn list_inferences(
            &self,
            request: ListInferencesRequest,
        ) -> Result<GetInferencesResponse, TensorZeroClientError>;

        async fn launch_optimization_workflow(
            &self,
            params: LaunchOptimizationWorkflowParams,
        ) -> Result<OptimizationJobHandle, TensorZeroClientError>;

        async fn poll_optimization(
            &self,
            job_handle: &OptimizationJobHandle,
        ) -> Result<OptimizationJobInfo, TensorZeroClientError>;

        async fn get_latest_feedback_id_by_metric(
            &self,
            target_id: Uuid,
        ) -> Result<LatestFeedbackIdByMetricResponse, TensorZeroClientError>;

        async fn get_feedback_by_variant(
            &self,
            metric_name: String,
            function_name: String,
            variant_names: Option<Vec<String>>,
        ) -> Result<Vec<FeedbackByVariant>, TensorZeroClientError>;

        async fn run_evaluation(
            &self,
            params: durable_tools::RunEvaluationParams,
        ) -> Result<durable_tools::RunEvaluationResponse, TensorZeroClientError>;
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
        raw_usage: None,
        original_response: None,
        finish_reason: None,
    })
}
