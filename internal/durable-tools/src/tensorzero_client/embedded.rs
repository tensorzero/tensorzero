//! Embedded TensorZero client that uses gateway state directly.
//!
//! This implementation is used when the worker runs inside the gateway process
//! and wants to call inference and autopilot endpoints without HTTP overhead.

use async_trait::async_trait;
use tensorzero::{
    ActionResponse, ClientInferenceParams, CreateDatapointRequest,
    CreateDatapointsFromInferenceRequestParams, CreateDatapointsResponse, DeleteDatapointsResponse,
    GetDatapointsResponse, InferenceOutput, InferenceResponse, ListDatapointsRequest,
    TensorZeroError, UpdateDatapointRequest, UpdateDatapointsResponse,
};
use tensorzero_core::config::snapshot::SnapshotHash;
use tensorzero_core::endpoints::datasets::v1::types::{
    CreateDatapointsFromInferenceRequest, CreateDatapointsRequest, DeleteDatapointsRequest,
    GetDatapointsRequest, UpdateDatapointsRequest,
};
use tensorzero_core::endpoints::feedback::internal::LatestFeedbackIdByMetricResponse;
use tensorzero_core::endpoints::inference::inference;
use tensorzero_core::endpoints::internal::action::{ActionInput, ActionInputInfo, action};
use tensorzero_core::endpoints::internal::autopilot::{create_event, list_events, list_sessions};
use tensorzero_core::utils::gateway::AppStateData;
use uuid::Uuid;

use super::{
    CreateEventRequest, CreateEventResponse, ListEventsParams, ListEventsResponse,
    ListSessionsParams, ListSessionsResponse, TensorZeroClient, TensorZeroClientError,
};

/// TensorZero client that uses an existing gateway's state directly.
///
/// This is used when the worker runs inside the gateway process and wants to
/// call inference and autopilot endpoints without HTTP overhead.
pub struct EmbeddedClient {
    app_state: AppStateData,
}

impl EmbeddedClient {
    /// Create a new embedded client from gateway state.
    pub fn new(app_state: AppStateData) -> Self {
        Self { app_state }
    }
}

#[async_trait]
impl TensorZeroClient for EmbeddedClient {
    async fn inference(
        &self,
        params: ClientInferenceParams,
    ) -> Result<InferenceResponse, TensorZeroClientError> {
        let internal_params = params
            .try_into()
            .map_err(|e: tensorzero_core::error::Error| {
                TensorZeroClientError::TensorZero(TensorZeroError::Other { source: e.into() })
            })?;

        let result = Box::pin(inference(
            self.app_state.config.clone(),
            &self.app_state.http_client,
            self.app_state.clickhouse_connection_info.clone(),
            self.app_state.postgres_connection_info.clone(),
            self.app_state.deferred_tasks.clone(),
            internal_params,
            None, // No API key in embedded mode
        ))
        .await
        .map_err(|e| {
            TensorZeroClientError::TensorZero(TensorZeroError::Other { source: e.into() })
        })?;

        match result.output {
            InferenceOutput::NonStreaming(response) => Ok(response),
            InferenceOutput::Streaming(_) => Err(TensorZeroClientError::StreamingNotSupported),
        }
    }

    async fn create_autopilot_event(
        &self,
        session_id: Uuid,
        request: CreateEventRequest,
    ) -> Result<CreateEventResponse, TensorZeroClientError> {
        let autopilot_client = self
            .app_state
            .autopilot_client
            .as_ref()
            .ok_or(TensorZeroClientError::AutopilotUnavailable)?;

        create_event(autopilot_client, session_id, request)
            .await
            .map_err(|e| {
                TensorZeroClientError::TensorZero(TensorZeroError::Other { source: e.into() })
            })
    }

    async fn list_autopilot_events(
        &self,
        session_id: Uuid,
        params: ListEventsParams,
    ) -> Result<ListEventsResponse, TensorZeroClientError> {
        let autopilot_client = self
            .app_state
            .autopilot_client
            .as_ref()
            .ok_or(TensorZeroClientError::AutopilotUnavailable)?;

        list_events(autopilot_client, session_id, params)
            .await
            .map_err(|e| {
                TensorZeroClientError::TensorZero(TensorZeroError::Other { source: e.into() })
            })
    }

    async fn list_autopilot_sessions(
        &self,
        params: ListSessionsParams,
    ) -> Result<ListSessionsResponse, TensorZeroClientError> {
        let autopilot_client = self
            .app_state
            .autopilot_client
            .as_ref()
            .ok_or(TensorZeroClientError::AutopilotUnavailable)?;

        list_sessions(autopilot_client, params).await.map_err(|e| {
            TensorZeroClientError::TensorZero(TensorZeroError::Other { source: e.into() })
        })
    }

    async fn action(
        &self,
        snapshot_hash: SnapshotHash,
        input: ActionInput,
    ) -> Result<InferenceResponse, TensorZeroClientError> {
        let action_input = ActionInputInfo {
            snapshot_hash,
            input,
        };

        let response = action(&self.app_state, action_input).await.map_err(|e| {
            TensorZeroClientError::TensorZero(TensorZeroError::Other { source: e.into() })
        })?;

        match response {
            ActionResponse::Inference(r) => Ok(r),
            ActionResponse::Feedback(_) => {
                Err(TensorZeroClientError::TensorZero(TensorZeroError::Other {
                    source: tensorzero_core::error::Error::new(
                        tensorzero_core::error::ErrorDetails::InternalError {
                            message: "Unexpected feedback response from action endpoint"
                                .to_string(),
                        },
                    )
                    .into(),
                }))
            }
        }
    }

    // ========== Datapoint CRUD Operations ==========

    async fn create_datapoints(
        &self,
        dataset_name: String,
        datapoints: Vec<CreateDatapointRequest>,
    ) -> Result<CreateDatapointsResponse, TensorZeroClientError> {
        let request = CreateDatapointsRequest { datapoints };

        tensorzero_core::endpoints::datasets::v1::create_datapoints(
            &self.app_state.config,
            &self.app_state.http_client,
            &self.app_state.clickhouse_connection_info,
            &dataset_name,
            request,
        )
        .await
        .map_err(|e| TensorZeroClientError::TensorZero(TensorZeroError::Other { source: e.into() }))
    }

    async fn create_datapoints_from_inferences(
        &self,
        dataset_name: String,
        params: CreateDatapointsFromInferenceRequestParams,
    ) -> Result<CreateDatapointsResponse, TensorZeroClientError> {
        let request = CreateDatapointsFromInferenceRequest { params };

        tensorzero_core::endpoints::datasets::v1::create_from_inferences(
            &self.app_state.config,
            &self.app_state.clickhouse_connection_info,
            dataset_name,
            request,
        )
        .await
        .map_err(|e| TensorZeroClientError::TensorZero(TensorZeroError::Other { source: e.into() }))
    }

    async fn list_datapoints(
        &self,
        dataset_name: String,
        request: ListDatapointsRequest,
    ) -> Result<GetDatapointsResponse, TensorZeroClientError> {
        tensorzero_core::endpoints::datasets::v1::list_datapoints(
            &self.app_state.clickhouse_connection_info,
            dataset_name,
            request,
        )
        .await
        .map_err(|e| TensorZeroClientError::TensorZero(TensorZeroError::Other { source: e.into() }))
    }

    async fn get_datapoints(
        &self,
        dataset_name: Option<String>,
        ids: Vec<Uuid>,
    ) -> Result<GetDatapointsResponse, TensorZeroClientError> {
        let request = GetDatapointsRequest { ids };

        tensorzero_core::endpoints::datasets::v1::get_datapoints(
            &self.app_state.clickhouse_connection_info,
            dataset_name,
            request,
        )
        .await
        .map_err(|e| TensorZeroClientError::TensorZero(TensorZeroError::Other { source: e.into() }))
    }

    async fn update_datapoints(
        &self,
        dataset_name: String,
        datapoints: Vec<UpdateDatapointRequest>,
    ) -> Result<UpdateDatapointsResponse, TensorZeroClientError> {
        let request = UpdateDatapointsRequest { datapoints };

        tensorzero_core::endpoints::datasets::v1::update_datapoints(
            &self.app_state,
            &dataset_name,
            request,
        )
        .await
        .map_err(|e| TensorZeroClientError::TensorZero(TensorZeroError::Other { source: e.into() }))
    }

    async fn delete_datapoints(
        &self,
        dataset_name: String,
        ids: Vec<Uuid>,
    ) -> Result<DeleteDatapointsResponse, TensorZeroClientError> {
        let request = DeleteDatapointsRequest { ids };

        tensorzero_core::endpoints::datasets::v1::delete_datapoints(
            &self.app_state.clickhouse_connection_info,
            &dataset_name,
            request,
        )
        .await
        .map_err(|e| TensorZeroClientError::TensorZero(TensorZeroError::Other { source: e.into() }))
    }

    // ========== Optimization Operations ==========

    async fn launch_optimization_workflow(
        &self,
        params: tensorzero_optimizers::endpoints::LaunchOptimizationWorkflowParams,
    ) -> Result<String, TensorZeroClientError> {
        let job_handle = tensorzero_optimizers::endpoints::launch_optimization_workflow(
            &self.app_state.http_client,
            self.app_state.config.clone(),
            &self.app_state.clickhouse_connection_info,
            params,
        )
        .await
        .map_err(|e| {
            TensorZeroClientError::TensorZero(TensorZeroError::Other { source: e.into() })
        })?;

        job_handle.to_base64_urlencoded().map_err(|e| {
            TensorZeroClientError::TensorZero(TensorZeroError::Other { source: e.into() })
        })
    }

    async fn poll_optimization(
        &self,
        job_handle: String,
    ) -> Result<super::OptimizationJobInfo, TensorZeroClientError> {
        let job_handle =
            tensorzero_core::optimization::OptimizationJobHandle::from_base64_urlencoded(
                &job_handle,
            )
            .map_err(|e| {
                TensorZeroClientError::TensorZero(TensorZeroError::Other { source: e.into() })
            })?;

        tensorzero_optimizers::endpoints::poll_optimization(
            &self.app_state.http_client,
            &job_handle,
            &self.app_state.config.models.default_credentials,
            &self.app_state.config.provider_types,
        )
        .await
        .map_err(|e| TensorZeroClientError::TensorZero(TensorZeroError::Other { source: e.into() }))
    }

    async fn get_latest_feedback_id_by_metric(
        &self,
        target_id: Uuid,
    ) -> Result<LatestFeedbackIdByMetricResponse, TensorZeroClientError> {
        tensorzero_core::endpoints::feedback::internal::get_latest_feedback_id_by_metric(
            &self.app_state.clickhouse_connection_info,
            target_id,
        )
        .await
        .map_err(|e| TensorZeroClientError::TensorZero(TensorZeroError::Other { source: e.into() }))
    }
}
