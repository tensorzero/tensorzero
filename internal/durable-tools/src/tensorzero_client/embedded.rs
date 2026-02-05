//! Embedded TensorZero client that uses gateway state directly.
//!
//! This implementation is used when the worker runs inside the gateway process
//! and wants to call inference and autopilot endpoints without HTTP overhead.

use async_trait::async_trait;
use autopilot_client::GatewayListEventsResponse;
use tensorzero::{
    ClientInferenceParams, CreateDatapointRequest, CreateDatapointsFromInferenceRequestParams,
    CreateDatapointsResponse, DeleteDatapointsResponse, FeedbackParams, FeedbackResponse,
    GetConfigResponse, GetDatapointsResponse, GetInferencesRequest, GetInferencesResponse,
    InferenceOutput, InferenceResponse, ListDatapointsRequest, ListDatasetsRequest,
    ListDatasetsResponse, ListInferencesRequest, TensorZeroError, UpdateDatapointRequest,
    UpdateDatapointsResponse, WriteConfigRequest, WriteConfigResponse,
};
use tensorzero_core::config::snapshot::{ConfigSnapshot, SnapshotHash};
use tensorzero_core::config::write_config_snapshot;
use tensorzero_core::db::ConfigQueries;
use tensorzero_core::db::feedback::FeedbackByVariant;
use tensorzero_core::db::feedback::FeedbackQueries;
use tensorzero_core::endpoints::datasets::v1::types::{
    CreateDatapointsFromInferenceRequest, CreateDatapointsRequest, DeleteDatapointsRequest,
    GetDatapointsRequest, UpdateDatapointsRequest,
};
use tensorzero_core::endpoints::feedback::feedback;
use tensorzero_core::endpoints::feedback::internal::LatestFeedbackIdByMetricResponse;
use tensorzero_core::endpoints::inference::inference;
use tensorzero_core::endpoints::internal::autopilot::{create_event, list_events, list_sessions};
use tensorzero_core::error::{Error, ErrorDetails};
use tensorzero_core::utils::gateway::AppStateData;
use uuid::Uuid;

use crate::action::{ActionInput, ActionInputInfo, ActionResponse};

use super::{
    CreateEventGatewayRequest, CreateEventResponse, ListEventsParams, ListSessionsParams,
    ListSessionsResponse, RunEvaluationParams, RunEvaluationResponse, TensorZeroClient,
    TensorZeroClientError,
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
            self.app_state.rate_limiting_manager.clone(),
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

    async fn feedback(
        &self,
        params: FeedbackParams,
    ) -> Result<FeedbackResponse, TensorZeroClientError> {
        feedback(self.app_state.clone(), params, None)
            .await
            .map_err(|e| {
                TensorZeroClientError::TensorZero(TensorZeroError::Other { source: e.into() })
            })
    }

    async fn create_autopilot_event(
        &self,
        session_id: Uuid,
        request: CreateEventGatewayRequest,
    ) -> Result<CreateEventResponse, TensorZeroClientError> {
        let autopilot_client = self
            .app_state
            .autopilot_client
            .as_ref()
            .ok_or(TensorZeroClientError::AutopilotUnavailable)?;

        // Get deployment_id from app_state
        let deployment_id = self
            .app_state
            .deployment_id
            .clone()
            .ok_or(TensorZeroClientError::AutopilotUnavailable)?;

        // Construct the full request with deployment_id from app state
        // If starting a new session (nil session_id), include the current config hash
        let config_snapshot_hash = if session_id.is_nil() {
            Some(self.app_state.config.hash.to_string())
        } else {
            None
        };
        let full_request = autopilot_client::CreateEventRequest {
            deployment_id,
            tensorzero_version: tensorzero_core::endpoints::status::TENSORZERO_VERSION.to_string(),
            payload: request.payload,
            previous_user_message_event_id: request.previous_user_message_event_id,
            config_snapshot_hash,
        };

        create_event(autopilot_client, session_id, full_request)
            .await
            .map_err(|e| {
                TensorZeroClientError::TensorZero(TensorZeroError::Other { source: e.into() })
            })
    }

    async fn list_autopilot_events(
        &self,
        session_id: Uuid,
        params: ListEventsParams,
    ) -> Result<GatewayListEventsResponse, TensorZeroClientError> {
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
    ) -> Result<ActionResponse, TensorZeroClientError> {
        let action_input = ActionInputInfo {
            snapshot_hash,
            input,
        };

        crate::action::action(&self.app_state, action_input)
            .await
            .map_err(|e| {
                TensorZeroClientError::TensorZero(TensorZeroError::Other { source: e.into() })
            })
    }

    async fn get_config_snapshot(
        &self,
        hash: Option<String>,
    ) -> Result<GetConfigResponse, TensorZeroClientError> {
        let snapshot_hash = match hash {
            Some(hash) => hash.parse().map_err(|_| {
                TensorZeroClientError::TensorZero(TensorZeroError::Other {
                    source: Error::new(ErrorDetails::ConfigSnapshotNotFound {
                        snapshot_hash: hash,
                    })
                    .into(),
                })
            })?,
            None => self.app_state.config.hash.clone(),
        };

        let snapshot = self
            .app_state
            .clickhouse_connection_info
            .get_config_snapshot(snapshot_hash)
            .await
            .map_err(|e| {
                TensorZeroClientError::TensorZero(TensorZeroError::Other { source: e.into() })
            })?;

        Ok(GetConfigResponse {
            hash: snapshot.hash.to_string(),
            config: snapshot.config.try_into().map_err(|e: &'static str| {
                TensorZeroClientError::TensorZero(TensorZeroError::Other {
                    source: tensorzero_core::error::Error::new(
                        tensorzero_core::error::ErrorDetails::Config {
                            message: e.to_string(),
                        },
                    )
                    .into(),
                })
            })?,
            extra_templates: snapshot.extra_templates,
            tags: snapshot.tags,
        })
    }

    async fn write_config(
        &self,
        request: WriteConfigRequest,
    ) -> Result<WriteConfigResponse, TensorZeroClientError> {
        let mut snapshot =
            ConfigSnapshot::new(request.config, request.extra_templates).map_err(|e| {
                TensorZeroClientError::TensorZero(TensorZeroError::Other { source: e.into() })
            })?;
        snapshot.tags = request.tags;

        let hash = snapshot.hash.to_string();

        write_config_snapshot(&self.app_state.clickhouse_connection_info, snapshot)
            .await
            .map_err(|e| {
                TensorZeroClientError::TensorZero(TensorZeroError::Other { source: e.into() })
            })?;

        Ok(WriteConfigResponse { hash })
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

    async fn list_datasets(
        &self,
        request: ListDatasetsRequest,
    ) -> Result<ListDatasetsResponse, TensorZeroClientError> {
        tensorzero_core::endpoints::datasets::v1::list_datasets(
            &self.app_state.clickhouse_connection_info,
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

    // ========== Inference Query Operations ==========

    async fn list_inferences(
        &self,
        request: ListInferencesRequest,
    ) -> Result<GetInferencesResponse, TensorZeroClientError> {
        tensorzero_core::endpoints::stored_inferences::v1::list_inferences(
            &self.app_state.config,
            &self.app_state.clickhouse_connection_info,
            request,
        )
        .await
        .map_err(|e| TensorZeroClientError::TensorZero(TensorZeroError::Other { source: e.into() }))
    }

    async fn get_inferences(
        &self,
        request: GetInferencesRequest,
    ) -> Result<GetInferencesResponse, TensorZeroClientError> {
        tensorzero_core::endpoints::stored_inferences::v1::get_inferences(
            &self.app_state.config,
            &self.app_state.clickhouse_connection_info,
            request,
        )
        .await
        .map_err(|e| TensorZeroClientError::TensorZero(TensorZeroError::Other { source: e.into() }))
    }

    // ========== Optimization Operations ==========

    async fn launch_optimization_workflow(
        &self,
        params: tensorzero_optimizers::endpoints::LaunchOptimizationWorkflowParams,
    ) -> Result<super::OptimizationJobHandle, TensorZeroClientError> {
        tensorzero_optimizers::endpoints::launch_optimization_workflow(
            &self.app_state.http_client,
            self.app_state.config.clone(),
            &self.app_state.clickhouse_connection_info,
            params,
        )
        .await
        .map_err(|e| TensorZeroClientError::TensorZero(TensorZeroError::Other { source: e.into() }))
    }

    async fn poll_optimization(
        &self,
        job_handle: &super::OptimizationJobHandle,
    ) -> Result<super::OptimizationJobInfo, TensorZeroClientError> {
        tensorzero_optimizers::endpoints::poll_optimization(
            &self.app_state.http_client,
            job_handle,
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

    async fn get_feedback_by_variant(
        &self,
        metric_name: String,
        function_name: String,
        variant_names: Option<Vec<String>>,
    ) -> Result<Vec<FeedbackByVariant>, TensorZeroClientError> {
        self.app_state
            .clickhouse_connection_info
            .get_feedback_by_variant(&metric_name, &function_name, variant_names.as_ref())
            .await
            .map_err(|e| {
                TensorZeroClientError::TensorZero(TensorZeroError::Other { source: e.into() })
            })
    }

    async fn run_evaluation(
        &self,
        params: RunEvaluationParams,
    ) -> Result<RunEvaluationResponse, TensorZeroClientError> {
        crate::run_evaluation::run_evaluation(self.app_state.clone(), &params)
            .await
            .map_err(|e| TensorZeroClientError::Evaluation(e.to_string()))
    }
}
