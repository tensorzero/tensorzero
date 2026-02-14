//! Implementation of `TensorZeroClient` for the TensorZero SDK `Client`.
//!
//! This extends the `tensorzero::Client` type with the `TensorZeroClient` trait,
//! handling both HTTP gateway and embedded gateway modes via the client's internal
//! mode switching.

use async_trait::async_trait;
use autopilot_client::AutopilotError;
use autopilot_client::GatewayListEventsResponse;
use tensorzero::{
    Client, ClientExt, ClientInferenceParams, ClientMode, CreateDatapointRequest,
    CreateDatapointsFromInferenceRequestParams, CreateDatapointsResponse, DeleteDatapointsResponse,
    FeedbackParams, FeedbackResponse, GetConfigResponse, GetDatapointsResponse,
    GetInferencesRequest, GetInferencesResponse, InferenceOutput, InferenceResponse,
    ListDatapointsRequest, ListDatasetsRequest, ListDatasetsResponse, ListInferencesRequest,
    TensorZeroError, UpdateDatapointRequest, UpdateDatapointsResponse, WriteConfigRequest,
    WriteConfigResponse,
};
use tensorzero_core::config::snapshot::SnapshotHash;
use tensorzero_core::db::feedback::FeedbackByVariant;
use tensorzero_core::db::feedback::FeedbackQueries;
use tensorzero_core::endpoints::feedback::internal::{
    LatestFeedbackIdByMetricResponse, get_latest_feedback_id_by_metric,
};
use tensorzero_core::endpoints::internal::autopilot::list_sessions;
use tensorzero_core::optimization::{OptimizationJobHandle, OptimizationJobInfo};
use tensorzero_optimizers::endpoints::{
    LaunchOptimizationWorkflowParams, launch_optimization_workflow, poll_optimization,
};
use uuid::Uuid;

use crate::action::{ActionInput, ActionInputInfo, ActionResponse};

use super::{
    CreateEventGatewayRequest, CreateEventResponse, ListEventsParams, ListSessionsParams,
    ListSessionsResponse, RunEvaluationParams, RunEvaluationResponse, TensorZeroClient,
    TensorZeroClientError,
};

/// Implementation of `TensorZeroClient` for the TensorZero SDK `Client`.
#[async_trait]
impl TensorZeroClient for Client {
    async fn inference(
        &self,
        params: ClientInferenceParams,
    ) -> Result<InferenceResponse, TensorZeroClientError> {
        match Client::inference(self, params).await? {
            InferenceOutput::NonStreaming(response) => Ok(response),
            InferenceOutput::Streaming(_) => Err(TensorZeroClientError::StreamingNotSupported),
        }
    }

    async fn feedback(
        &self,
        params: FeedbackParams,
    ) -> Result<FeedbackResponse, TensorZeroClientError> {
        Client::feedback(self, params)
            .await
            .map_err(TensorZeroClientError::TensorZero)
    }

    async fn create_autopilot_event(
        &self,
        session_id: Uuid,
        request: CreateEventGatewayRequest,
    ) -> Result<CreateEventResponse, TensorZeroClientError> {
        match self.mode() {
            ClientMode::HTTPGateway(http) => {
                // HTTP mode: call the internal endpoint
                // The gateway will inject deployment_id from its app state
                let url = http
                    .base_url
                    .join(&format!(
                        "internal/autopilot/v1/sessions/{session_id}/events"
                    ))
                    .map_err(|e: url::ParseError| {
                        TensorZeroClientError::Autopilot(AutopilotError::InvalidUrl(e))
                    })?;

                let response = http
                    .http_client
                    .post(url)
                    .json(&request)
                    .send()
                    .await
                    .map_err(|e| TensorZeroClientError::Autopilot(AutopilotError::Request(e)))?;

                if !response.status().is_success() {
                    let status = response.status().as_u16();
                    let text = response.text().await.unwrap_or_default();
                    return Err(TensorZeroClientError::Autopilot(AutopilotError::Http {
                        status_code: status,
                        message: text,
                    }));
                }

                response
                    .json()
                    .await
                    .map_err(|e| TensorZeroClientError::Autopilot(AutopilotError::Request(e)))
            }
            ClientMode::EmbeddedGateway {
                gateway,
                timeout: _,
            } => {
                // Embedded mode: call the core function directly
                let autopilot_client = gateway
                    .handle
                    .app_state
                    .autopilot_client
                    .as_ref()
                    .ok_or(TensorZeroClientError::AutopilotUnavailable)?;

                // Get deployment_id from app_state
                let deployment_id = gateway
                    .handle
                    .app_state
                    .deployment_id
                    .clone()
                    .ok_or(TensorZeroClientError::AutopilotUnavailable)?;

                // Construct the full request with deployment_id from app state
                // If starting a new session (nil session_id), include the current config hash
                let config_snapshot_hash = if session_id.is_nil() {
                    Some(gateway.handle.app_state.config.hash.to_string())
                } else {
                    None
                };
                let full_request = autopilot_client::CreateEventRequest {
                    deployment_id,
                    tensorzero_version: tensorzero_core::endpoints::status::TENSORZERO_VERSION
                        .to_string(),
                    payload: request.payload,
                    previous_user_message_event_id: request.previous_user_message_event_id,
                    config_snapshot_hash,
                };

                tensorzero_core::endpoints::internal::autopilot::create_event(
                    autopilot_client,
                    session_id,
                    full_request,
                )
                .await
                .map_err(|e| {
                    TensorZeroClientError::TensorZero(TensorZeroError::Other { source: e.into() })
                })
            }
        }
    }

    async fn list_autopilot_events(
        &self,
        session_id: Uuid,
        params: ListEventsParams,
    ) -> Result<GatewayListEventsResponse, TensorZeroClientError> {
        match self.mode() {
            ClientMode::HTTPGateway(http) => {
                let mut url = http
                    .base_url
                    .join(&format!(
                        "internal/autopilot/v1/sessions/{session_id}/events"
                    ))
                    .map_err(|e: url::ParseError| {
                        TensorZeroClientError::Autopilot(AutopilotError::InvalidUrl(e))
                    })?;

                // Add query params
                if let Some(limit) = params.limit {
                    url.query_pairs_mut()
                        .append_pair("limit", &limit.to_string());
                }
                if let Some(before) = params.before {
                    url.query_pairs_mut()
                        .append_pair("before", &before.to_string());
                }

                let response =
                    http.http_client.get(url).send().await.map_err(|e| {
                        TensorZeroClientError::Autopilot(AutopilotError::Request(e))
                    })?;

                if !response.status().is_success() {
                    let status = response.status().as_u16();
                    let text = response.text().await.unwrap_or_default();
                    return Err(TensorZeroClientError::Autopilot(AutopilotError::Http {
                        status_code: status,
                        message: text,
                    }));
                }

                response
                    .json()
                    .await
                    .map_err(|e| TensorZeroClientError::Autopilot(AutopilotError::Request(e)))
            }
            ClientMode::EmbeddedGateway {
                gateway,
                timeout: _,
            } => {
                let autopilot_client = gateway
                    .handle
                    .app_state
                    .autopilot_client
                    .as_ref()
                    .ok_or(TensorZeroClientError::AutopilotUnavailable)?;

                tensorzero_core::endpoints::internal::autopilot::list_events(
                    autopilot_client,
                    session_id,
                    params,
                )
                .await
                .map_err(|e| {
                    TensorZeroClientError::TensorZero(TensorZeroError::Other { source: e.into() })
                })
            }
        }
    }

    async fn list_autopilot_sessions(
        &self,
        params: ListSessionsParams,
    ) -> Result<ListSessionsResponse, TensorZeroClientError> {
        match self.mode() {
            ClientMode::HTTPGateway(http) => {
                let mut url = http
                    .base_url
                    .join("internal/autopilot/v1/sessions")
                    .map_err(|e: url::ParseError| {
                        TensorZeroClientError::Autopilot(AutopilotError::InvalidUrl(e))
                    })?;

                // Add query params
                if let Some(limit) = params.limit {
                    url.query_pairs_mut()
                        .append_pair("limit", &limit.to_string());
                }
                if let Some(offset) = params.offset {
                    url.query_pairs_mut()
                        .append_pair("offset", &offset.to_string());
                }

                let response =
                    http.http_client.get(url).send().await.map_err(|e| {
                        TensorZeroClientError::Autopilot(AutopilotError::Request(e))
                    })?;

                if !response.status().is_success() {
                    let status = response.status().as_u16();
                    let text = response.text().await.unwrap_or_default();
                    return Err(TensorZeroClientError::Autopilot(AutopilotError::Http {
                        status_code: status,
                        message: text,
                    }));
                }

                response
                    .json()
                    .await
                    .map_err(|e| TensorZeroClientError::Autopilot(AutopilotError::Request(e)))
            }
            ClientMode::EmbeddedGateway {
                gateway,
                timeout: _,
            } => {
                let autopilot_client = gateway
                    .handle
                    .app_state
                    .autopilot_client
                    .as_ref()
                    .ok_or(TensorZeroClientError::AutopilotUnavailable)?;

                list_sessions(autopilot_client, params).await.map_err(|e| {
                    TensorZeroClientError::TensorZero(TensorZeroError::Other { source: e.into() })
                })
            }
        }
    }

    async fn action(
        &self,
        snapshot_hash: SnapshotHash,
        input: ActionInput,
    ) -> Result<ActionResponse, TensorZeroClientError> {
        match self.mode() {
            ClientMode::HTTPGateway(http) => {
                let url = http
                    .base_url
                    .join("internal/action")
                    .map_err(|e: url::ParseError| {
                        TensorZeroClientError::Autopilot(AutopilotError::InvalidUrl(e))
                    })?;

                let action_input = ActionInputInfo {
                    snapshot_hash,
                    input,
                };

                let response = http
                    .http_client
                    .post(url)
                    .json(&action_input)
                    .send()
                    .await
                    .map_err(|e| TensorZeroClientError::Autopilot(AutopilotError::Request(e)))?;

                if !response.status().is_success() {
                    let status = response.status().as_u16();
                    let text = response.text().await.unwrap_or_default();
                    return Err(TensorZeroClientError::Autopilot(AutopilotError::Http {
                        status_code: status,
                        message: text,
                    }));
                }

                response
                    .json()
                    .await
                    .map_err(|e| TensorZeroClientError::Autopilot(AutopilotError::Request(e)))
            }
            ClientMode::EmbeddedGateway { .. } => {
                // Action endpoint is only available via HTTP gateway.
                // In embedded mode, callers should use the specific methods directly
                // (e.g., inference(), feedback(), run_evaluation()).
                Err(TensorZeroClientError::NotSupported(
                    "action is only supported in HTTP gateway mode".to_string(),
                ))
            }
        }
    }

    async fn get_config_snapshot(
        &self,
        hash: Option<String>,
    ) -> Result<GetConfigResponse, TensorZeroClientError> {
        ClientExt::get_config_snapshot(self, hash.as_deref())
            .await
            .map_err(TensorZeroClientError::TensorZero)
    }

    async fn write_config(
        &self,
        request: WriteConfigRequest,
    ) -> Result<WriteConfigResponse, TensorZeroClientError> {
        ClientExt::write_config(self, request)
            .await
            .map_err(TensorZeroClientError::TensorZero)
    }

    // ========== Datapoint CRUD Operations ==========

    async fn create_datapoints(
        &self,
        dataset_name: String,
        datapoints: Vec<CreateDatapointRequest>,
    ) -> Result<CreateDatapointsResponse, TensorZeroClientError> {
        ClientExt::create_datapoints(self, dataset_name, datapoints)
            .await
            .map_err(TensorZeroClientError::TensorZero)
    }

    async fn create_datapoints_from_inferences(
        &self,
        dataset_name: String,
        params: CreateDatapointsFromInferenceRequestParams,
    ) -> Result<CreateDatapointsResponse, TensorZeroClientError> {
        ClientExt::create_datapoints_from_inferences(self, dataset_name, params)
            .await
            .map_err(TensorZeroClientError::TensorZero)
    }

    async fn list_datasets(
        &self,
        request: ListDatasetsRequest,
    ) -> Result<ListDatasetsResponse, TensorZeroClientError> {
        ClientExt::list_datasets(self, request)
            .await
            .map_err(TensorZeroClientError::TensorZero)
    }

    async fn list_datapoints(
        &self,
        dataset_name: String,
        request: ListDatapointsRequest,
    ) -> Result<GetDatapointsResponse, TensorZeroClientError> {
        ClientExt::list_datapoints(self, dataset_name, request)
            .await
            .map_err(TensorZeroClientError::TensorZero)
    }

    async fn get_datapoints(
        &self,
        dataset_name: Option<String>,
        ids: Vec<Uuid>,
    ) -> Result<GetDatapointsResponse, TensorZeroClientError> {
        ClientExt::get_datapoints(self, dataset_name, ids)
            .await
            .map_err(TensorZeroClientError::TensorZero)
    }

    async fn update_datapoints(
        &self,
        dataset_name: String,
        datapoints: Vec<UpdateDatapointRequest>,
    ) -> Result<UpdateDatapointsResponse, TensorZeroClientError> {
        ClientExt::update_datapoints(self, dataset_name, datapoints)
            .await
            .map_err(TensorZeroClientError::TensorZero)
    }

    async fn delete_datapoints(
        &self,
        dataset_name: String,
        ids: Vec<Uuid>,
    ) -> Result<DeleteDatapointsResponse, TensorZeroClientError> {
        ClientExt::delete_datapoints(self, dataset_name, ids)
            .await
            .map_err(TensorZeroClientError::TensorZero)
    }

    // ========== Inference Query Operations ==========

    async fn list_inferences(
        &self,
        request: ListInferencesRequest,
    ) -> Result<GetInferencesResponse, TensorZeroClientError> {
        ClientExt::list_inferences(self, request)
            .await
            .map_err(TensorZeroClientError::TensorZero)
    }

    async fn get_inferences(
        &self,
        request: GetInferencesRequest,
    ) -> Result<GetInferencesResponse, TensorZeroClientError> {
        ClientExt::get_inferences(
            self,
            request.ids,
            request.function_name,
            request.output_source,
        )
        .await
        .map_err(TensorZeroClientError::TensorZero)
    }

    // ========== Optimization Operations ==========

    async fn launch_optimization_workflow(
        &self,
        params: LaunchOptimizationWorkflowParams,
    ) -> Result<OptimizationJobHandle, TensorZeroClientError> {
        match self.mode() {
            ClientMode::HTTPGateway(http) => {
                let url = http
                    .base_url
                    .join("experimental_optimization_workflow")
                    .map_err(|e: url::ParseError| {
                        TensorZeroClientError::Autopilot(AutopilotError::InvalidUrl(e))
                    })?;

                let builder = http.http_client.post(url).json(&params);
                let response_text = http.send_request(builder).await?;

                // The endpoint returns the base64-encoded job handle as plain text
                OptimizationJobHandle::from_base64_urlencoded(&response_text).map_err(|e| {
                    TensorZeroClientError::TensorZero(TensorZeroError::Other { source: e.into() })
                })
            }
            ClientMode::EmbeddedGateway {
                gateway,
                timeout: _,
            } => launch_optimization_workflow(
                &gateway.handle.app_state.http_client,
                gateway.handle.app_state.config.clone(),
                &gateway.handle.app_state.clickhouse_connection_info,
                params,
            )
            .await
            .map_err(|e| {
                TensorZeroClientError::TensorZero(TensorZeroError::Other { source: e.into() })
            }),
        }
    }

    async fn poll_optimization(
        &self,
        job_handle: &OptimizationJobHandle,
    ) -> Result<OptimizationJobInfo, TensorZeroClientError> {
        match self.mode() {
            ClientMode::HTTPGateway(http) => {
                let encoded_handle = job_handle.to_base64_urlencoded().map_err(|e| {
                    TensorZeroClientError::TensorZero(TensorZeroError::Other { source: e.into() })
                })?;

                let url = http
                    .base_url
                    .join(&format!("experimental_optimization/{encoded_handle}"))
                    .map_err(|e: url::ParseError| {
                        TensorZeroClientError::Autopilot(AutopilotError::InvalidUrl(e))
                    })?;

                let builder = http.http_client.get(url);
                let (response, _) = http.send_and_parse_http_response(builder).await?;
                Ok(response)
            }
            ClientMode::EmbeddedGateway {
                gateway,
                timeout: _,
            } => poll_optimization(
                &gateway.handle.app_state.http_client,
                job_handle,
                &gateway.handle.app_state.config.models.default_credentials,
                &gateway.handle.app_state.config.provider_types,
            )
            .await
            .map_err(|e| {
                TensorZeroClientError::TensorZero(TensorZeroError::Other { source: e.into() })
            }),
        }
    }

    async fn get_latest_feedback_id_by_metric(
        &self,
        target_id: Uuid,
    ) -> Result<LatestFeedbackIdByMetricResponse, TensorZeroClientError> {
        match self.mode() {
            ClientMode::HTTPGateway(http) => {
                let url = http
                    .base_url
                    .join(&format!(
                        "internal/feedback/{target_id}/latest_id_by_metric"
                    ))
                    .map_err(|e: url::ParseError| {
                        TensorZeroClientError::Autopilot(AutopilotError::InvalidUrl(e))
                    })?;

                let response =
                    http.http_client.get(url).send().await.map_err(|e| {
                        TensorZeroClientError::Autopilot(AutopilotError::Request(e))
                    })?;

                if !response.status().is_success() {
                    let status = response.status().as_u16();
                    let text = response.text().await.unwrap_or_default();
                    return Err(TensorZeroClientError::Autopilot(AutopilotError::Http {
                        status_code: status,
                        message: text,
                    }));
                }

                response
                    .json()
                    .await
                    .map_err(|e| TensorZeroClientError::Autopilot(AutopilotError::Request(e)))
            }
            ClientMode::EmbeddedGateway {
                gateway,
                timeout: _,
            } => get_latest_feedback_id_by_metric(
                &gateway.handle.app_state.clickhouse_connection_info,
                target_id,
            )
            .await
            .map_err(|e| {
                TensorZeroClientError::TensorZero(TensorZeroError::Other { source: e.into() })
            }),
        }
    }

    async fn get_feedback_by_variant(
        &self,
        metric_name: String,
        function_name: String,
        variant_names: Option<Vec<String>>,
    ) -> Result<Vec<FeedbackByVariant>, TensorZeroClientError> {
        match self.mode() {
            ClientMode::HTTPGateway(_) => Err(TensorZeroClientError::NotSupported(
                "get_feedback_by_variant is only available in embedded mode".to_string(),
            )),
            ClientMode::EmbeddedGateway {
                gateway,
                timeout: _,
            } => gateway
                .handle
                .app_state
                .clickhouse_connection_info
                .get_feedback_by_variant(&metric_name, &function_name, variant_names.as_ref())
                .await
                .map_err(|e| {
                    TensorZeroClientError::TensorZero(TensorZeroError::Other { source: e.into() })
                }),
        }
    }

    async fn run_evaluation(
        &self,
        params: RunEvaluationParams,
    ) -> Result<RunEvaluationResponse, TensorZeroClientError> {
        match self.mode() {
            ClientMode::HTTPGateway(_) => Err(TensorZeroClientError::NotSupported(
                "run_evaluation is only supported in embedded gateway mode".to_string(),
            )),
            ClientMode::EmbeddedGateway {
                gateway,
                timeout: _,
            } => crate::run_evaluation::run_evaluation(gateway.handle.app_state.clone(), &params)
                .await
                .map_err(|e| TensorZeroClientError::Evaluation(e.to_string())),
        }
    }
}
