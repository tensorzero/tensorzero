//! Implementation of `TensorZeroClient` for the TensorZero SDK `Client`.
//!
//! This extends the `tensorzero::Client` type with the `TensorZeroClient` trait,
//! handling both HTTP gateway and embedded gateway modes via the client's internal
//! mode switching.

use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;
use autopilot_client::AutopilotError;
use evaluations::stats::EvaluationStats;
use evaluations::types::{EvaluationCoreArgs, EvaluationVariant};
use evaluations::{EvaluationUpdate, OutputFormat, run_evaluation_core_streaming};
use tensorzero::{
    Client, ClientBuilder, ClientBuilderMode, ClientExt, ClientInferenceParams, ClientMode,
    CreateDatapointRequest, CreateDatapointsFromInferenceRequestParams, CreateDatapointsResponse,
    DeleteDatapointsResponse, FeedbackParams, FeedbackResponse, GetDatapointsResponse,
    InferenceOutput, InferenceResponse, ListDatapointsRequest, TensorZeroError,
    UpdateDatapointRequest, UpdateDatapointsResponse,
};
use tensorzero_core::config::snapshot::SnapshotHash;
use tensorzero_core::endpoints::feedback::internal::{
    LatestFeedbackIdByMetricResponse, get_latest_feedback_id_by_metric,
};
use tensorzero_core::endpoints::internal::action::{ActionInput, ActionInputInfo};
use tensorzero_core::endpoints::internal::autopilot::list_sessions;
use tensorzero_core::evaluations::{EvaluationConfig, EvaluationFunctionConfig};
use uuid::Uuid;

use super::{
    CreateEventRequest, CreateEventResponse, EvaluatorStatsResponse, ListEventsParams,
    ListEventsResponse, ListSessionsParams, ListSessionsResponse, RunEvaluationParams,
    RunEvaluationResponse, TensorZeroClient, TensorZeroClientError,
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
        request: CreateEventRequest,
    ) -> Result<CreateEventResponse, TensorZeroClientError> {
        match self.mode() {
            ClientMode::HTTPGateway(http) => {
                // HTTP mode: call the internal endpoint
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

                tensorzero_core::endpoints::internal::autopilot::create_event(
                    autopilot_client,
                    session_id,
                    request,
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
    ) -> Result<ListEventsResponse, TensorZeroClientError> {
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
    ) -> Result<InferenceResponse, TensorZeroClientError> {
        match self.mode() {
            ClientMode::HTTPGateway(http) => {
                let url = http
                    .base_url
                    .join("internal/action")
                    .map_err(|e: url::ParseError| {
                        TensorZeroClientError::Autopilot(AutopilotError::InvalidUrl(e))
                    })?;

                let response = http
                    .http_client
                    .post(url)
                    .json(&input)
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
                let action_input = ActionInputInfo {
                    snapshot_hash,
                    input,
                };

                let result = tensorzero_core::endpoints::internal::action::action(
                    &gateway.handle.app_state,
                    action_input,
                )
                .await
                .map_err(|e| {
                    TensorZeroClientError::TensorZero(TensorZeroError::Other { source: e.into() })
                })?;

                match result {
                    tensorzero_core::endpoints::internal::action::ActionResponse::Inference(
                        response,
                    ) => Ok(response),
                    tensorzero_core::endpoints::internal::action::ActionResponse::Feedback(_) => {
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
        }
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

    async fn run_evaluation(
        &self,
        params: RunEvaluationParams,
    ) -> Result<RunEvaluationResponse, TensorZeroClientError> {
        match self.mode() {
            ClientMode::HTTPGateway(_) => Err(TensorZeroClientError::NotSupported(
                "run_evaluation is only supported in embedded gateway mode".to_string(),
            )),
            ClientMode::EmbeddedGateway { gateway, timeout } => {
                let app_state = &gateway.handle.app_state;

                // Look up the evaluation config
                let evaluation_config = app_state
                    .config
                    .evaluations
                    .get(&params.evaluation_name)
                    .ok_or_else(|| {
                        TensorZeroClientError::Evaluation(format!(
                            "Evaluation '{}' not found in config",
                            params.evaluation_name
                        ))
                    })?
                    .clone();

                // Build function configs table for the evaluation
                let function_configs: HashMap<String, EvaluationFunctionConfig> = app_state
                    .config
                    .functions
                    .iter()
                    .map(|(name, func)| {
                        (name.clone(), EvaluationFunctionConfig::from(func.as_ref()))
                    })
                    .collect();
                let function_configs = Arc::new(function_configs);

                // Build a Client from our existing components
                let tensorzero_client = ClientBuilder::new(ClientBuilderMode::FromComponents {
                    config: app_state.config.clone(),
                    clickhouse_connection_info: app_state.clickhouse_connection_info.clone(),
                    postgres_connection_info: app_state.postgres_connection_info.clone(),
                    http_client: app_state.http_client.clone(),
                    timeout: *timeout,
                })
                .build()
                .await
                .map_err(|e| {
                    TensorZeroClientError::Evaluation(format!("Failed to build client: {e}"))
                })?;

                let evaluation_run_id = Uuid::now_v7();

                let core_args = EvaluationCoreArgs {
                    tensorzero_client,
                    clickhouse_client: app_state.clickhouse_connection_info.clone(),
                    evaluation_config,
                    function_configs,
                    dataset_name: params.dataset_name,
                    datapoint_ids: params.datapoint_ids,
                    variant: EvaluationVariant::Name(params.variant_name),
                    evaluation_name: params.evaluation_name,
                    evaluation_run_id,
                    inference_cache: params.inference_cache,
                    concurrency: params.concurrency,
                };

                // Run the evaluation with optional adaptive stopping via precision_targets
                let result = run_evaluation_core_streaming(
                    core_args,
                    params.max_datapoints,
                    params.precision_targets,
                )
                .await
                .map_err(|e| {
                    TensorZeroClientError::Evaluation(format!("Evaluation failed: {e}"))
                })?;

                let mut receiver = result.receiver;
                let num_datapoints = result.run_info.num_datapoints;

                // Collect evaluation results from the channel.
                // We skip RunInfo since we already have that data from result.run_info.
                // Success and Error updates are accumulated in evaluation_stats for
                // computing final statistics. The dummy_writer discards serialized output
                // since we only need the in-memory statistics, not CLI output.
                let mut evaluation_stats =
                    EvaluationStats::new(OutputFormat::Jsonl, num_datapoints);
                let mut dummy_writer = std::io::sink();

                while let Some(update) = receiver.recv().await {
                    match update {
                        EvaluationUpdate::RunInfo(_) => continue,
                        update => {
                            let _ = evaluation_stats.push(update, &mut dummy_writer);
                        }
                    }
                }

                // Compute statistics
                let EvaluationConfig::Inference(inference_config) = &*result.evaluation_config;
                let stats = evaluation_stats.compute_stats(&inference_config.evaluators);

                // Convert to response format
                let stats_response: HashMap<String, EvaluatorStatsResponse> = stats
                    .into_iter()
                    .map(|(name, s)| {
                        (
                            name,
                            EvaluatorStatsResponse {
                                mean: s.mean,
                                stderr: s.stderr,
                                count: s.count,
                            },
                        )
                    })
                    .collect();

                Ok(RunEvaluationResponse {
                    evaluation_run_id,
                    num_datapoints,
                    num_successes: evaluation_stats.evaluation_infos.len(),
                    num_errors: evaluation_stats.evaluation_errors.len(),
                    stats: stats_response,
                })
            }
        }
    }
}
