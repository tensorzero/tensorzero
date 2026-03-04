//! A `TensorZeroClient` wrapper that checkpoints every method call as a durable step.
//!
//! This is useful when calling `TensorZeroClient` from within a durable task,
//! ensuring that each operation is cached and not re-executed on retry.

use async_trait::async_trait;
use tokio::sync::Mutex;
use uuid::Uuid;

use tensorzero_core::config::snapshot::SnapshotHash;
use tensorzero_core::db::feedback::FeedbackByVariant;
use tensorzero_core::endpoints::embeddings::{EmbeddingResponse, EmbeddingsParams};
use tensorzero_core::endpoints::feedback::internal::{
    GetFeedbackByTargetIdResponse, LatestFeedbackIdByMetricResponse,
};
use tensorzero_core::optimization::{OptimizationJobHandle, OptimizationJobInfo};
use tensorzero_optimizers::endpoints::LaunchOptimizationWorkflowParams;

use tensorzero::{
    ClientInferenceParams, CreateDatapointRequest, CreateDatapointsFromInferenceRequestParams,
    CreateDatapointsResponse, DeleteDatapointsResponse, FeedbackParams, FeedbackResponse,
    GetConfigResponse, GetDatapointsResponse, GetInferencesRequest, GetInferencesResponse,
    InferenceResponse, ListDatapointsRequest, ListDatasetsRequest, ListDatasetsResponse,
    ListEpisodesRequest, ListEpisodesResponse, ListInferencesRequest, UpdateDatapointRequest,
    UpdateDatapointsResponse, WriteConfigRequest, WriteConfigResponse,
};

use crate::action::{ActionInput, ActionResponse};
use crate::context::ToolContext;
use crate::run_evaluation::{RunEvaluationParams, RunEvaluationResponse};

use super::{
    CreateEventGatewayRequest, CreateEventResponse, GatewayListEventsResponse, ListEventsParams,
    ListSessionsParams, ListSessionsResponse, S3UploadRequest, S3UploadResponse, TensorZeroClient,
    TensorZeroClientError,
};

/// A `TensorZeroClient` that wraps each method call in a durable checkpoint step.
///
/// This ensures that operations are not re-executed on task retry, providing
/// exactly-once semantics for all client calls within a durable task.
///
/// # Usage
///
/// ```ignore
/// let checkpointed = CheckpointedTensorzeroClient::new(ctx);
/// // Each call is now checkpointed:
/// let response = checkpointed.inference(params).await?;
/// ```
pub struct CheckpointedTensorzeroClient<S: Clone + Send + Sync + 'static = ()> {
    ctx: Mutex<ToolContext<S>>,
}

impl<S: Clone + Send + Sync + 'static> CheckpointedTensorzeroClient<S> {
    /// Create a new checkpointed client wrapping the given tool context.
    pub fn new(ctx: ToolContext<S>) -> Self {
        Self {
            ctx: Mutex::new(ctx),
        }
    }

    /// Consume this wrapper and return the inner `ToolContext`.
    pub fn into_inner(self) -> ToolContext<S> {
        self.ctx.into_inner()
    }
}

/// Convert a `ToolError` into a `TensorZeroClientError`.
///
/// Control flow errors (suspend/cancel) are preserved as `ControlFlow` variants
/// so callers can convert them back to `ToolError::Control` for the durable runtime.
fn tool_error_to_client_error(err: crate::error::ToolError) -> TensorZeroClientError {
    match err {
        crate::error::ToolError::Control(cf) => TensorZeroClientError::ControlFlow(cf),
        other => TensorZeroClientError::NotSupported(format!("durable step error: {other}")),
    }
}

#[async_trait]
impl<S: Clone + Send + Sync + 'static> TensorZeroClient for CheckpointedTensorzeroClient<S> {
    async fn inference(
        &self,
        params: ClientInferenceParams,
    ) -> Result<InferenceResponse, TensorZeroClientError> {
        self.ctx
            .lock()
            .await
            .step("inference", params, |params, state| async move {
                state
                    .t0_client()
                    .inference(params)
                    .await
                    .map_err(|e| anyhow::anyhow!("{e}"))
            })
            .await
            .map_err(tool_error_to_client_error)
    }

    async fn embeddings(
        &self,
        _params: EmbeddingsParams,
    ) -> Result<EmbeddingResponse, TensorZeroClientError> {
        Err(TensorZeroClientError::NotSupported(
            "embeddings are not supported in checkpointed client".to_string(),
        ))
    }

    async fn feedback(
        &self,
        params: FeedbackParams,
    ) -> Result<FeedbackResponse, TensorZeroClientError> {
        self.ctx
            .lock()
            .await
            .step("feedback", params, |params, state| async move {
                state
                    .t0_client()
                    .feedback(params)
                    .await
                    .map_err(|e| anyhow::anyhow!("{e}"))
            })
            .await
            .map_err(tool_error_to_client_error)
    }

    async fn get_latest_feedback_id_by_metric(
        &self,
        target_id: Uuid,
    ) -> Result<LatestFeedbackIdByMetricResponse, TensorZeroClientError> {
        self.ctx
            .lock()
            .await
            .step(
                "get_latest_feedback_id_by_metric",
                target_id,
                |target_id, state| async move {
                    state
                        .t0_client()
                        .get_latest_feedback_id_by_metric(target_id)
                        .await
                        .map_err(|e| anyhow::anyhow!("{e}"))
                },
            )
            .await
            .map_err(tool_error_to_client_error)
    }

    async fn get_feedback_by_target_id(
        &self,
        target_id: Uuid,
        before: Option<Uuid>,
        after: Option<Uuid>,
        limit: Option<u32>,
    ) -> Result<GetFeedbackByTargetIdResponse, TensorZeroClientError> {
        self.ctx
            .lock()
            .await
            .step(
                "get_feedback_by_target_id",
                (target_id, before, after, limit),
                |(target_id, before, after, limit), state| async move {
                    state
                        .t0_client()
                        .get_feedback_by_target_id(target_id, before, after, limit)
                        .await
                        .map_err(|e| anyhow::anyhow!("{e}"))
                },
            )
            .await
            .map_err(tool_error_to_client_error)
    }

    async fn get_feedback_by_variant(
        &self,
        metric_name: String,
        function_name: String,
        variant_names: Option<Vec<String>>,
    ) -> Result<Vec<FeedbackByVariant>, TensorZeroClientError> {
        self.ctx
            .lock()
            .await
            .step(
                "get_feedback_by_variant",
                (metric_name, function_name, variant_names),
                |(metric_name, function_name, variant_names), state| async move {
                    state
                        .t0_client()
                        .get_feedback_by_variant(metric_name, function_name, variant_names)
                        .await
                        .map_err(|e| anyhow::anyhow!("{e}"))
                },
            )
            .await
            .map_err(tool_error_to_client_error)
    }

    async fn create_autopilot_event(
        &self,
        session_id: Uuid,
        request: CreateEventGatewayRequest,
    ) -> Result<CreateEventResponse, TensorZeroClientError> {
        self.ctx
            .lock()
            .await
            .step(
                "create_autopilot_event",
                (session_id, request),
                |(session_id, request), state| async move {
                    state
                        .t0_client()
                        .create_autopilot_event(session_id, request)
                        .await
                        .map_err(|e| anyhow::anyhow!("{e}"))
                },
            )
            .await
            .map_err(tool_error_to_client_error)
    }

    async fn list_autopilot_events(
        &self,
        session_id: Uuid,
        params: ListEventsParams,
    ) -> Result<GatewayListEventsResponse, TensorZeroClientError> {
        self.ctx
            .lock()
            .await
            .step(
                "list_autopilot_events",
                (session_id, params),
                |(session_id, params), state| async move {
                    state
                        .t0_client()
                        .list_autopilot_events(session_id, params)
                        .await
                        .map_err(|e| anyhow::anyhow!("{e}"))
                },
            )
            .await
            .map_err(tool_error_to_client_error)
    }

    async fn list_autopilot_sessions(
        &self,
        params: ListSessionsParams,
    ) -> Result<ListSessionsResponse, TensorZeroClientError> {
        self.ctx
            .lock()
            .await
            .step(
                "list_autopilot_sessions",
                params,
                |params, state| async move {
                    state
                        .t0_client()
                        .list_autopilot_sessions(params)
                        .await
                        .map_err(|e| anyhow::anyhow!("{e}"))
                },
            )
            .await
            .map_err(tool_error_to_client_error)
    }

    async fn s3_initiate_upload(
        &self,
        session_id: Uuid,
        request: S3UploadRequest,
    ) -> Result<S3UploadResponse, TensorZeroClientError> {
        self.ctx
            .lock()
            .await
            .step(
                "s3_initiate_upload",
                (session_id, request),
                |(session_id, request), state| async move {
                    state
                        .t0_client()
                        .s3_initiate_upload(session_id, request)
                        .await
                        .map_err(|e| anyhow::anyhow!("{e}"))
                },
            )
            .await
            .map_err(tool_error_to_client_error)
    }

    async fn action(
        &self,
        snapshot_hash: SnapshotHash,
        input: ActionInput,
    ) -> Result<ActionResponse, TensorZeroClientError> {
        self.ctx
            .lock()
            .await
            .step(
                "action",
                (snapshot_hash, input),
                |(snapshot_hash, input), state| async move {
                    state
                        .t0_client()
                        .action(snapshot_hash, input)
                        .await
                        .map_err(|e| anyhow::anyhow!("{e}"))
                },
            )
            .await
            .map_err(tool_error_to_client_error)
    }

    async fn get_config_snapshot(
        &self,
        hash: Option<String>,
    ) -> Result<GetConfigResponse, TensorZeroClientError> {
        self.ctx
            .lock()
            .await
            .step("get_config_snapshot", hash, |hash, state| async move {
                state
                    .t0_client()
                    .get_config_snapshot(hash)
                    .await
                    .map_err(|e| anyhow::anyhow!("{e}"))
            })
            .await
            .map_err(tool_error_to_client_error)
    }

    async fn write_config(
        &self,
        request: WriteConfigRequest,
    ) -> Result<WriteConfigResponse, TensorZeroClientError> {
        self.ctx
            .lock()
            .await
            .step("write_config", request, |request, state| async move {
                state
                    .t0_client()
                    .write_config(request)
                    .await
                    .map_err(|e| anyhow::anyhow!("{e}"))
            })
            .await
            .map_err(tool_error_to_client_error)
    }

    async fn create_datapoints(
        &self,
        dataset_name: String,
        datapoints: Vec<CreateDatapointRequest>,
    ) -> Result<CreateDatapointsResponse, TensorZeroClientError> {
        self.ctx
            .lock()
            .await
            .step(
                "create_datapoints",
                (dataset_name, datapoints),
                |(dataset_name, datapoints), state| async move {
                    state
                        .t0_client()
                        .create_datapoints(dataset_name, datapoints)
                        .await
                        .map_err(|e| anyhow::anyhow!("{e}"))
                },
            )
            .await
            .map_err(tool_error_to_client_error)
    }

    async fn create_datapoints_from_inferences(
        &self,
        dataset_name: String,
        params: CreateDatapointsFromInferenceRequestParams,
    ) -> Result<CreateDatapointsResponse, TensorZeroClientError> {
        self.ctx
            .lock()
            .await
            .step(
                "create_datapoints_from_inferences",
                (dataset_name, params),
                |(dataset_name, params), state| async move {
                    state
                        .t0_client()
                        .create_datapoints_from_inferences(dataset_name, params)
                        .await
                        .map_err(|e| anyhow::anyhow!("{e}"))
                },
            )
            .await
            .map_err(tool_error_to_client_error)
    }

    async fn list_datasets(
        &self,
        request: ListDatasetsRequest,
    ) -> Result<ListDatasetsResponse, TensorZeroClientError> {
        self.ctx
            .lock()
            .await
            .step("list_datasets", request, |request, state| async move {
                state
                    .t0_client()
                    .list_datasets(request)
                    .await
                    .map_err(|e| anyhow::anyhow!("{e}"))
            })
            .await
            .map_err(tool_error_to_client_error)
    }

    async fn list_datapoints(
        &self,
        dataset_name: String,
        request: ListDatapointsRequest,
    ) -> Result<GetDatapointsResponse, TensorZeroClientError> {
        self.ctx
            .lock()
            .await
            .step(
                "list_datapoints",
                (dataset_name, request),
                |(dataset_name, request), state| async move {
                    state
                        .t0_client()
                        .list_datapoints(dataset_name, request)
                        .await
                        .map_err(|e| anyhow::anyhow!("{e}"))
                },
            )
            .await
            .map_err(tool_error_to_client_error)
    }

    async fn get_datapoints(
        &self,
        dataset_name: Option<String>,
        ids: Vec<Uuid>,
    ) -> Result<GetDatapointsResponse, TensorZeroClientError> {
        self.ctx
            .lock()
            .await
            .step(
                "get_datapoints",
                (dataset_name, ids),
                |(dataset_name, ids), state| async move {
                    state
                        .t0_client()
                        .get_datapoints(dataset_name, ids)
                        .await
                        .map_err(|e| anyhow::anyhow!("{e}"))
                },
            )
            .await
            .map_err(tool_error_to_client_error)
    }

    async fn update_datapoints(
        &self,
        dataset_name: String,
        datapoints: Vec<UpdateDatapointRequest>,
    ) -> Result<UpdateDatapointsResponse, TensorZeroClientError> {
        self.ctx
            .lock()
            .await
            .step(
                "update_datapoints",
                (dataset_name, datapoints),
                |(dataset_name, datapoints), state| async move {
                    state
                        .t0_client()
                        .update_datapoints(dataset_name, datapoints)
                        .await
                        .map_err(|e| anyhow::anyhow!("{e}"))
                },
            )
            .await
            .map_err(tool_error_to_client_error)
    }

    async fn delete_datapoints(
        &self,
        dataset_name: String,
        ids: Vec<Uuid>,
    ) -> Result<DeleteDatapointsResponse, TensorZeroClientError> {
        self.ctx
            .lock()
            .await
            .step(
                "delete_datapoints",
                (dataset_name, ids),
                |(dataset_name, ids), state| async move {
                    state
                        .t0_client()
                        .delete_datapoints(dataset_name, ids)
                        .await
                        .map_err(|e| anyhow::anyhow!("{e}"))
                },
            )
            .await
            .map_err(tool_error_to_client_error)
    }

    async fn delete_dataset(
        &self,
        dataset_name: String,
    ) -> Result<DeleteDatapointsResponse, TensorZeroClientError> {
        self.ctx
            .lock()
            .await
            .step(
                "delete_dataset",
                dataset_name,
                |dataset_name, state| async move {
                    state
                        .t0_client()
                        .delete_dataset(dataset_name)
                        .await
                        .map_err(|e| anyhow::anyhow!("{e}"))
                },
            )
            .await
            .map_err(tool_error_to_client_error)
    }

    async fn list_inferences(
        &self,
        request: ListInferencesRequest,
    ) -> Result<GetInferencesResponse, TensorZeroClientError> {
        self.ctx
            .lock()
            .await
            .step("list_inferences", request, |request, state| async move {
                state
                    .t0_client()
                    .list_inferences(request)
                    .await
                    .map_err(|e| anyhow::anyhow!("{e}"))
            })
            .await
            .map_err(tool_error_to_client_error)
    }

    async fn get_inferences(
        &self,
        request: GetInferencesRequest,
    ) -> Result<GetInferencesResponse, TensorZeroClientError> {
        self.ctx
            .lock()
            .await
            .step("get_inferences", request, |request, state| async move {
                state
                    .t0_client()
                    .get_inferences(request)
                    .await
                    .map_err(|e| anyhow::anyhow!("{e}"))
            })
            .await
            .map_err(tool_error_to_client_error)
    }

    async fn list_episodes(
        &self,
        request: ListEpisodesRequest,
    ) -> Result<ListEpisodesResponse, TensorZeroClientError> {
        self.ctx
            .lock()
            .await
            .step("list_episodes", request, |request, state| async move {
                state
                    .t0_client()
                    .list_episodes(request)
                    .await
                    .map_err(|e| anyhow::anyhow!("{e}"))
            })
            .await
            .map_err(tool_error_to_client_error)
    }

    async fn launch_optimization_workflow(
        &self,
        params: LaunchOptimizationWorkflowParams,
    ) -> Result<OptimizationJobHandle, TensorZeroClientError> {
        self.ctx
            .lock()
            .await
            .step(
                "launch_optimization_workflow",
                params,
                |params, state| async move {
                    state
                        .t0_client()
                        .launch_optimization_workflow(params)
                        .await
                        .map_err(|e| anyhow::anyhow!("{e}"))
                },
            )
            .await
            .map_err(tool_error_to_client_error)
    }

    async fn poll_optimization(
        &self,
        job_handle: &OptimizationJobHandle,
    ) -> Result<OptimizationJobInfo, TensorZeroClientError> {
        self.ctx
            .lock()
            .await
            .step(
                "poll_optimization",
                job_handle.clone(),
                |job_handle, state| async move {
                    state
                        .t0_client()
                        .poll_optimization(&job_handle)
                        .await
                        .map_err(|e| anyhow::anyhow!("{e}"))
                },
            )
            .await
            .map_err(tool_error_to_client_error)
    }

    async fn run_evaluation(
        &self,
        params: RunEvaluationParams,
    ) -> Result<RunEvaluationResponse, TensorZeroClientError> {
        self.ctx
            .lock()
            .await
            .step("run_evaluation", params, |params, state| async move {
                state
                    .t0_client()
                    .run_evaluation(params)
                    .await
                    .map_err(|e| anyhow::anyhow!("{e}"))
            })
            .await
            .map_err(tool_error_to_client_error)
    }
}
