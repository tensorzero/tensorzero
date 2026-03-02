// Path stubs are only used by utoipa's derive macro, not called at runtime.
#![expect(dead_code, clippy::print_stdout, clippy::expect_used)]

use tensorzero_core::endpoints::{
    batch_inference, datasets, episodes, feedback, functions, inference, internal, object_storage,
    status, stored_inferences, ui, variant_probabilities, workflow_evaluation_run,
    workflow_evaluations,
};
use utoipa::OpenApi;

/// Error response body returned by the API.
#[derive(utoipa::ToSchema)]
struct ErrorBody {
    error: String,
}

// ============================================================================
// Inference
// ============================================================================

#[utoipa::path(
    post,
    path = "/inference",
    request_body = inline(inference::Params),
    responses(
        (status = 200, description = "Inference response (non-streaming)", body = inference::InferenceResponse),
        (status = 400, description = "Bad request", body = ErrorBody),
    ),
    tag = "Inference"
)]
fn inference() {}

// ============================================================================
// Batch Inference
// ============================================================================

#[utoipa::path(
    post,
    path = "/batch_inference",
    request_body = inline(batch_inference::StartBatchInferenceParams),
    responses(
        (status = 200, description = "Batch inference started", body = batch_inference::PrepareBatchInferenceOutput),
        (status = 400, description = "Bad request", body = ErrorBody),
    ),
    tag = "Batch Inference"
)]
fn start_batch_inference() {}

#[utoipa::path(
    get,
    path = "/batch_inference/{batch_id}",
    params(
        ("batch_id" = String, Path, description = "The batch inference ID"),
    ),
    responses(
        (status = 200, description = "Batch inference result", body = batch_inference::CompletedBatchInferenceResponse),
        (status = 400, description = "Bad request", body = ErrorBody),
    ),
    tag = "Batch Inference"
)]
fn poll_batch_inference() {}

#[utoipa::path(
    get,
    path = "/batch_inference/{batch_id}/inference/{inference_id}",
    params(
        ("batch_id" = String, Path, description = "The batch inference ID"),
        ("inference_id" = String, Path, description = "A specific inference ID within the batch"),
    ),
    responses(
        (status = 200, description = "Single batch inference result", body = batch_inference::CompletedBatchInferenceResponse),
        (status = 400, description = "Bad request", body = ErrorBody),
    ),
    tag = "Batch Inference"
)]
fn poll_batch_inference_single() {}

// ============================================================================
// Feedback
// ============================================================================

#[utoipa::path(
    post,
    path = "/feedback",
    request_body = inline(feedback::Params),
    responses(
        (status = 200, description = "Feedback recorded", body = feedback::FeedbackResponse),
        (status = 400, description = "Bad request", body = ErrorBody),
    ),
    tag = "Feedback"
)]
fn feedback() {}

// ============================================================================
// Stored Inferences (Observability)
// ============================================================================

#[utoipa::path(
    post,
    path = "/v1/inferences/list_inferences",
    request_body = inline(stored_inferences::v1::types::ListInferencesRequest),
    responses(
        (status = 200, description = "List of inferences", body = stored_inferences::v1::types::GetInferencesResponse),
        (status = 400, description = "Bad request", body = ErrorBody),
    ),
    tag = "Observability"
)]
fn list_inferences() {}

#[utoipa::path(
    post,
    path = "/v1/inferences/get_inferences",
    request_body = inline(stored_inferences::v1::types::GetInferencesRequest),
    responses(
        (status = 200, description = "Retrieved inferences", body = stored_inferences::v1::types::GetInferencesResponse),
        (status = 400, description = "Bad request", body = ErrorBody),
    ),
    tag = "Observability"
)]
fn get_inferences() {}

// ============================================================================
// Datasets
// ============================================================================

#[utoipa::path(
    post,
    path = "/v1/datasets/{dataset_name}/datapoints",
    params(
        ("dataset_name" = String, Path, description = "The dataset name"),
    ),
    request_body = inline(datasets::v1::types::CreateDatapointsRequest),
    responses(
        (status = 200, description = "Datapoints created", body = datasets::v1::types::CreateDatapointsResponse),
        (status = 400, description = "Bad request", body = ErrorBody),
    ),
    tag = "Datasets"
)]
fn create_datapoints() {}

#[utoipa::path(
    patch,
    path = "/v1/datasets/{dataset_name}/datapoints",
    params(
        ("dataset_name" = String, Path, description = "The dataset name"),
    ),
    request_body = inline(datasets::v1::types::UpdateDatapointsRequest),
    responses(
        (status = 200, description = "Datapoints updated", body = datasets::v1::types::UpdateDatapointsResponse),
        (status = 400, description = "Bad request", body = ErrorBody),
    ),
    tag = "Datasets"
)]
fn update_datapoints() {}

#[utoipa::path(
    delete,
    path = "/v1/datasets/{dataset_name}/datapoints",
    params(
        ("dataset_name" = String, Path, description = "The dataset name"),
    ),
    request_body = inline(datasets::v1::types::DeleteDatapointsRequest),
    responses(
        (status = 200, description = "Datapoints deleted", body = datasets::v1::types::DeleteDatapointsResponse),
        (status = 400, description = "Bad request", body = ErrorBody),
    ),
    tag = "Datasets"
)]
fn delete_datapoints() {}

#[utoipa::path(
    patch,
    path = "/v1/datasets/{dataset_name}/datapoints/metadata",
    params(
        ("dataset_name" = String, Path, description = "The dataset name"),
    ),
    request_body = inline(datasets::v1::types::UpdateDatapointsMetadataRequest),
    responses(
        (status = 200, description = "Metadata updated", body = datasets::v1::types::UpdateDatapointsResponse),
        (status = 400, description = "Bad request", body = ErrorBody),
    ),
    tag = "Datasets"
)]
fn update_datapoints_metadata() {}

#[utoipa::path(
    post,
    path = "/v1/datasets/{dataset_name}/from_inferences",
    params(
        ("dataset_name" = String, Path, description = "The dataset name"),
    ),
    request_body = inline(datasets::v1::types::CreateDatapointsFromInferenceRequest),
    responses(
        (status = 200, description = "Datapoints created from inferences", body = datasets::v1::types::CreateDatapointsResponse),
        (status = 400, description = "Bad request", body = ErrorBody),
    ),
    tag = "Datasets"
)]
fn create_from_inferences() {}

#[utoipa::path(
    post,
    path = "/v1/datasets/{dataset_name}/list_datapoints",
    params(
        ("dataset_name" = String, Path, description = "The dataset name"),
    ),
    request_body = inline(datasets::v1::types::ListDatapointsRequest),
    responses(
        (status = 200, description = "List of datapoints", body = datasets::v1::types::GetDatapointsResponse),
        (status = 400, description = "Bad request", body = ErrorBody),
    ),
    tag = "Datasets"
)]
fn list_datapoints() {}

#[utoipa::path(
    delete,
    path = "/v1/datasets/{dataset_name}",
    params(
        ("dataset_name" = String, Path, description = "The dataset name"),
    ),
    responses(
        (status = 200, description = "Dataset deleted", body = datasets::v1::types::DeleteDatapointsResponse),
        (status = 400, description = "Bad request", body = ErrorBody),
    ),
    tag = "Datasets"
)]
fn delete_dataset() {}

#[utoipa::path(
    post,
    path = "/v1/datasets/{dataset_name}/get_datapoints",
    params(
        ("dataset_name" = String, Path, description = "The dataset name"),
    ),
    request_body = inline(datasets::v1::types::GetDatapointsRequest),
    responses(
        (status = 200, description = "Retrieved datapoints", body = datasets::v1::types::GetDatapointsResponse),
        (status = 400, description = "Bad request", body = ErrorBody),
    ),
    tag = "Datasets"
)]
fn get_datapoints_by_dataset() {}

// ============================================================================
// Workflow Evaluation
// ============================================================================

#[utoipa::path(
    post,
    path = "/workflow_evaluation_run",
    request_body = inline(workflow_evaluation_run::WorkflowEvaluationRunParams),
    responses(
        (status = 200, description = "Workflow evaluation run created", body = workflow_evaluation_run::WorkflowEvaluationRunResponse),
        (status = 400, description = "Bad request", body = ErrorBody),
    ),
    tag = "Workflow Evaluation"
)]
fn create_workflow_evaluation_run() {}

#[utoipa::path(
    post,
    path = "/workflow_evaluation_run/{run_id}/episode",
    params(
        ("run_id" = String, Path, description = "The workflow evaluation run ID"),
    ),
    request_body = inline(workflow_evaluation_run::WorkflowEvaluationRunEpisodeParams),
    responses(
        (status = 200, description = "Episode created", body = workflow_evaluation_run::WorkflowEvaluationRunEpisodeResponse),
        (status = 400, description = "Bad request", body = ErrorBody),
    ),
    tag = "Workflow Evaluation"
)]
fn create_workflow_evaluation_run_episode() {}

// ============================================================================
// Status / Health
// ============================================================================

#[utoipa::path(
    get,
    path = "/status",
    responses(
        (status = 200, description = "Gateway status", body = status::StatusResponse),
    ),
    tag = "Status"
)]
fn gateway_status() {}

#[utoipa::path(
    get,
    path = "/health",
    responses(
        (status = 200, description = "Health check"),
    ),
    tag = "Status"
)]
fn health() {}

#[utoipa::path(
    get,
    path = "/metrics",
    responses(
        (status = 200, description = "Prometheus metrics", content_type = "text/plain"),
    ),
    tag = "Status"
)]
fn metrics() {}

// ============================================================================
// Internal: Variant Sampling Probabilities
// ============================================================================

#[utoipa::path(
    get,
    path = "/variant_sampling_probabilities",
    responses(
        (status = 200, description = "Variant sampling probabilities", body = variant_probabilities::GetVariantSamplingProbabilitiesResponse),
        (status = 400, description = "Bad request", body = ErrorBody),
    ),
    tag = "Internal"
)]
fn get_variant_sampling_probabilities() {}

#[utoipa::path(
    get,
    path = "/internal/functions/{function_name}/variant_sampling_probabilities",
    params(
        ("function_name" = String, Path, description = "The function name"),
    ),
    responses(
        (status = 200, description = "Variant sampling probabilities", body = variant_probabilities::GetVariantSamplingProbabilitiesResponse),
        (status = 400, description = "Bad request", body = ErrorBody),
    ),
    tag = "Internal"
)]
fn get_variant_sampling_probabilities_by_function() {}

// ============================================================================
// Internal: Function Metrics & Variant Performances
// ============================================================================

#[utoipa::path(
    get,
    path = "/internal/functions/{function_name}/metrics",
    params(
        ("function_name" = String, Path, description = "The function name"),
    ),
    responses(
        (status = 200, description = "Metrics with feedback statistics", body = functions::internal::MetricsWithFeedbackResponse),
        (status = 400, description = "Bad request", body = ErrorBody),
    ),
    tag = "Internal"
)]
fn get_function_metrics() {}

#[utoipa::path(
    get,
    path = "/internal/functions/{function_name}/variant_performances",
    params(
        ("function_name" = String, Path, description = "The function name"),
    ),
    responses(
        (status = 200, description = "Variant performance statistics", body = functions::internal::VariantPerformancesResponse),
        (status = 400, description = "Bad request", body = ErrorBody),
    ),
    tag = "Internal"
)]
fn get_variant_performances() {}

// ============================================================================
// Internal: Inference Count
// ============================================================================

#[utoipa::path(
    get,
    path = "/internal/functions/inference_counts",
    responses(
        (status = 200, description = "Functions with inference counts", body = internal::inference_count::ListFunctionsWithInferenceCountResponse),
        (status = 400, description = "Bad request", body = ErrorBody),
    ),
    tag = "Internal"
)]
fn list_functions_with_inference_count() {}

#[utoipa::path(
    get,
    path = "/internal/functions/{function_name}/inference_count",
    params(
        ("function_name" = String, Path, description = "The function name"),
    ),
    responses(
        (status = 200, description = "Inference count", body = internal::inference_count::InferenceCountResponse),
        (status = 400, description = "Bad request", body = ErrorBody),
    ),
    tag = "Internal"
)]
fn get_inference_count() {}

#[utoipa::path(
    get,
    path = "/internal/functions/{function_name}/inference_count/{metric_name}",
    params(
        ("function_name" = String, Path, description = "The function name"),
        ("metric_name" = String, Path, description = "The metric name"),
    ),
    responses(
        (status = 200, description = "Inference and feedback counts", body = internal::inference_count::InferenceWithFeedbackCountResponse),
        (status = 400, description = "Bad request", body = ErrorBody),
    ),
    tag = "Internal"
)]
fn get_inference_with_feedback_count() {}

#[utoipa::path(
    get,
    path = "/internal/functions/{function_name}/throughput_by_variant",
    params(
        ("function_name" = String, Path, description = "The function name"),
    ),
    responses(
        (status = 200, description = "Function throughput by variant", body = internal::inference_count::GetFunctionThroughputByVariantResponse),
        (status = 400, description = "Bad request", body = ErrorBody),
    ),
    tag = "Internal"
)]
fn get_function_throughput_by_variant() {}

// ============================================================================
// Internal: Feedback
// ============================================================================

#[utoipa::path(
    get,
    path = "/internal/feedback/{target_id}",
    params(
        ("target_id" = String, Path, description = "The target ID (inference or episode)"),
    ),
    responses(
        (status = 200, description = "Feedback for target", body = feedback::internal::GetFeedbackByTargetIdResponse),
        (status = 400, description = "Bad request", body = ErrorBody),
    ),
    tag = "Internal"
)]
fn get_feedback_by_target_id() {}

#[utoipa::path(
    get,
    path = "/internal/feedback/{target_id}/bounds",
    params(
        ("target_id" = String, Path, description = "The target ID"),
    ),
    responses(
        (status = 200, description = "Feedback bounds for target", body = feedback::internal::GetFeedbackBoundsResponse),
        (status = 400, description = "Bad request", body = ErrorBody),
    ),
    tag = "Internal"
)]
fn get_feedback_bounds_by_target_id() {}

#[utoipa::path(
    get,
    path = "/internal/feedback/{target_id}/latest_id_by_metric",
    params(
        ("target_id" = String, Path, description = "The target ID"),
    ),
    responses(
        (status = 200, description = "Latest feedback ID per metric", body = feedback::internal::LatestFeedbackIdByMetricResponse),
        (status = 400, description = "Bad request", body = ErrorBody),
    ),
    tag = "Internal"
)]
fn get_latest_feedback_id_by_metric() {}

#[utoipa::path(
    get,
    path = "/internal/feedback/{target_id}/count",
    params(
        ("target_id" = String, Path, description = "The target ID"),
    ),
    responses(
        (status = 200, description = "Feedback count for target", body = feedback::internal::CountFeedbackByTargetIdResponse),
        (status = 400, description = "Bad request", body = ErrorBody),
    ),
    tag = "Internal"
)]
fn count_feedback_by_target_id() {}

#[utoipa::path(
    get,
    path = "/internal/feedback/timeseries",
    responses(
        (status = 200, description = "Cumulative feedback timeseries", body = feedback::internal::GetCumulativeFeedbackTimeseriesResponse),
        (status = 400, description = "Bad request", body = ErrorBody),
    ),
    tag = "Internal"
)]
fn get_cumulative_feedback_timeseries() {}

#[utoipa::path(
    get,
    path = "/internal/feedback/{inference_id}/demonstrations",
    params(
        ("inference_id" = String, Path, description = "The inference ID"),
    ),
    responses(
        (status = 200, description = "Demonstration feedback", body = feedback::internal::GetDemonstrationFeedbackResponse),
        (status = 400, description = "Bad request", body = ErrorBody),
    ),
    tag = "Internal"
)]
fn get_demonstration_feedback() {}

// ============================================================================
// Internal: Model Inferences
// ============================================================================

#[utoipa::path(
    get,
    path = "/internal/model_inferences/{inference_id}",
    params(
        ("inference_id" = String, Path, description = "The inference ID"),
    ),
    responses(
        (status = 200, description = "Model inferences for inference", body = internal::model_inferences::GetModelInferencesResponse),
        (status = 400, description = "Bad request", body = ErrorBody),
    ),
    tag = "Internal"
)]
fn get_model_inferences() {}

// ============================================================================
// Internal: Inference Metadata
// ============================================================================

#[utoipa::path(
    get,
    path = "/internal/inference_metadata",
    responses(
        (status = 200, description = "Inference metadata list", body = internal::inference_metadata::ListInferenceMetadataResponse),
        (status = 400, description = "Bad request", body = ErrorBody),
    ),
    tag = "Internal"
)]
fn get_inference_metadata() {}

// ============================================================================
// Internal: UI Config
// ============================================================================

#[utoipa::path(
    get,
    path = "/internal/ui_config",
    responses(
        (status = 200, description = "UI config", body = ui::get_config::UiConfig),
    ),
    tag = "Internal"
)]
fn ui_config() {}

#[utoipa::path(
    get,
    path = "/internal/ui_config/{hash}",
    params(
        ("hash" = String, Path, description = "Config snapshot hash"),
    ),
    responses(
        (status = 200, description = "UI config by hash", body = ui::get_config::UiConfig),
        (status = 400, description = "Bad request", body = ErrorBody),
    ),
    tag = "Internal"
)]
fn ui_config_by_hash() {}

// ============================================================================
// Internal: Episodes
// ============================================================================

#[utoipa::path(
    get,
    path = "/internal/episodes",
    responses(
        (status = 200, description = "List of episodes", body = episodes::internal::ListEpisodesResponse),
        (status = 400, description = "Bad request", body = ErrorBody),
    ),
    tag = "Internal"
)]
fn list_episodes_get() {}

#[utoipa::path(
    post,
    path = "/internal/episodes",
    request_body = inline(episodes::internal::ListEpisodesRequest),
    responses(
        (status = 200, description = "List of episodes (with filters)", body = episodes::internal::ListEpisodesResponse),
        (status = 400, description = "Bad request", body = ErrorBody),
    ),
    tag = "Internal"
)]
fn list_episodes_post() {}

#[utoipa::path(
    get,
    path = "/internal/episodes/bounds",
    responses(
        (status = 200, description = "Episode table bounds", body = inline(tensorzero_core::db::TableBoundsWithCount)),
        (status = 400, description = "Bad request", body = ErrorBody),
    ),
    tag = "Internal"
)]
fn query_episode_table_bounds() {}

#[utoipa::path(
    get,
    path = "/internal/episodes/{episode_id}/inference_count",
    params(
        ("episode_id" = String, Path, description = "The episode ID"),
    ),
    responses(
        (status = 200, description = "Episode inference count", body = episodes::internal::GetEpisodeInferenceCountResponse),
        (status = 400, description = "Bad request", body = ErrorBody),
    ),
    tag = "Internal"
)]
fn get_episode_inference_count() {}

// ============================================================================
// Internal: Datasets
// ============================================================================

#[utoipa::path(
    post,
    path = "/internal/datasets/{dataset_name}/datapoints/clone",
    params(
        ("dataset_name" = String, Path, description = "The target dataset name"),
    ),
    request_body = inline(datasets::internal::CloneDatapointsRequest),
    responses(
        (status = 200, description = "Datapoints cloned", body = datasets::internal::CloneDatapointsResponse),
        (status = 400, description = "Bad request", body = ErrorBody),
    ),
    tag = "Internal"
)]
fn clone_datapoints() {}

#[utoipa::path(
    get,
    path = "/internal/datasets/{dataset_name}/datapoints/count",
    params(
        ("dataset_name" = String, Path, description = "The dataset name"),
    ),
    responses(
        (status = 200, description = "Datapoint count", body = datasets::internal::GetDatapointCountResponse),
        (status = 400, description = "Bad request", body = ErrorBody),
    ),
    tag = "Internal"
)]
fn get_datapoint_count() {}

#[utoipa::path(
    get,
    path = "/internal/object_storage",
    responses(
        (status = 200, description = "Object data", body = inline(object_storage::ObjectResponse)),
        (status = 400, description = "Bad request", body = ErrorBody),
    ),
    tag = "Internal"
)]
fn get_object() {}

#[utoipa::path(
    get,
    path = "/internal/datasets",
    responses(
        (status = 200, description = "List of datasets", body = datasets::v1::types::ListDatasetsResponse),
        (status = 400, description = "Bad request", body = ErrorBody),
    ),
    tag = "Internal"
)]
fn internal_list_datasets() {}

// ============================================================================
// Internal: Models
// ============================================================================

#[utoipa::path(
    get,
    path = "/internal/models/count",
    responses(
        (status = 200, description = "Model count", body = internal::models::CountModelsResponse),
        (status = 400, description = "Bad request", body = ErrorBody),
    ),
    tag = "Internal"
)]
fn count_models() {}

#[utoipa::path(
    get,
    path = "/internal/models/usage",
    responses(
        (status = 200, description = "Model usage timeseries", body = internal::models::GetModelUsageResponse),
        (status = 400, description = "Bad request", body = ErrorBody),
    ),
    tag = "Internal"
)]
fn get_model_usage() {}

#[utoipa::path(
    get,
    path = "/internal/models/latency",
    responses(
        (status = 200, description = "Model latency quantiles", body = internal::models::GetModelLatencyResponse),
        (status = 400, description = "Bad request", body = ErrorBody),
    ),
    tag = "Internal"
)]
fn get_model_latency() {}

// ============================================================================
// Internal: Evaluations
// ============================================================================

#[utoipa::path(
    get,
    path = "/internal/evaluations/runs/count",
    responses(
        (status = 200, description = "Evaluation runs count", body = internal::evaluations::types::EvaluationRunStatsResponse),
        (status = 400, description = "Bad request", body = ErrorBody),
    ),
    tag = "Internal"
)]
fn count_evaluation_runs() {}

#[utoipa::path(
    get,
    path = "/internal/evaluations/datapoint_count",
    responses(
        (status = 200, description = "Datapoint count for evaluations", body = internal::evaluations::types::DatapointStatsResponse),
        (status = 400, description = "Bad request", body = ErrorBody),
    ),
    tag = "Internal"
)]
fn count_evaluation_datapoints() {}

#[utoipa::path(
    get,
    path = "/internal/evaluations/runs",
    responses(
        (status = 200, description = "List of evaluation runs", body = internal::evaluations::types::ListEvaluationRunsResponse),
        (status = 400, description = "Bad request", body = ErrorBody),
    ),
    tag = "Internal"
)]
fn list_evaluation_runs() {}

#[utoipa::path(
    get,
    path = "/internal/evaluations/runs/search",
    responses(
        (status = 200, description = "Search results", body = internal::evaluations::types::SearchEvaluationRunsResponse),
        (status = 400, description = "Bad request", body = ErrorBody),
    ),
    tag = "Internal"
)]
fn search_evaluation_runs() {}

#[utoipa::path(
    get,
    path = "/internal/evaluations/run_infos",
    responses(
        (status = 200, description = "Evaluation run infos", body = internal::evaluations::GetEvaluationRunInfosResponse),
        (status = 400, description = "Bad request", body = ErrorBody),
    ),
    tag = "Internal"
)]
fn get_evaluation_run_infos() {}

#[utoipa::path(
    get,
    path = "/internal/evaluations/datapoints/{datapoint_id}/run_infos",
    params(
        ("datapoint_id" = String, Path, description = "The datapoint ID"),
    ),
    responses(
        (status = 200, description = "Evaluation run infos for datapoint", body = internal::evaluations::GetEvaluationRunInfosResponse),
        (status = 400, description = "Bad request", body = ErrorBody),
    ),
    tag = "Internal"
)]
fn get_evaluation_run_infos_for_datapoint() {}

#[utoipa::path(
    get,
    path = "/internal/evaluations/statistics",
    responses(
        (status = 200, description = "Evaluation statistics", body = internal::evaluations::types::GetEvaluationStatisticsResponse),
        (status = 400, description = "Bad request", body = ErrorBody),
    ),
    tag = "Internal"
)]
fn get_evaluation_statistics() {}

#[utoipa::path(
    get,
    path = "/internal/evaluations/results",
    responses(
        (status = 200, description = "Evaluation results", body = internal::evaluations::GetEvaluationResultsResponse),
        (status = 400, description = "Bad request", body = ErrorBody),
    ),
    tag = "Internal"
)]
fn get_evaluation_results() {}

#[utoipa::path(
    post,
    path = "/internal/evaluations/datapoints/{datapoint_id}/get_human_feedback",
    params(
        ("datapoint_id" = String, Path, description = "The datapoint ID"),
    ),
    request_body = inline(internal::evaluations::GetHumanFeedbackRequest),
    responses(
        (status = 200, description = "Human feedback result", body = internal::evaluations::GetHumanFeedbackResponse),
        (status = 400, description = "Bad request", body = ErrorBody),
    ),
    tag = "Internal"
)]
fn get_human_feedback() {}

// ============================================================================
// Internal: Workflow Evaluations
// ============================================================================

#[utoipa::path(
    get,
    path = "/internal/workflow_evaluations/projects",
    responses(
        (status = 200, description = "Workflow evaluation projects", body = workflow_evaluations::internal::GetWorkflowEvaluationProjectsResponse),
        (status = 400, description = "Bad request", body = ErrorBody),
    ),
    tag = "Internal"
)]
fn get_workflow_evaluation_projects() {}

#[utoipa::path(
    get,
    path = "/internal/workflow_evaluations/projects/count",
    responses(
        (status = 200, description = "Workflow evaluation project count", body = workflow_evaluations::internal::GetWorkflowEvaluationProjectCountResponse),
        (status = 400, description = "Bad request", body = ErrorBody),
    ),
    tag = "Internal"
)]
fn get_workflow_evaluation_project_count() {}

#[utoipa::path(
    get,
    path = "/internal/workflow_evaluations/list_runs",
    responses(
        (status = 200, description = "List of workflow evaluation runs", body = workflow_evaluations::internal::ListWorkflowEvaluationRunsResponse),
        (status = 400, description = "Bad request", body = ErrorBody),
    ),
    tag = "Internal"
)]
fn list_workflow_evaluation_runs() {}

#[utoipa::path(
    get,
    path = "/internal/workflow_evaluations/get_runs",
    responses(
        (status = 200, description = "Workflow evaluation runs by IDs", body = workflow_evaluations::internal::GetWorkflowEvaluationRunsResponse),
        (status = 400, description = "Bad request", body = ErrorBody),
    ),
    tag = "Internal"
)]
fn get_workflow_evaluation_runs() {}

#[utoipa::path(
    get,
    path = "/internal/workflow_evaluations/runs/count",
    responses(
        (status = 200, description = "Workflow evaluation runs count", body = workflow_evaluations::internal::CountWorkflowEvaluationRunsResponse),
        (status = 400, description = "Bad request", body = ErrorBody),
    ),
    tag = "Internal"
)]
fn count_workflow_evaluation_runs() {}

#[utoipa::path(
    get,
    path = "/internal/workflow_evaluations/runs/search",
    responses(
        (status = 200, description = "Search workflow evaluation runs", body = workflow_evaluations::internal::SearchWorkflowEvaluationRunsResponse),
        (status = 400, description = "Bad request", body = ErrorBody),
    ),
    tag = "Internal"
)]
fn search_workflow_evaluation_runs() {}

#[utoipa::path(
    get,
    path = "/internal/workflow_evaluations/run_statistics",
    responses(
        (status = 200, description = "Workflow evaluation run statistics", body = workflow_evaluations::internal::GetWorkflowEvaluationRunStatisticsResponse),
        (status = 400, description = "Bad request", body = ErrorBody),
    ),
    tag = "Internal"
)]
fn get_workflow_evaluation_run_statistics() {}

#[utoipa::path(
    get,
    path = "/internal/workflow_evaluations/episodes_by_task_name",
    responses(
        (status = 200, description = "Episodes grouped by task name", body = workflow_evaluations::internal::ListWorkflowEvaluationRunEpisodesByTaskNameResponse),
        (status = 400, description = "Bad request", body = ErrorBody),
    ),
    tag = "Internal"
)]
fn list_workflow_evaluation_run_episodes_by_task_name() {}

#[utoipa::path(
    get,
    path = "/internal/workflow_evaluations/episodes_by_task_name/count",
    responses(
        (status = 200, description = "Episode groups count", body = workflow_evaluations::internal::CountWorkflowEvaluationRunEpisodesByTaskNameResponse),
        (status = 400, description = "Bad request", body = ErrorBody),
    ),
    tag = "Internal"
)]
fn count_workflow_evaluation_run_episodes() {}

#[utoipa::path(
    get,
    path = "/internal/workflow_evaluations/run_episodes",
    responses(
        (status = 200, description = "Workflow evaluation run episodes", body = workflow_evaluations::internal::GetWorkflowEvaluationRunEpisodesWithFeedbackResponse),
        (status = 400, description = "Bad request", body = ErrorBody),
    ),
    tag = "Internal"
)]
fn get_workflow_evaluation_run_episodes() {}

#[utoipa::path(
    get,
    path = "/internal/workflow_evaluations/run_episodes/count",
    responses(
        (status = 200, description = "Total episodes count for run", body = workflow_evaluations::internal::CountWorkflowEvaluationRunEpisodesResponse),
        (status = 400, description = "Bad request", body = ErrorBody),
    ),
    tag = "Internal"
)]
fn count_workflow_evaluation_run_episodes_total() {}

// ============================================================================
// Internal: Config Snapshots
// ============================================================================

#[utoipa::path(
    get,
    path = "/internal/config",
    responses(
        (status = 200, description = "Live config snapshot", body = internal::config::GetConfigResponse),
        (status = 400, description = "Bad request", body = ErrorBody),
    ),
    tag = "Internal"
)]
fn get_live_config() {}

#[utoipa::path(
    post,
    path = "/internal/config",
    request_body = inline(internal::config::WriteConfigRequest),
    responses(
        (status = 200, description = "Config snapshot written", body = internal::config::WriteConfigResponse),
        (status = 400, description = "Bad request", body = ErrorBody),
    ),
    tag = "Internal"
)]
fn write_config() {}

#[utoipa::path(
    get,
    path = "/internal/config/{hash}",
    params(
        ("hash" = String, Path, description = "Config snapshot hash"),
    ),
    responses(
        (status = 200, description = "Config snapshot by hash", body = internal::config::GetConfigResponse),
        (status = 400, description = "Bad request", body = ErrorBody),
    ),
    tag = "Internal"
)]
fn get_config_by_hash() {}

// ============================================================================
// Internal: Count Inferences
// ============================================================================

#[utoipa::path(
    post,
    path = "/internal/inferences/count",
    request_body = inline(internal::count_inferences::CountInferencesRequest),
    responses(
        (status = 200, description = "Inference count", body = internal::count_inferences::CountInferencesResponse),
        (status = 400, description = "Bad request", body = ErrorBody),
    ),
    tag = "Internal"
)]
fn count_inferences() {}

// ============================================================================
// Internal: Resolve UUID
// ============================================================================

#[utoipa::path(
    get,
    path = "/internal/resolve_uuid/{id}",
    params(
        ("id" = String, Path, description = "The UUID to resolve"),
    ),
    responses(
        (status = 200, description = "Resolved UUID", body = inline(tensorzero_core::db::resolve_uuid::ResolveUuidResponse)),
        (status = 400, description = "Bad request", body = ErrorBody),
    ),
    tag = "Internal"
)]
fn resolve_uuid() {}

// ============================================================================
// Internal: Autopilot
// ============================================================================

#[utoipa::path(
    get,
    path = "/internal/autopilot/status",
    responses(
        (status = 200, description = "Autopilot status", body = internal::autopilot::AutopilotStatusResponse),
    ),
    tag = "Internal"
)]
fn autopilot_status() {}

#[utoipa::path(
    get,
    path = "/internal/autopilot/v1/sessions",
    responses(
        (status = 200, description = "List of autopilot sessions", body = inline(autopilot_client::ListSessionsResponse)),
        (status = 400, description = "Bad request", body = ErrorBody),
    ),
    tag = "Internal"
)]
fn list_autopilot_sessions() {}

#[utoipa::path(
    get,
    path = "/internal/autopilot/v1/sessions/{session_id}/events",
    params(
        ("session_id" = String, Path, description = "The session ID"),
    ),
    responses(
        (status = 200, description = "List of events", body = inline(autopilot_client::GatewayListEventsResponse)),
        (status = 400, description = "Bad request", body = ErrorBody),
    ),
    tag = "Internal"
)]
fn list_autopilot_events() {}

#[utoipa::path(
    post,
    path = "/internal/autopilot/v1/sessions/{session_id}/events",
    params(
        ("session_id" = String, Path, description = "The session ID"),
    ),
    request_body = inline(internal::autopilot::CreateEventGatewayRequest),
    responses(
        (status = 200, description = "Event created", body = inline(autopilot_client::CreateEventResponse)),
        (status = 400, description = "Bad request", body = ErrorBody),
    ),
    tag = "Internal"
)]
fn create_autopilot_event() {}

#[utoipa::path(
    get,
    path = "/internal/autopilot/v1/sessions/{session_id}/events/stream",
    params(
        ("session_id" = String, Path, description = "The session ID"),
    ),
    responses(
        (status = 200, description = "SSE stream of events", content_type = "text/event-stream"),
        (status = 400, description = "Bad request", body = ErrorBody),
    ),
    tag = "Internal"
)]
fn stream_autopilot_events() {}

#[utoipa::path(
    post,
    path = "/internal/autopilot/v1/sessions/{session_id}/actions/approve_all",
    params(
        ("session_id" = String, Path, description = "The session ID"),
    ),
    request_body = inline(internal::autopilot::ApproveAllToolCallsGatewayRequest),
    responses(
        (status = 200, description = "Tool calls approved", body = inline(autopilot_client::ApproveAllToolCallsResponse)),
        (status = 400, description = "Bad request", body = ErrorBody),
    ),
    tag = "Internal"
)]
fn approve_all_tool_calls() {}

#[utoipa::path(
    post,
    path = "/internal/autopilot/v1/sessions/{session_id}/actions/interrupt",
    params(
        ("session_id" = String, Path, description = "The session ID"),
    ),
    responses(
        (status = 200, description = "Session interrupted"),
        (status = 400, description = "Bad request", body = ErrorBody),
    ),
    tag = "Internal"
)]
fn interrupt_session() {}

#[utoipa::path(
    get,
    path = "/internal/autopilot/v1/sessions/{session_id}/config-writes",
    params(
        ("session_id" = String, Path, description = "The session ID"),
    ),
    responses(
        (status = 200, description = "Config writes for session", body = inline(autopilot_client::GatewayListConfigWritesResponse)),
        (status = 400, description = "Bad request", body = ErrorBody),
    ),
    tag = "Internal"
)]
fn list_config_writes() {}

#[utoipa::path(
    post,
    path = "/internal/autopilot/v1/sessions/{session_id}/aws/s3_initiate_upload",
    params(
        ("session_id" = String, Path, description = "The session ID"),
    ),
    request_body = inline(internal::autopilot::S3InitiateUploadGatewayRequest),
    responses(
        (status = 200, description = "S3 upload initiated", body = inline(autopilot_client::S3UploadResponse)),
        (status = 400, description = "Bad request", body = ErrorBody),
    ),
    tag = "Internal"
)]
fn s3_initiate_upload() {}

// ============================================================================
// Internal: Action
// ============================================================================

#[utoipa::path(
    post,
    path = "/internal/action",
    request_body = inline(Object),
    responses(
        (status = 200, description = "Action result", body = Object),
        (status = 400, description = "Bad request", body = ErrorBody),
    ),
    tag = "Internal"
)]
fn internal_action() {}

// ============================================================================
// Internal: Run Evaluation (SSE)
// ============================================================================

#[utoipa::path(
    post,
    path = "/internal/evaluations/run",
    responses(
        (status = 200, description = "SSE stream of evaluation results", content_type = "text/event-stream"),
        (status = 400, description = "Bad request", body = ErrorBody),
    ),
    tag = "Internal"
)]
fn run_evaluation() {}

// ============================================================================
// OpenAPI Spec
// ============================================================================

#[derive(OpenApi)]
#[openapi(
    info(
        title = "TensorZero Gateway API",
        version = env!("CARGO_PKG_VERSION"),
        description = "TensorZero Gateway — inference, feedback, and observability API",
    ),
    paths(
        // Inference
        inference,
        // Batch Inference
        start_batch_inference,
        poll_batch_inference,
        poll_batch_inference_single,
        // Feedback
        feedback,
        // Observability
        list_inferences,
        get_inferences,
        // Datasets
        create_datapoints,
        update_datapoints,
        delete_datapoints,
        update_datapoints_metadata,
        create_from_inferences,
        list_datapoints,
        delete_dataset,
        get_datapoints_by_dataset,
        // Workflow Evaluation
        create_workflow_evaluation_run,
        create_workflow_evaluation_run_episode,
        // Status
        gateway_status,
        health,
        metrics,
        // Internal: Variant Sampling Probabilities
        get_variant_sampling_probabilities,
        get_variant_sampling_probabilities_by_function,
        // Internal: Function Metrics & Variant Performances
        get_function_metrics,
        get_variant_performances,
        // Internal: Inference Count
        list_functions_with_inference_count,
        get_inference_count,
        get_inference_with_feedback_count,
        get_function_throughput_by_variant,
        // Internal: Feedback
        get_feedback_by_target_id,
        get_feedback_bounds_by_target_id,
        get_latest_feedback_id_by_metric,
        count_feedback_by_target_id,
        get_cumulative_feedback_timeseries,
        get_demonstration_feedback,
        // Internal: Model Inferences
        get_model_inferences,
        // Internal: Inference Metadata
        get_inference_metadata,
        // Internal: UI Config
        ui_config,
        ui_config_by_hash,
        // Internal: Episodes
        list_episodes_get,
        list_episodes_post,
        query_episode_table_bounds,
        get_episode_inference_count,
        // Internal: Datasets
        clone_datapoints,
        get_datapoint_count,
        get_object,
        internal_list_datasets,
        // Internal: Models
        count_models,
        get_model_usage,
        get_model_latency,
        // Internal: Evaluations
        count_evaluation_runs,
        count_evaluation_datapoints,
        list_evaluation_runs,
        search_evaluation_runs,
        get_evaluation_run_infos,
        get_evaluation_run_infos_for_datapoint,
        get_evaluation_statistics,
        get_evaluation_results,
        get_human_feedback,
        run_evaluation,
        // Internal: Workflow Evaluations
        get_workflow_evaluation_projects,
        get_workflow_evaluation_project_count,
        list_workflow_evaluation_runs,
        get_workflow_evaluation_runs,
        count_workflow_evaluation_runs,
        search_workflow_evaluation_runs,
        get_workflow_evaluation_run_statistics,
        list_workflow_evaluation_run_episodes_by_task_name,
        count_workflow_evaluation_run_episodes,
        get_workflow_evaluation_run_episodes,
        count_workflow_evaluation_run_episodes_total,
        // Internal: Config Snapshots
        get_live_config,
        write_config,
        get_config_by_hash,
        // Internal: Count Inferences
        count_inferences,
        // Internal: Action
        internal_action,
        // Internal: Resolve UUID
        resolve_uuid,
        // Internal: Autopilot
        autopilot_status,
        list_autopilot_sessions,
        list_autopilot_events,
        create_autopilot_event,
        stream_autopilot_events,
        approve_all_tool_calls,
        interrupt_session,
        list_config_writes,
        s3_initiate_upload,
    ),
    tags(
        (name = "Inference", description = "Run model inference"),
        (name = "Batch Inference", description = "Run batch model inference"),
        (name = "Feedback", description = "Submit feedback on inferences"),
        (name = "Observability", description = "Query stored inferences"),
        (name = "Datasets", description = "Manage datasets and datapoints"),
        (name = "Workflow Evaluation", description = "Manage workflow evaluation runs"),
        (name = "Status", description = "Gateway status, health, and metrics"),
        (name = "Internal", description = "Internal API endpoints"),
    )
)]
struct ApiDoc;

fn main() {
    let spec = ApiDoc::openapi()
        .to_pretty_json()
        .expect("Failed to serialize OpenAPI spec");
    print!("{spec}");
}
