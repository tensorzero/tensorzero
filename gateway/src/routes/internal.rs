//! Internal route definitions for the TensorZero Gateway API.
//!
//! These routes are for internal use. They are unstable and might change without notice,
//! and do not export any OpenTelemetry spans.

use axum::{
    Router,
    routing::{get, post, put},
};
use tensorzero_core::endpoints;
use tensorzero_core::utils::gateway::AppStateData;

pub fn build_internal_non_otel_enabled_routes() -> Router<AppStateData> {
    Router::new()
        // Deprecated (#4652): Remove the endpoint without the `/internal` prefix.
        .route(
            "/variant_sampling_probabilities",
            get(endpoints::variant_probabilities::get_variant_sampling_probabilities_handler),
        )
        .route(
            "/internal/functions/{function_name}/variant_sampling_probabilities",
            get(endpoints::variant_probabilities::get_variant_sampling_probabilities_by_function_handler),
        )
        .route(
            "/internal/functions/{function_name}/metrics",
            get(endpoints::functions::internal::get_function_metrics_handler),
        )
        .route(
            "/internal/functions/{function_name}/inference-stats",
            get(endpoints::internal::inference_stats::get_inference_stats_handler),
        )
        .route(
            "/internal/functions/{function_name}/inference-stats/{metric_name}",
            get(endpoints::internal::inference_stats::get_inference_with_feedback_stats_handler),
        )
        .route(
            "/internal/feedback/{target_id}",
            get(endpoints::feedback::internal::get_feedback_by_target_id_handler),
        )
        .route(
            "/internal/feedback/{target_id}/bounds",
            get(endpoints::feedback::internal::get_feedback_bounds_by_target_id_handler),
        )
        .route(
            "/internal/feedback/{target_id}/latest-id-by-metric",
            get(endpoints::feedback::internal::get_latest_feedback_id_by_metric_handler),
        )
        .route(
            "/internal/feedback/{target_id}/count",
            get(endpoints::feedback::internal::count_feedback_by_target_id_handler),
        )
        .route(
            "/internal/model_inferences/{inference_id}",
            get(endpoints::internal::model_inferences::get_model_inferences_handler),
        )
        .route(
            "/internal/inference_metadata",
            get(endpoints::internal::inference_metadata::get_inference_metadata_handler),
        )
        .route(
            "/internal/ui-config",
            get(endpoints::ui::get_config::ui_config_handler),
        )
        .route(
            "/internal/episodes",
            get(endpoints::episodes::internal::list_episodes_handler),
        )
        .route(
            "/internal/episodes/bounds",
            get(endpoints::episodes::internal::query_episode_table_bounds_handler),
        )
        .route(
            "/internal/episodes/{episode_id}/inference-count",
            get(endpoints::episodes::internal::get_episode_inference_count_handler),
        )
        .route(
            "/internal/datasets/{dataset_name}/datapoints",
            #[expect(deprecated)]
            post(endpoints::datasets::deprecated_create_datapoints_from_inferences_handler),
        )
        .route(
            "/internal/datasets/{dataset_name}/datapoints/clone",
            post(endpoints::datasets::internal::clone_datapoints_handler),
        )
        .route(
            "/internal/datasets/{dataset_name}/datapoints/{datapoint_id}",
            put(endpoints::datasets::update_datapoint_handler),
        )
        .route(
            "/internal/datasets/{dataset_name}/datapoints/count",
            get(endpoints::datasets::internal::get_datapoint_count_handler),
        )
        .route(
            "/internal/object_storage",
            get(endpoints::object_storage::get_object_handler),
        )
        .route(
            "/internal/datasets",
            get(endpoints::datasets::v1::list_datasets_handler),
        )
         // Model statistics endpoints
         .route(
             "/internal/models/count",
             get(endpoints::internal::models::count_models_handler),
         )
        // Evaluation endpoints
        .route(
            "/internal/evaluations/run-stats",
            get(endpoints::internal::evaluations::get_evaluation_run_stats_handler),
        )
        .route(
            "/internal/evaluations/datapoint-count",
            get(endpoints::internal::evaluations::count_datapoints_handler),
        )
        .route(
            "/internal/evaluations/runs",
            get(endpoints::internal::evaluations::list_evaluation_runs_handler),
        )
        .route(
            "/internal/evaluations/runs/search",
            get(endpoints::internal::evaluations::search_evaluation_runs_handler),
        )
        .route(
            "/internal/evaluations/run-infos",
            get(endpoints::internal::evaluations::get_evaluation_run_infos_handler),
        )
        // Workflow evaluation endpoints
        .route(
            "/internal/workflow-evaluations/projects",
            get(endpoints::workflow_evaluations::internal::get_workflow_evaluation_projects_handler),
        )
        .route(
            "/internal/workflow-evaluations/projects/count",
            get(
                endpoints::workflow_evaluations::internal::get_workflow_evaluation_project_count_handler,
            ),
        )
        .route(
            "/internal/workflow-evaluations/list-runs",
            get(endpoints::workflow_evaluations::internal::list_workflow_evaluation_runs_handler),
        )
        .route(
            "/internal/workflow-evaluations/runs/count",
            get(endpoints::workflow_evaluations::internal::count_workflow_evaluation_runs_handler),
        )
        .route(
            "/internal/workflow-evaluations/runs/search",
            get(endpoints::workflow_evaluations::internal::search_workflow_evaluation_runs_handler),
        )
        .route(
            "/internal/models/usage",
            get(endpoints::internal::models::get_model_usage_handler),
        )
        .route(
            "/internal/models/latency",
            get(endpoints::internal::models::get_model_latency_handler),
        )
        // Config snapshot endpoints
        .route(
            "/internal/config",
            get(endpoints::internal::config::get_live_config_handler),
        )
        .route(
            "/internal/config/{hash}",
            get(endpoints::internal::config::get_config_by_hash_handler),
        )
        // Inference count endpoint
        .route(
            "/internal/inferences/count",
            post(endpoints::internal::count_inferences::count_inferences_handler),
        )
        // Action endpoint for executing with historical config snapshots
        .route(
            "/internal/action",
            post(endpoints::internal::action::action_handler),
        )
}
