#![expect(clippy::print_stdout, clippy::expect_used)]

use tensorzero_core::utils::gateway::AppStateData;
use utoipa::OpenApi;
use utoipa_axum::router::OpenApiRouter;
use utoipa_axum::routes;

#[derive(OpenApi)]
#[openapi(
    info(
        title = "TensorZero Gateway API",
        version = env!("CARGO_PKG_VERSION"),
        description = "TensorZero Gateway — inference, feedback, and observability API",
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
    let (_, api) = OpenApiRouter::<AppStateData>::with_openapi(ApiDoc::openapi())
        // Inference
        .routes(routes!(tensorzero_core::endpoints::inference::inference_handler))
        // Batch Inference
        .routes(routes!(tensorzero_core::endpoints::batch_inference::start_batch_inference_handler))
        .routes(routes!(tensorzero_core::endpoints::batch_inference::poll_batch_inference_handler))
        .routes(routes!(tensorzero_core::endpoints::batch_inference::poll_batch_inference_by_inference_handler))
        // Feedback
        .routes(routes!(tensorzero_core::endpoints::feedback::feedback_handler))
        // Stored Inferences
        .routes(routes!(tensorzero_core::endpoints::stored_inferences::v1::get_inferences::get_inferences_handler))
        .routes(routes!(tensorzero_core::endpoints::stored_inferences::v1::get_inferences::list_inferences_handler))
        // Datasets (v1)
        .routes(routes!(tensorzero_core::endpoints::datasets::v1::create_datapoints::create_datapoints_handler))
        .routes(routes!(tensorzero_core::endpoints::datasets::v1::create_from_inferences::create_from_inferences_handler))
        .routes(routes!(tensorzero_core::endpoints::datasets::v1::delete_datapoints::delete_datapoints_handler))
        .routes(routes!(tensorzero_core::endpoints::datasets::v1::delete_datapoints::delete_dataset_handler))
        .routes(routes!(tensorzero_core::endpoints::datasets::v1::get_datapoints::get_datapoints_by_dataset_handler))
        .routes(routes!(tensorzero_core::endpoints::datasets::v1::get_datapoints::list_datapoints_handler))
        .routes(routes!(tensorzero_core::endpoints::datasets::v1::list_datasets::list_datasets_handler))
        .routes(routes!(tensorzero_core::endpoints::datasets::v1::update_datapoints::update_datapoints_handler))
        .routes(routes!(tensorzero_core::endpoints::datasets::v1::update_datapoints::update_datapoints_metadata_handler))
        // Workflow Evaluation Run
        .routes(routes!(tensorzero_core::endpoints::workflow_evaluation_run::workflow_evaluation_run_handler))
        .routes(routes!(tensorzero_core::endpoints::workflow_evaluation_run::workflow_evaluation_run_episode_handler))
        // Status
        .routes(routes!(tensorzero_core::endpoints::status::status_handler))
        .routes(routes!(tensorzero_core::endpoints::status::health_handler))
        // Variant Probabilities
        .routes(routes!(tensorzero_core::endpoints::variant_probabilities::get_variant_sampling_probabilities_handler))
        .routes(routes!(tensorzero_core::endpoints::variant_probabilities::get_variant_sampling_probabilities_by_function_handler))
        // Internal: Functions
        .routes(routes!(tensorzero_core::endpoints::functions::internal::get_function_metrics_handler))
        .routes(routes!(tensorzero_core::endpoints::functions::internal::get_variant_performances_handler))
        // Internal: Inference Count
        .routes(routes!(tensorzero_core::endpoints::internal::inference_count::get_inference_count_handler))
        .routes(routes!(tensorzero_core::endpoints::internal::inference_count::get_inference_with_feedback_count_handler))
        .routes(routes!(tensorzero_core::endpoints::internal::inference_count::list_functions_with_inference_count_handler))
        .routes(routes!(tensorzero_core::endpoints::internal::inference_count::get_function_throughput_by_variant_handler))
        // Internal: Feedback
        .routes(routes!(tensorzero_core::endpoints::feedback::internal::get_feedback_by_target_id::get_feedback_by_target_id_handler))
        .routes(routes!(tensorzero_core::endpoints::feedback::internal::count_feedback::count_feedback_by_target_id_handler))
        .routes(routes!(tensorzero_core::endpoints::feedback::internal::get_feedback_bounds::get_feedback_bounds_by_target_id_handler))
        .routes(routes!(tensorzero_core::endpoints::feedback::internal::get_demonstration_feedback::get_demonstration_feedback_handler))
        .routes(routes!(tensorzero_core::endpoints::feedback::internal::latest_feedback_by_metric::get_latest_feedback_id_by_metric_handler))
        .routes(routes!(tensorzero_core::endpoints::feedback::internal::cumulative_feedback_timeseries::get_cumulative_feedback_timeseries_handler))
        // Internal: Model Inferences
        .routes(routes!(tensorzero_core::endpoints::internal::model_inferences::get_model_inferences_handler))
        // Internal: Inference Metadata
        .routes(routes!(tensorzero_core::endpoints::internal::inference_metadata::get_inference_metadata_handler))
        // Internal: UI Config
        .routes(routes!(tensorzero_core::endpoints::ui::get_config::ui_config_handler))
        .routes(routes!(tensorzero_core::endpoints::ui::get_config::ui_config_by_hash_handler))
        // Internal: Episodes
        .routes(routes!(tensorzero_core::endpoints::episodes::internal::list_episodes::list_episodes_handler))
        .routes(routes!(tensorzero_core::endpoints::episodes::internal::list_episodes::list_episodes_post_handler))
        .routes(routes!(tensorzero_core::endpoints::episodes::internal::list_episodes::query_episode_table_bounds_handler))
        .routes(routes!(tensorzero_core::endpoints::episodes::internal::get_episode_inference_count::get_episode_inference_count_handler))
        // Internal: Datasets
        .routes(routes!(tensorzero_core::endpoints::datasets::internal::clone_datapoints::clone_datapoints_handler))
        .routes(routes!(tensorzero_core::endpoints::datasets::internal::get_datapoint_count::get_datapoint_count_handler))
        // Internal: Object Storage
        .routes(routes!(tensorzero_core::endpoints::object_storage::get_object_handler))
        // Internal: Models
        .routes(routes!(tensorzero_core::endpoints::internal::models::get_model_usage_handler))
        .routes(routes!(tensorzero_core::endpoints::internal::models::get_model_latency_handler))
        .routes(routes!(tensorzero_core::endpoints::internal::models::count_models_handler))
        // Internal: Evaluations
        .routes(routes!(tensorzero_core::endpoints::internal::evaluations::list_runs::list_evaluation_runs_handler))
        .routes(routes!(tensorzero_core::endpoints::internal::evaluations::count_runs::count_evaluation_runs_handler))
        .routes(routes!(tensorzero_core::endpoints::internal::evaluations::search_runs::search_evaluation_runs_handler))
        .routes(routes!(tensorzero_core::endpoints::internal::evaluations::get_evaluation_results::get_evaluation_results_handler))
        .routes(routes!(tensorzero_core::endpoints::internal::evaluations::get_statistics::get_evaluation_statistics_handler))
        .routes(routes!(tensorzero_core::endpoints::internal::evaluations::get_run_infos::get_evaluation_run_infos_handler))
        .routes(routes!(tensorzero_core::endpoints::internal::evaluations::get_run_infos::get_evaluation_run_infos_for_datapoint_handler))
        .routes(routes!(tensorzero_core::endpoints::internal::evaluations::get_human_feedback::get_human_feedback_handler))
        .routes(routes!(tensorzero_core::endpoints::internal::evaluations::count_datapoints::count_datapoints_handler))
        // Internal: Workflow Evaluations
        .routes(routes!(tensorzero_core::endpoints::workflow_evaluations::internal::get_projects::get_workflow_evaluation_projects_handler))
        .routes(routes!(tensorzero_core::endpoints::workflow_evaluations::internal::get_project_count::get_workflow_evaluation_project_count_handler))
        .routes(routes!(tensorzero_core::endpoints::workflow_evaluations::internal::get_runs::get_workflow_evaluation_runs_handler))
        .routes(routes!(tensorzero_core::endpoints::workflow_evaluations::internal::list_runs::list_workflow_evaluation_runs_handler))
        .routes(routes!(tensorzero_core::endpoints::workflow_evaluations::internal::count_runs::count_workflow_evaluation_runs_handler))
        .routes(routes!(tensorzero_core::endpoints::workflow_evaluations::internal::search_runs::search_workflow_evaluation_runs_handler))
        .routes(routes!(tensorzero_core::endpoints::workflow_evaluations::internal::get_run_statistics::get_workflow_evaluation_run_statistics_handler))
        .routes(routes!(tensorzero_core::endpoints::workflow_evaluations::internal::list_episodes_by_task_name::list_workflow_evaluation_run_episodes_by_task_name_handler))
        .routes(routes!(tensorzero_core::endpoints::workflow_evaluations::internal::count_episodes_by_task_name::count_workflow_evaluation_run_episodes_handler))
        .routes(routes!(tensorzero_core::endpoints::workflow_evaluations::internal::get_run_episodes::get_workflow_evaluation_run_episodes_handler))
        .routes(routes!(tensorzero_core::endpoints::workflow_evaluations::internal::count_run_episodes::count_workflow_evaluation_run_episodes_total_handler))
        // Internal: Config Snapshots
        .routes(routes!(tensorzero_core::endpoints::internal::config::get_live_config_handler))
        .routes(routes!(tensorzero_core::endpoints::internal::config::write_config_handler))
        .routes(routes!(tensorzero_core::endpoints::internal::config::get_config_by_hash_handler))
        // Internal: Count Inferences
        .routes(routes!(tensorzero_core::endpoints::internal::count_inferences::count_inferences_handler))
        // Internal: Resolve UUID
        .routes(routes!(tensorzero_core::endpoints::internal::resolve_uuid::resolve_uuid_handler))
        // Internal: Autopilot
        .routes(routes!(tensorzero_core::endpoints::internal::autopilot::autopilot_status_handler))
        .routes(routes!(tensorzero_core::endpoints::internal::autopilot::list_sessions_handler))
        .routes(routes!(tensorzero_core::endpoints::internal::autopilot::list_events_handler))
        .routes(routes!(tensorzero_core::endpoints::internal::autopilot::create_event_handler))
        .routes(routes!(tensorzero_core::endpoints::internal::autopilot::stream_events_handler))
        .routes(routes!(tensorzero_core::endpoints::internal::autopilot::approve_all_tool_calls_handler))
        .routes(routes!(tensorzero_core::endpoints::internal::autopilot::interrupt_session_handler))
        .routes(routes!(tensorzero_core::endpoints::internal::autopilot::list_config_writes_handler))
        .routes(routes!(tensorzero_core::endpoints::internal::autopilot::s3_initiate_upload_handler))
        // Gateway: Action
        .routes(routes!(gateway::routes::action::action_handler))
        // Gateway: Evaluation Run
        .routes(routes!(gateway::routes::evaluations::run_evaluation_handler))
        .split_for_parts();

    let spec = api
        .to_pretty_json()
        .expect("Failed to serialize OpenAPI spec");
    print!("{spec}");
}
