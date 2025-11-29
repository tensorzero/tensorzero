//! External route definitions for the TensorZero Gateway API.
//!
//! This file should remain minimal, containing only endpoint path definitions and their handler mappings.
//! Router construction logic belongs in `router.rs`. This constraint exists because CODEOWNERS
//! requires specific review for route changes.

use axum::{
    routing::{delete, get, patch, post},
    Router,
};
use metrics_exporter_prometheus::PrometheusHandle;
use tensorzero_core::endpoints::openai_compatible::build_openai_compatible_routes;
use tensorzero_core::observability::OtelEnabledRoutes;
use tensorzero_core::{endpoints, utils::gateway::AppStateData};

/// Defines routes that should have top-level OpenTelemetry HTTP spans created
/// All of these routes will have a span named `METHOD <ROUTE>` (e.g. `POST /batch_inference/{batch_id}`)
/// sent to OpenTelemetry
pub fn build_otel_enabled_routes() -> (OtelEnabledRoutes, Router<AppStateData>) {
    let mut routes = vec![
        ("/inference", post(endpoints::inference::inference_handler)),
        (
            "/batch_inference",
            post(endpoints::batch_inference::start_batch_inference_handler),
        ),
        (
            "/batch_inference/{batch_id}",
            get(endpoints::batch_inference::poll_batch_inference_handler),
        ),
        (
            "/batch_inference/{batch_id}/inference/{inference_id}",
            get(endpoints::batch_inference::poll_batch_inference_handler),
        ),
        ("/feedback", post(endpoints::feedback::feedback_handler)),
    ];
    routes.extend(build_openai_compatible_routes().routes);
    let mut router = Router::new();
    let mut route_names = Vec::with_capacity(routes.len());
    for (path, handler) in routes {
        route_names.push(path);
        router = router.route(path, handler);
    }
    (
        OtelEnabledRoutes {
            routes: route_names,
        },
        router,
    )
}

/// Builds external routes that don't have OpenTelemetry tracing.
pub fn build_non_otel_enabled_routes(metrics_handle: PrometheusHandle) -> Router<AppStateData> {
    Router::new()
        .merge(build_observability_routes())
        .merge(build_datasets_routes())
        .merge(build_optimization_routes())
        .merge(build_evaluations_routes())
        .merge(build_meta_observability_routes(metrics_handle))
}

/// This function builds the public routes for observability.
fn build_observability_routes() -> Router<AppStateData> {
    Router::new()
        .route(
            "/v1/inferences/list_inferences",
            post(endpoints::stored_inferences::v1::list_inferences_handler),
        )
        .route(
            "/v1/inferences/get_inferences",
            post(endpoints::stored_inferences::v1::get_inferences_handler),
        )
}

/// This function builds the public routes for datasets.
fn build_datasets_routes() -> Router<AppStateData> {
    Router::new()
        .route(
            "/datasets/{dataset_name}/datapoints",
            #[expect(deprecated)]
            post(endpoints::datasets::create_datapoints_handler),
        )
        // TODO(#3459): Deprecated in #3721. Remove in a future release.
        .route(
            "/datasets/{dataset_name}/datapoints/bulk",
            #[expect(deprecated)]
            post(endpoints::datasets::bulk_insert_datapoints_handler),
        )
        .route(
            "/datasets/{dataset_name}/datapoints/{datapoint_id}",
            delete(endpoints::datasets::delete_datapoint_handler),
        )
        .route(
            "/datasets/{dataset_name}/datapoints",
            get(endpoints::datasets::list_datapoints_handler),
        )
        .route(
            "/datasets/{dataset_name}",
            delete(endpoints::datasets::stale_dataset_handler),
        )
        .route(
            "/datasets/{dataset_name}/datapoints/{datapoint_id}",
            get(endpoints::datasets::get_datapoint_handler),
        )
        .route(
            "/v1/datasets/{dataset_name}/datapoints",
            post(endpoints::datasets::v1::create_datapoints_handler)
                .patch(endpoints::datasets::v1::update_datapoints_handler)
                .delete(endpoints::datasets::v1::delete_datapoints_handler),
        )
        .route(
            "/v1/datasets/{dataset_name}/datapoints/metadata",
            patch(endpoints::datasets::v1::update_datapoints_metadata_handler),
        )
        .route(
            "/v1/datasets/{dataset_name}/from_inferences",
            post(endpoints::datasets::v1::create_from_inferences_handler),
        )
        .route(
            "/v1/datasets/{dataset_name}/list_datapoints",
            post(endpoints::datasets::v1::list_datapoints_handler),
        )
        .route(
            "/v1/datasets/{dataset_name}",
            delete(endpoints::datasets::v1::delete_dataset_handler),
        )
        .route(
            "/v1/datasets/{dataset_name}/get_datapoints",
            post(endpoints::datasets::v1::get_datapoints_by_dataset_handler),
        )
        // DEPRECATED: prefer /v1/datasets/{dataset_name}/get_datapoints
        .route(
            "/v1/datasets/get_datapoints",
            post(endpoints::datasets::v1::get_datapoints_handler),
        )
}

/// This function builds the public routes for optimization.
fn build_optimization_routes() -> Router<AppStateData> {
    Router::new()
        .route(
            "/experimental_optimization_workflow",
            post(tensorzero_optimizers::endpoints::launch_optimization_workflow_handler),
        )
        .route(
            "/experimental_optimization/{job_handle}",
            get(tensorzero_optimizers::endpoints::poll_optimization_handler),
        )
}

/// This function builds the public routes for evaluations.
fn build_evaluations_routes() -> Router<AppStateData> {
    Router::new()
        // Workflow evaluation endpoints (new)
        .route(
            "/workflow_evaluation_run",
            post(endpoints::workflow_evaluation_run::workflow_evaluation_run_handler),
        )
        .route(
            "/workflow_evaluation_run/{run_id}/episode",
            post(endpoints::workflow_evaluation_run::workflow_evaluation_run_episode_handler),
        )
        // DEPRECATED: Use /workflow_evaluation_run endpoints instead
        .route(
            "/dynamic_evaluation_run",
            post(endpoints::workflow_evaluation_run::dynamic_evaluation_run_handler),
        )
        .route(
            "/dynamic_evaluation_run/{run_id}/episode",
            post(endpoints::workflow_evaluation_run::dynamic_evaluation_run_episode_handler),
        )
}

/// This function builds the public routes for meta-observability (e.g. gateway health).
fn build_meta_observability_routes(metrics_handle: PrometheusHandle) -> Router<AppStateData> {
    Router::new()
        .route(
            "/metrics",
            get(move || std::future::ready(metrics_handle.render())),
        )
        .route("/status", get(endpoints::status::status_handler))
        .route("/health", get(endpoints::status::health_handler))
}
