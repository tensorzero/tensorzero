//! Route definitions and endpoint mappings for the TensorZero Gateway API.
//!
//! This file should remain minimal, containing only endpoint path definitions and their handler mappings.
//! Router construction logic belongs in `router.rs`. This constraint exists because CODEOWNERS
//! requires specific review for route changes.

use axum::{
    routing::{delete, get, patch, post, put},
    Router,
};
use metrics_exporter_prometheus::PrometheusHandle;
use std::sync::Arc;
use tensorzero_core::observability::OtelEnabledRoutes;
use tensorzero_core::{endpoints, utils::gateway::AppStateData};
use tensorzero_core::{
    endpoints::openai_compatible::build_openai_compatible_routes,
    observability::{RouterExt as _, TracerWrapper},
};

pub fn build_api_routes(
    otel_tracer: Option<Arc<TracerWrapper>>,
    metrics_handle: PrometheusHandle,
) -> Router<AppStateData> {
    let (otel_enabled_routes, otel_enabled_router) = build_otel_enabled_routes();
    Router::new()
        .merge(otel_enabled_router)
        .merge(build_non_otel_enabled_routes(metrics_handle))
        .apply_top_level_otel_http_trace_layer(otel_tracer, otel_enabled_routes)
}

/// Defines routes that should have top-level OpenTelemetry HTTP spans created
/// All of these routes will have a span named `METHOD <ROUTE>` (e.g. `POST /batch_inference/{batch_id}`)
/// sent to OpenTelemetry
fn build_otel_enabled_routes() -> (OtelEnabledRoutes, Router<AppStateData>) {
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

// Defines routes that should not have top-level OpenTelemetry HTTP spans created
// We use this for internal routes which we don't want to expose to users,
// or uninteresting routes like /health
fn build_non_otel_enabled_routes(metrics_handle: PrometheusHandle) -> Router<AppStateData> {
    Router::new()
        .merge(build_observability_routes())
        .merge(build_datasets_routes())
        .merge(build_optimization_routes())
        .merge(build_evaluations_routes())
        .merge(build_internal_routes())
        .merge(build_meta_observability_routes(metrics_handle))
}

/// This function builds the public routes for observability.
///
/// IMPORTANT: Add internal routes to `build_internal_routes` instead.
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
///
/// IMPORTANT: Add internal routes to `build_internal_routes` instead.
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
///
/// IMPORTANT: Add internal routes to `build_internal_routes` instead.
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
///
/// IMPORTANT: Add internal routes to `build_internal_routes` instead.
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

/// This function builds the internal routes.
///
/// IMPORTANT: These routes are for internal use. They are unstable and might change without notice.
fn build_internal_routes() -> Router<AppStateData> {
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
            "/internal/datasets/{dataset_name}/datapoints",
            post(endpoints::datasets::insert_from_existing_datapoint_handler),
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
            "/internal/object_storage",
            get(endpoints::object_storage::get_object_handler),
        )
        .route(
            "/internal/inferences/bounds",
            get(endpoints::stored_inferences::v1::get_inference_bounds_handler),
        )
        .route(
            "/internal/inferences",
            get(endpoints::stored_inferences::v1::list_inferences_by_id_handler),
        )
}

/// This function builds the public routes for meta-observability (e.g. gateway health).
///
/// IMPORTANT: Add internal routes to `build_internal_routes` instead.
fn build_meta_observability_routes(metrics_handle: PrometheusHandle) -> Router<AppStateData> {
    Router::new()
        .route(
            "/metrics",
            get(move || std::future::ready(metrics_handle.render())),
        )
        .route("/status", get(endpoints::status::status_handler))
        .route("/health", get(endpoints::status::health_handler))
}
