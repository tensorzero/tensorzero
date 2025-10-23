use std::sync::Arc;

use axum::{
    extract::{DefaultBodyLimit, Request},
    middleware::Next,
    response::Response,
    routing::{delete, get, patch, post, put},
    Router,
};
use metrics_exporter_prometheus::PrometheusHandle;
use tensorzero_core::observability::{RouterExt as _, TracerWrapper};
use tensorzero_core::{
    endpoints::{self, openai_compatible::RouterExt},
    utils::gateway::AppStateData,
};
use tower_http::{
    metrics::{in_flight_requests::InFlightRequestsCounter, InFlightRequestsLayer},
    trace::{DefaultOnFailure, TraceLayer},
};
use tracing::Level;

/// Builds the final Axum router for the gateway,
/// which can be passed to `axum::serve` to start the server.
pub fn build_axum_router(
    base_path: &str,
    otel_tracer: Option<Arc<TracerWrapper>>,
    app_state: AppStateData,
    metrics_handle: PrometheusHandle,
) -> (Router, InFlightRequestsCounter) {
    let api_routes = build_api_routes(otel_tracer, metrics_handle);
    // The path was just `/` (or multiple slashes)
    let router = if base_path.is_empty() {
        Router::new().merge(api_routes)
    } else {
        Router::new().nest(base_path, api_routes)
    };

    let (in_flight_requests_layer, in_flight_requests_counter) = InFlightRequestsLayer::pair();

    let final_router = router
        .fallback(endpoints::fallback::handle_404)
        .layer(axum::middleware::from_fn(add_version_header))
        .layer(DefaultBodyLimit::max(100 * 1024 * 1024)) // increase the default body limit from 2MB to 100MB
        .layer(axum::middleware::from_fn(
            crate::warn_early_drop::warn_on_early_connection_drop,
        ))
        // Note - this is intentionally *not* used by our OTEL exporter (it creates a span without any `http.` or `otel.` fields)
        // This is only used to output request/response information to our logs
        // OTEL exporting is done by the `OtelAxumLayer` above, which is only enabled for certain routes (and includes much more information)
        // We log failed requests messages at 'DEBUG', since we already have our own error-logging code,
        .layer(TraceLayer::new_for_http().on_failure(DefaultOnFailure::new().level(Level::DEBUG)))
        // This should always be the very last layer in the stack, so that we start counting as soon as we begin processing a request
        .layer(in_flight_requests_layer)
        .with_state(app_state.clone());
    (final_router, in_flight_requests_counter)
}

async fn add_version_header(request: Request, next: Next) -> Response {
    #[cfg_attr(not(feature = "e2e_tests"), expect(unused_mut))]
    let mut version = axum::http::HeaderValue::from_static(endpoints::status::TENSORZERO_VERSION);

    #[cfg(feature = "e2e_tests")]
    {
        if request
            .headers()
            .contains_key("x-tensorzero-e2e-version-remove")
        {
            tracing::info!("Removing version header due to e2e header");
            return next.run(request).await;
        }
        if let Some(header_version) = request.headers().get("x-tensorzero-e2e-version-override") {
            tracing::info!("Overriding version header with e2e header: {header_version:?}");
            version = header_version.clone();
        }
    }

    let mut response = next.run(request).await;
    response
        .headers_mut()
        .insert("x-tensorzero-gateway-version", version);
    response
}

fn build_api_routes(
    otel_tracer: Option<Arc<TracerWrapper>>,
    metrics_handle: PrometheusHandle,
) -> Router<AppStateData> {
    Router::new()
        .merge(build_otel_enabled_routes())
        .apply_otel_http_trace_layer(otel_tracer)
        .merge(build_non_otel_enabled_routes(metrics_handle))
}

/// Defines routes that should have top-level OpenTelemetry HTTP spans created
/// All of these routes will have a span named `METHOD <ROUTE>` (e.g. `POST /batch_inference/{batch_id}`)
/// sent to OpenTelemetry
fn build_otel_enabled_routes() -> Router<AppStateData> {
    Router::new()
        .route("/inference", post(endpoints::inference::inference_handler))
        .route(
            "/batch_inference",
            post(endpoints::batch_inference::start_batch_inference_handler),
        )
        .route(
            "/batch_inference/{batch_id}",
            get(endpoints::batch_inference::poll_batch_inference_handler),
        )
        .route(
            "/batch_inference/{batch_id}/inference/{inference_id}",
            get(endpoints::batch_inference::poll_batch_inference_handler),
        )
        .register_openai_compatible_routes()
        .route("/feedback", post(endpoints::feedback::feedback_handler))
}

// Defines routes that should not have top-level OpenTelemetry HTTP spans created
// We use this for internal routes which we don't want to expose to users,
// or uninteresting routes like /health
fn build_non_otel_enabled_routes(metrics_handle: PrometheusHandle) -> Router<AppStateData> {
    Router::new()
        .route("/status", get(endpoints::status::status_handler))
        .route("/health", get(endpoints::status::health_handler))
        .route(
            "/datasets/{dataset_name}/datapoints",
            post(endpoints::datasets::create_datapoints_handler),
        )
        // TODO(#3459): Deprecated in #3721. Remove in a future release.
        .route(
            "/datasets/{dataset_name}/datapoints/bulk",
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
            patch(endpoints::datasets::v1::update_datapoints_handler),
        )
        .route(
            "/internal/datasets/{dataset_name}/datapoints",
            post(endpoints::datasets::insert_from_existing_datapoint_handler),
        )
        .route(
            "/internal/datasets/{dataset_name}/datapoints/{datapoint_id}",
            put(endpoints::datasets::update_datapoint_handler),
        )
        .route(
            "/internal/object_storage",
            get(endpoints::object_storage::get_object_handler),
        )
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
        .route(
            "/metrics",
            get(move || std::future::ready(metrics_handle.render())),
        )
        .route(
            "/experimental_optimization_workflow",
            post(endpoints::optimization::launch_optimization_workflow_handler),
        )
        .route(
            "/experimental_optimization/{job_handle}",
            get(endpoints::optimization::poll_optimization_handler),
        )
}
