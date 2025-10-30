use std::sync::Arc;

use axum::{
    extract::{DefaultBodyLimit, MatchedPath, Request, State},
    middleware::{self, Next},
    response::{IntoResponse, Response},
    routing::{delete, get, patch, post, put},
    Router,
};
use metrics_exporter_prometheus::PrometheusHandle;
use tensorzero_auth::{key::TensorZeroApiKey, postgres::AuthResult};
use tensorzero_core::{endpoints, utils::gateway::AppStateData};
use tensorzero_core::{
    endpoints::openai_compatible::build_openai_compatible_routes,
    observability::{RouterExt as _, TracerWrapper},
};
use tensorzero_core::{
    error::{Error, ErrorDetails},
    observability::OtelEnabledRoutes,
};
use tower_http::{
    metrics::{in_flight_requests::InFlightRequestsCounter, InFlightRequestsLayer},
    trace::{DefaultOnFailure, TraceLayer},
};
use tracing::{Instrument, Level};

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
    let mut router = if base_path.is_empty() {
        Router::new().merge(api_routes)
    } else {
        Router::new().nest(base_path, api_routes)
    };

    let (in_flight_requests_layer, in_flight_requests_counter) = InFlightRequestsLayer::pair();

    router = router.fallback(endpoints::fallback::handle_404);

    if app_state.config.gateway.auth.enabled {
        router = router.layer(middleware::from_fn_with_state(
            app_state.clone(),
            tensorzero_auth_middleware,
        ));
    }
    // Everything added from this point onwards does *NOT* have authentication applied - that is,
    // it wraps the authentication middleware
    // increase the default body limit from 2MB to 100MB
    let final_router = router
        .layer(axum::middleware::from_fn(add_version_header))
        .layer(DefaultBodyLimit::max(100 * 1024 * 1024))
        .layer(axum::middleware::from_fn(
            tensorzero_core::observability::warn_early_drop::warn_on_early_connection_drop,
        ))
        // We want tracing/status to run for all requests, regardless of whether or not authentication succeeds
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

/// Routes that should not require authentication
/// We apply authentication to all routes *except* these ones, to make it difficult
/// to accidentally skip running authentication on a route, especially if we later refactor
/// how we build up our router.
const UNAUTHENTICATED_ROUTES: &[&str] = &["/status", "/health"];

#[axum::debug_middleware]
async fn tensorzero_auth_middleware(
    State(app_state): State<AppStateData>,
    request: Request,
    next: Next,
) -> Response {
    let route = request
        .extensions()
        .get::<MatchedPath>()
        .map(MatchedPath::as_str);
    if let Some(route) = route {
        if UNAUTHENTICATED_ROUTES.contains(&route) {
            return next.run(request).await;
        }
    }
    let headers = request.headers();
    // This block holds all of the actual authentication logic.
    // We use `.instrument` on this future, so that we don't include the '.next.run(request)' inside
    // of our `tensorzero_auth` OpenTelemetry span.
    let do_auth = async {
        let Some(auth_header) = headers.get(http::header::AUTHORIZATION) else {
            return Err(Error::new(ErrorDetails::TensorZeroAuth {
                message: "Authorization header is required".to_string(),
            }));
        };
        let auth_header_value = auth_header.to_str().map_err(|e| {
            Error::new(ErrorDetails::TensorZeroAuth {
                message: format!("Invalid authorization header: {e}"),
            })
        })?;
        let raw_api_key = auth_header_value.strip_prefix("Bearer ").ok_or_else(|| {
            Error::new(ErrorDetails::TensorZeroAuth {
                message: "Authorization header must start with 'Bearer '".to_string(),
            })
        })?;

        let parsed_key = match TensorZeroApiKey::parse(raw_api_key) {
            Ok(key) => key,
            Err(e) => {
                return Err(Error::new(ErrorDetails::TensorZeroAuth {
                    message: format!("Invalid API key: {e}"),
                }))
            }
        };
        let Some(pool) = app_state.postgres_connection_info.get_alpha_pool() else {
            return Err(Error::new(ErrorDetails::TensorZeroAuth {
                message: "PostgreSQL connection is disabled".to_string(),
            }));
        };
        let postgres_key = match tensorzero_auth::postgres::check_key(&parsed_key, pool).await {
            Ok(key) => key,
            Err(e) => {
                return Err(Error::new(ErrorDetails::TensorZeroAuth {
                    message: format!("Failed to check API key: {e}"),
                }));
            }
        };
        match postgres_key {
            AuthResult::Success(key_info) => Ok(key_info),
            AuthResult::Disabled(disabled_at) => Err(Error::new(ErrorDetails::TensorZeroAuth {
                message: format!("API key was disabled at: {disabled_at}"),
            })),
            AuthResult::MissingKey => Err(Error::new(ErrorDetails::TensorZeroAuth {
                message: "Provided API key does not exist in the database".to_string(),
            })),
        }
    }
    .instrument(tracing::trace_span!(
        "tensorzero_auth",
        otel.name = "tensorzero_auth"
    ));

    match do_auth.await {
        Ok(_key_info) => next.run(request).await,
        Err(e) => e.into_response(),
    }
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
        .route(
            "/variant_sampling_probabilities",
            get(endpoints::variant_probabilities::get_variant_sampling_probabilities_handler),
        )
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
            patch(endpoints::datasets::v1::update_datapoints_handler)
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
            "/v1/datasets/get_datapoints",
            post(endpoints::datasets::v1::get_datapoints_handler),
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
        .route("/status", get(endpoints::status::status_handler))
        .route("/health", get(endpoints::status::health_handler))
}
