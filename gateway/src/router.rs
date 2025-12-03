//! Router construction and middleware for the TensorZero Gateway.
//!
//! This module builds the final Axum router with all layers (auth, tracing, metrics)
//! and defines authentication and version header middlewares.

use axum::{
    extract::{DefaultBodyLimit, Request},
    middleware::{self, Next},
    response::Response,
    Router,
};
use metrics_exporter_prometheus::PrometheusHandle;
use std::sync::Arc;
use tensorzero_auth::middleware::TensorzeroAuthMiddlewareStateInner;
use tensorzero_core::endpoints::TensorzeroAuthMiddlewareState;
use tensorzero_core::observability::TracerWrapper;
use tensorzero_core::{endpoints, utils::gateway::AppStateData};
use tower_http::metrics::{in_flight_requests::InFlightRequestsCounter, InFlightRequestsLayer};

use crate::routes::build_api_routes;

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
        let state = TensorzeroAuthMiddlewareState::new(TensorzeroAuthMiddlewareStateInner {
            unauthenticated_routes: UNAUTHENTICATED_ROUTES,
            auth_cache: app_state.auth_cache.clone(),
            pool: app_state.postgres_connection_info.get_alpha_pool().cloned(),
            error_json: app_state.config.gateway.unstable_error_json,
        });
        router = router.layer(middleware::from_fn_with_state(
            state,
            tensorzero_auth::middleware::tensorzero_auth_middleware,
        ));
    }
    // Everything added from this point onwards does *NOT* have authentication applied - that is,
    // it wraps the authentication middleware
    // increase the default body limit from 2MB to 100MB
    let final_router = router
        .layer(axum::middleware::from_fn(add_version_header))
        .layer(DefaultBodyLimit::max(100 * 1024 * 1024))
        .layer(axum::middleware::from_fn(
            tensorzero_core::observability::request_logging::request_logging_middleware,
        ))
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
