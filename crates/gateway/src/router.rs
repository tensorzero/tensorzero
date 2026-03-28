//! Router construction and middleware for the TensorZero Gateway.
//!
//! This module builds the final Axum router with all layers (auth, tracing, metrics)
//! and defines authentication and version header middlewares.

use crate::routes::build_api_routes;
use axum::{
    Router,
    extract::{DefaultBodyLimit, Request, State},
    middleware::{self, Next},
    response::{IntoResponse, Response},
};
use http_body_util::{BodyStream, StreamBody};
use metrics_exporter_prometheus::PrometheusHandle;
use std::sync::Arc;
use tensorzero_auth::middleware::TensorzeroAuthMiddlewareStateInner;
use tensorzero_core::observability::TracerWrapper;
use tensorzero_core::observability::request_logging::InFlightRequestsData;
use tensorzero_core::{endpoints, utils::gateway::AppStateData};
use tensorzero_core::{
    endpoints::TensorzeroAuthMiddlewareState,
    error::{Error, ErrorDetails},
};
use tokio_stream::{StreamExt, wrappers::ReceiverStream};
use tokio_util::sync::CancellationToken;
use tower_http::decompression::RequestDecompressionLayer;
use tracing::Instrument;

/// Builds the final Axum router for the gateway,
/// which can be passed to `axum::serve` to start the server.
pub fn build_axum_router(
    base_path: &str,
    otel_tracer: Option<Arc<TracerWrapper>>,
    app_state: AppStateData,
    metrics_handle: PrometheusHandle,
    shutdown_token: CancellationToken,
) -> (Router, InFlightRequestsData) {
    let api_routes = build_api_routes(otel_tracer, metrics_handle);
    // The path was just `/` (or multiple slashes)
    let mut router = if base_path.is_empty() {
        Router::new().merge(api_routes)
    } else {
        Router::new().nest(base_path, api_routes)
    };

    // Serve the MCP endpoint on the same port at `/mcp`
    let mcp_router = tensorzero_mcp::build_mcp_router(Arc::new(app_state.clone()), shutdown_token);
    router = router.nest_service("/mcp", mcp_router);

    router = router.fallback(endpoints::fallback::handle_404);

    if app_state.config.gateway.auth.enabled {
        let state = TensorzeroAuthMiddlewareState::new(TensorzeroAuthMiddlewareStateInner {
            unauthenticated_routes: UNAUTHENTICATED_ROUTES,
            auth_cache: app_state.auth_cache.clone(),
            pool: app_state.postgres_connection_info.get_pool().cloned(),
            error_json: app_state.config.gateway.unstable_error_json,
            base_path: (!base_path.is_empty()).then(|| base_path.to_string()),
        });
        router = router.layer(middleware::from_fn_with_state(
            state,
            tensorzero_auth::middleware::tensorzero_auth_middleware,
        ));
    }

    let in_flight_requests_data =
        tensorzero_core::observability::request_logging::InFlightRequestsData::new();
    // Everything added from this point onwards does *NOT* have authentication applied - that is,
    // it wraps the authentication middleware
    // increase the default body limit from 2MB to 100MB
    let final_router = router
        .layer(axum::middleware::from_fn(add_version_header))
        .layer(DefaultBodyLimit::max(100 * 1024 * 1024))
        // Accept encoded requests and transparently decompress them.
        // Supported encodings: gzip, br, zstd.
        .layer(RequestDecompressionLayer::new())
        .layer(axum::middleware::from_fn_with_state(
            app_state.clone(),
            possibly_prevent_request_cancellation,
        ))
        // This should always be the very last layer in the stack, so that we start counting as soon as we begin processing a request
        .layer(axum::middleware::from_fn_with_state(
            in_flight_requests_data.clone(),
            tensorzero_core::observability::request_logging::request_logging_middleware,
        ))
        .with_state(app_state.clone());
    (final_router, in_flight_requests_data)
}

/// Routes that should not require authentication
/// We apply authentication to all routes *except* these ones, to make it difficult
/// to accidentally skip running authentication on a route, especially if we later refactor
/// how we build up our router.
const UNAUTHENTICATED_ROUTES: &[&str] = &["/status", "/health", "/internal/autopilot/status"];

/// This middleware spawns all non-'safe' (e.g. non GET/HEAD/OPTIONS) requests on a new tokio task,
/// and awaits the task completion. This prevents the handler future from getting dropped if
/// the client closes the connection early.
///
/// This ensures that POST requests (e.g. POST /inference) continue to run even if the client closes the connection early.
/// so that we perform all of our desired side effects (writing to the database, handling rate limiting, etc.)
///
/// Note that it's inherently impossible to guarantee that we *start* processing a request,
/// since the client might close the connection before sending the complete HTTP request
/// (e.g. it doesn't write all of the headers or body).
/// We guarantee that if we *start* processing a request, we will run our handler/stream logic to completion,
/// including all our side effects (writing to the database, handling rate limiting, etc.).
async fn possibly_prevent_request_cancellation(
    State(state): State<AppStateData>,
    request: Request,
    next: Next,
) -> Response {
    if request.method().is_safe() {
        return next.run(request).await;
    }

    let deferred_tasks = state.deferred_tasks.clone();

    // Capture the current tracing span so that the spawned task inherits the span context.
    // This is critical for the `OverheadTimingLayer`, which relies on span parent-child
    // relationships to detect external spans and subtract their time from the overhead metric.
    let span = tracing::Span::current();
    let task = async move {
        let resp = next.run(request).await;
        let (parts, body) = resp.into_parts();
        let mut stream = BodyStream::new(body);
        // We spawn a separate tokio task to drive the stream to completion
        // To match the behavior of the original stream, we only buffer a single frame
        // (meaning that the `send` call will suspend until axum processes the previous frame from the `StreamBody`
        // on the other end).
        // This ensures that the stream will always be driven to completion, even if the client closes the connection early.
        let (send, recv) = tokio::sync::mpsc::channel(1);
        let inner_span = tracing::Span::current();
        deferred_tasks.spawn(
            async move {
                while let Some(chunk) = stream.next().await {
                    // We deliberately ignore errors here - even if the receiver is dropped
                    // (due to the client closing the connection early), we still want to
                    // drive the original stream to completion, which will finish executing
                    // all of the tensorzero stream logic (e.g. writing the collected chunks
                    // to the database, handling rate limiting, etc.)
                    let _ = send.send(chunk).await;
                }
            }
            .instrument(inner_span),
        );
        let spawned_stream = axum::body::Body::new(StreamBody::new(ReceiverStream::new(recv)));
        Response::from_parts(parts, spawned_stream)
    }
    .instrument(span);
    // Our 'task' includes the call to 'next.run(request)', which ensures that the request handler
    // will still be executed to completion, even if the client closes the connection early.
    // The 'task' future includes additional handling for streaming responses
    match state.deferred_tasks.spawn(task).await {
        Ok(resp) => resp,
        Err(e) => Error::new(ErrorDetails::InternalError {
            message: format!("Failed to join task in possibly_prevent_request_cancellation: {e:?}"),
        })
        .into_response(),
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
