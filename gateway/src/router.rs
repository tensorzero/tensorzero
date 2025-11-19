//! Router construction and middleware for the TensorZero Gateway.
//!
//! This module builds the final Axum router with all layers (auth, tracing, metrics)
//! and defines authentication and version header middlewares.

use axum::{
    body::Body,
    extract::{DefaultBodyLimit, MatchedPath, Request, State},
    middleware::{self, Next},
    response::{IntoResponse, Response},
    Router,
};
use metrics_exporter_prometheus::PrometheusHandle;
use std::sync::Arc;
use tensorzero_auth::{key::TensorZeroApiKey, postgres::AuthResult};
use tensorzero_core::endpoints::RequestApiKeyExtension;
use tensorzero_core::{endpoints, utils::gateway::AppStateData};
use tensorzero_core::{
    error::{Error, ErrorDetails},
    observability::TracerWrapper,
};
use tower_http::{
    metrics::{in_flight_requests::InFlightRequestsCounter, InFlightRequestsLayer},
    trace::{DefaultOnFailure, TraceLayer},
};
use tracing::{Instrument, Level};

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
        .layer(
            TraceLayer::new_for_http()
                .on_failure(DefaultOnFailure::new().level(Level::DEBUG))
                .make_span_with(|request: &Request<Body>| {
                    // This is a copy of `DefaultMakeSpan` from `tower-http`.
                    // We invoke the `tracing` macro ourselves, so that the `target` is `gateway`,
                    // which will cause the span to get emitted even when `debug = false` in the gateway config.
                    // This ensures that warnings will have proper HTTP request information attached when logged to the console
                    // Entering and exiting the span itself does not produce any new console logs
                    tracing::info_span!(
                        "request",
                        method = %request.method(),
                        uri = %request.uri(),
                        version = ?request.version(),
                    )
                }),
        )
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
    mut request: Request,
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

        // Check cache first if available
        if let Some(cache) = &app_state.auth_cache {
            let cache_key = parsed_key.cache_key();
            if let Some(cached_result) = cache.get(&cache_key) {
                return match cached_result {
                    AuthResult::Success(key_info) => Ok((parsed_key, key_info)),
                    AuthResult::Disabled(disabled_at) => {
                        Err(Error::new(ErrorDetails::TensorZeroAuth {
                            message: format!("API key was disabled at: {disabled_at}"),
                        }))
                    }
                    AuthResult::MissingKey => Err(Error::new(ErrorDetails::TensorZeroAuth {
                        message: "Provided API key does not exist in the database".to_string(),
                    })),
                };
            }
        }

        // Cache miss or no cache - query database
        let postgres_key = match tensorzero_auth::postgres::check_key(&parsed_key, pool).await {
            Ok(key) => key,
            Err(e) => {
                return Err(Error::new(ErrorDetails::TensorZeroAuth {
                    message: format!("Failed to check API key: {e}"),
                }));
            }
        };

        // Store result in cache if available
        if let Some(cache) = &app_state.auth_cache {
            let cache_key = parsed_key.cache_key();
            cache.insert(cache_key, postgres_key.clone());
        }

        match postgres_key {
            AuthResult::Success(key_info) => Ok((parsed_key, key_info)),
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
        Ok((parsed_key, _key_info)) => {
            request.extensions_mut().insert(RequestApiKeyExtension {
                api_key: Arc::new(parsed_key),
            });
            next.run(request).await
        }
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
