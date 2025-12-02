use std::sync::Arc;

use axum::{
    Json,
    extract::{MatchedPath, Request, State},
    http::{self, StatusCode},
    middleware::Next,
    response::{IntoResponse, Response},
};
use moka::sync::Cache;
use serde_json::json;
use tracing::Instrument;

use crate::{
    key::{TensorZeroApiKey, TensorZeroAuthError},
    postgres::AuthResult,
};

#[derive(Clone)]
pub struct TensorzeroAuthMiddlewareState(Arc<TensorzeroAuthMiddlewareStateInner>);

impl TensorzeroAuthMiddlewareState {
    pub fn new(inner: TensorzeroAuthMiddlewareStateInner) -> Self {
        Self(Arc::new(inner))
    }
}

pub struct TensorzeroAuthMiddlewareStateInner {
    /// Routes that should not require authentication
    /// We apply authentication to all routes *except* these ones, to make it difficult
    /// to accidentally skip running authentication on a route, especially if we later refactor
    /// how we build up our router.I
    pub unauthenticated_routes: &'static [&'static str],
    pub auth_cache: Option<Cache<String, AuthResult>>,
    pub pool: Option<sqlx::PgPool>,
    pub error_json: bool,
}

#[axum::debug_middleware]
pub async fn tensorzero_auth_middleware(
    State(state): State<TensorzeroAuthMiddlewareState>,
    mut request: Request,
    next: Next,
) -> Response {
    let state = &*state.0;
    let route = request
        .extensions()
        .get::<MatchedPath>()
        .map(MatchedPath::as_str);
    if let Some(route) = route
        && state.unauthenticated_routes.contains(&route)
    {
        return next.run(request).await;
    }
    let headers = request.headers();
    // This block holds all of the actual authentication logic.
    // We use `.instrument` on this future, so that we don't include the '.next.run(request)' inside
    // of our `tensorzero_auth` OpenTelemetry span.
    let do_auth = async {
        let Some(auth_header) = headers.get(http::header::AUTHORIZATION) else {
            return Err(TensorZeroAuthError::Middleware {
                message: "Authorization header is required".to_string(),
            });
        };
        let auth_header_value =
            auth_header
                .to_str()
                .map_err(|e| TensorZeroAuthError::Middleware {
                    message: format!("Invalid authorization header: {e}"),
                })?;
        let raw_api_key = auth_header_value.strip_prefix("Bearer ").ok_or_else(|| {
            TensorZeroAuthError::Middleware {
                message: "Authorization header must start with 'Bearer '".to_string(),
            }
        })?;

        let parsed_key = TensorZeroApiKey::parse(raw_api_key)?;
        let Some(pool) = &state.pool else {
            return Err(TensorZeroAuthError::Middleware {
                message: "PostgreSQL connection is disabled".to_string(),
            });
        };

        // Check cache first if available
        if let Some(cache) = &state.auth_cache {
            let cache_key = parsed_key.cache_key();
            if let Some(cached_result) = cache.get(&cache_key) {
                return match cached_result {
                    AuthResult::Success(key_info) => Ok((parsed_key, key_info)),
                    AuthResult::Disabled(disabled_at) => Err(TensorZeroAuthError::Middleware {
                        message: format!("API key was disabled at: {disabled_at}"),
                    }),
                    AuthResult::MissingKey => Err(TensorZeroAuthError::Middleware {
                        message: "Provided API key does not exist in the database".to_string(),
                    }),
                };
            }
        }

        // Cache miss or no cache - query database
        let postgres_key = crate::postgres::check_key(&parsed_key, pool).await?;

        // Store result in cache if available
        if let Some(cache) = &state.auth_cache {
            let cache_key = parsed_key.cache_key();
            cache.insert(cache_key, postgres_key.clone());
        }

        match postgres_key {
            AuthResult::Success(key_info) => Ok((parsed_key, key_info)),
            AuthResult::Disabled(disabled_at) => Err(TensorZeroAuthError::Middleware {
                message: format!("API key was disabled at: {disabled_at}"),
            }),
            AuthResult::MissingKey => Err(TensorZeroAuthError::Middleware {
                message: "Provided API key does not exist in the database".to_string(),
            }),
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
        Err(e) => {
            let message = e.to_string();
            let mut body = json!({
                "error": format!("TensorZero authentication error: {message}"),
            });
            if state.error_json {
                body["error_json"] = json!(e.to_string());
            }
            let mut response = (StatusCode::UNAUTHORIZED, Json(body)).into_response();
            // Attach the error to the response, so that we can set a nice message in our
            // `apply_otel_http_trace_layer` middleware
            response.extensions_mut().insert(e);
            response
        }
    }
}

// Our auth middleware stores this in the request extensions when auth is enabled,
// *and* the API key is valid.
// This is used to get access to the API key (e.g. for rate-limiting), *not* to require that a route is authenticated -
// the `auth` middleware itself rejects requests without API keys
#[derive(Debug, Clone)]
pub struct RequestApiKeyExtension {
    pub api_key: Arc<TensorZeroApiKey>,
}
