use std::sync::Arc;

use axum::{
    Json,
    extract::{MatchedPath, Request, State},
    http::{self, StatusCode},
    middleware::Next,
    response::{IntoResponse, Response},
};
use moka::sync::Cache;
use serde_json::{Value, json};
use tracing::{Instrument, field::Empty};

use crate::{
    key::{TensorZeroApiKey, TensorZeroAuthError},
    postgres::{AuthResult, KeyInfo},
};

/// Builds an error response body in either TensorZero or OpenAI format.
///
/// When `openai_format` is true, returns `{"error": {"message": "..."}}`.
/// When `openai_format` is false, returns `{"error": "..."}`.
///
/// If `error_json` is provided, includes structured error details.
fn build_error_response_body(
    message: &str,
    openai_format: bool,
    error_json: Option<Value>,
) -> Value {
    let mut body = if openai_format {
        json!({"error": {"message": message}})
    } else {
        json!({"error": message})
    };
    if let Some(error_json) = error_json {
        if openai_format {
            body["error"]["error_json"] = error_json.clone(); // DEPRECATED (#5821 / 2026.4+)
            body["error"]["tensorzero_error_json"] = error_json;
        } else {
            body["error_json"] = error_json;
        }
    }
    body
}

fn is_openai_compatible_route(matched_path: Option<&MatchedPath>, base_path: Option<&str>) -> bool {
    matched_path.is_some_and(|p| match base_path {
        Some(base) => p.as_str().starts_with(&format!("{base}/openai/v1")),
        None => p.as_str().starts_with("/openai/v1"),
    })
}

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
    /// how we build up our router.
    pub unauthenticated_routes: &'static [&'static str],
    pub auth_cache: Option<Cache<String, AuthResult>>,
    pub pool: Option<sqlx::PgPool>,
    pub error_json: bool,
    /// Optional base path prefix for all routes (e.g., "/custom/prefix")
    pub base_path: Option<String>,
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

    let auth_span = tracing::info_span!(
        "tensorzero_auth",
        otel.name = "tensorzero_auth",
        key.public_id = Empty,
        key.organization = Empty,
        key.workspace = Empty,
    );
    // This block holds all of the actual authentication logic.
    // We use `.instrument` on this future, so that we don't include the '.next.run(request)' inside
    // of our `tensorzero_auth` OpenTelemetry span.
    let do_auth = async {
        let Some(auth_header) = headers.get(http::header::AUTHORIZATION) else {
            return Err(TensorZeroAuthError::Middleware {
                message: "Authorization header is required".to_string(),
                key_info: None,
            });
        };
        let auth_header_value =
            auth_header
                .to_str()
                .map_err(|e| TensorZeroAuthError::Middleware {
                    message: format!("Invalid authorization header: {e}"),
                    key_info: None,
                })?;
        let raw_api_key = auth_header_value.strip_prefix("Bearer ").ok_or_else(|| {
            TensorZeroAuthError::Middleware {
                message: "Authorization header must start with 'Bearer '".to_string(),
                key_info: None,
            }
        })?;

        let parsed_key = TensorZeroApiKey::parse(raw_api_key)?;

        // Record the public ID immediately, in case we fail to look up the key in the database/cache
        let span = tracing::Span::current();
        span.record("key.public_id", parsed_key.get_public_id());

        let Some(pool) = &state.pool else {
            return Err(TensorZeroAuthError::Middleware {
                message: "PostgreSQL connection is disabled".to_string(),
                key_info: None,
            });
        };

        // Check cache first if available
        if let Some(cache) = &state.auth_cache {
            let cache_key = parsed_key.cache_key();
            if let Some(cached_result) = cache.get(&cache_key) {
                return match cached_result {
                    AuthResult::Success(key_info) => Ok((parsed_key, key_info)),
                    AuthResult::Disabled(disabled_at, key_info) => {
                        Err(TensorZeroAuthError::Middleware {
                            message: format!("API key was disabled at: {disabled_at}"),
                            key_info: Some(Box::new(key_info)),
                        })
                    }
                    AuthResult::MissingKey => Err(TensorZeroAuthError::Middleware {
                        message: "Provided API key does not exist in the database".to_string(),
                        key_info: None,
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
            AuthResult::Disabled(disabled_at, key_info) => Err(TensorZeroAuthError::Middleware {
                message: format!("API key was disabled at: {disabled_at}"),
                key_info: Some(Box::new(key_info)),
            }),
            AuthResult::MissingKey => Err(TensorZeroAuthError::Middleware {
                message: "Provided API key does not exist in the database".to_string(),
                key_info: None,
            }),
        }
    }
    .instrument(auth_span.clone());

    match do_auth.await {
        Ok((parsed_key, key_info)) => {
            auth_span.record("key.organization", &key_info.organization);
            auth_span.record("key.workspace", &key_info.workspace);
            request.extensions_mut().insert(RequestApiKeyExtension {
                api_key: Arc::new(parsed_key),
                key_info: key_info.clone(),
            });
            next.run(request).await
        }
        Err(e) => {
            if let TensorZeroAuthError::Middleware {
                key_info: Some(key_info),
                ..
            } = &e
            {
                auth_span.record("key.organization", &key_info.organization);
                auth_span.record("key.workspace", &key_info.workspace);
            }
            let message = format!("TensorZero authentication error: {e}");
            let matched_path = request.extensions().get::<MatchedPath>();
            let is_openai_format =
                is_openai_compatible_route(matched_path, state.base_path.as_deref());
            let error_json = state.error_json.then(|| json!(e.to_string()));
            let body = build_error_response_body(&message, is_openai_format, error_json);
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
    pub key_info: KeyInfo,
}
