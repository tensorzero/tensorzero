use axum::body::{to_bytes, Body};
use axum::extract::{Request, State};
use axum::http::StatusCode;
use axum::middleware::Next;
use axum::response::{IntoResponse, Response};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;
use std::sync::{Arc, RwLock};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApiKeyMetadata {
    pub endpoint_id: String,
    pub model_id: String,
    pub project_id: String,
}

pub type APIConfig = HashMap<String, ApiKeyMetadata>;

// Common error response helper
fn auth_error_response(status: StatusCode, error_type: &str, message: &str) -> Response {
    let body = serde_json::json!({
        "type": error_type,
        "error": message
    });
    (status, axum::Json(body)).into_response()
}

#[derive(Clone)]
pub struct Auth {
    api_keys: Arc<RwLock<HashMap<String, APIConfig>>>,
}

impl Auth {
    pub fn new(api_keys: HashMap<String, APIConfig>) -> Self {
        Self {
            api_keys: Arc::new(RwLock::new(api_keys)),
        }
    }

    pub fn update_api_keys(&self, api_key: &str, api_config: APIConfig) {
        // In practice, a poisoned RwLock indicates a panic in another thread while holding the lock.
        // This is a catastrophic failure that should not be recovered from.
        #[expect(clippy::expect_used)]
        let mut api_keys = self.api_keys.write().expect("RwLock poisoned");
        api_keys.insert(api_key.to_string(), api_config);
    }

    pub fn delete_api_key(&self, api_key: &str) {
        #[expect(clippy::expect_used)]
        let mut api_keys = self.api_keys.write().expect("RwLock poisoned");
        api_keys.remove(api_key);
    }

    pub fn validate_api_key(&self, api_key: &str) -> Result<APIConfig, StatusCode> {
        #[expect(clippy::expect_used)]
        let api_keys = self.api_keys.read().expect("RwLock poisoned");
        if let Some(api_config) = api_keys.get(api_key) {
            return Ok(api_config.clone());
        }
        Err(StatusCode::UNAUTHORIZED)
    }
}

pub async fn require_api_key(
    State(auth): State<Auth>,
    request: Request,
    next: Next,
) -> Result<Response, Response> {
    let (parts, body) = request.into_parts();
    let bytes = to_bytes(body, 1024 * 1024).await.unwrap_or_default();

    let key = parts
        .headers
        .get("authorization")
        .and_then(|v| v.to_str().ok());

    let key = match key {
        Some(key) => {
            // Strip "Bearer " prefix if present (case-insensitive)
            let key = key.trim();
            key.strip_prefix("Bearer ").unwrap_or(key)
        }
        None => {
            return Err(auth_error_response(
                StatusCode::UNAUTHORIZED,
                "missing_authorization",
                "Missing authorization header",
            ))
        }
    };

    let api_config = auth.validate_api_key(key);
    if api_config.is_err() {
        return Err(auth_error_response(
            StatusCode::UNAUTHORIZED,
            "invalid_api_key",
            "Invalid API key",
        ));
    }

    // Parse the JSON body to validate and extract model name
    let val: Value = match serde_json::from_slice(&bytes) {
        Ok(v) => v,
        Err(_) => {
            return Err(auth_error_response(
                StatusCode::BAD_REQUEST,
                "invalid_request_body",
                "Invalid request body",
            ))
        }
    };

    let model = match val.get("model").and_then(|v| v.as_str()) {
        Some(m) => m,
        None => {
            return Err(auth_error_response(
                StatusCode::BAD_REQUEST,
                "invalid_request_body",
                "Missing model name in request body",
            ))
        }
    };

    // We already checked that api_config is Ok in the if statement above
    #[expect(clippy::unwrap_used)]
    let api_config = api_config.unwrap();
    let metadata = match api_config.get(model) {
        Some(v) => v,
        None => {
            return Err(auth_error_response(
                StatusCode::NOT_FOUND,
                "model_not_found",
                "Model not found in API key",
            ))
        }
    };

    // Create request without modifying the body
    let mut request = Request::from_parts(parts, Body::from(bytes));

    // Add the model name as a custom header for downstream handlers
    if let Ok(header_value) = model.parse() {
        request
            .headers_mut()
            .insert("x-tensorzero-model-name", header_value);
    }

    // Add metadata headers for observability
    if let Ok(header_value) = metadata.project_id.parse() {
        request
            .headers_mut()
            .insert("x-tensorzero-project-id", header_value);
    }
    if let Ok(header_value) = metadata.endpoint_id.parse() {
        request
            .headers_mut()
            .insert("x-tensorzero-endpoint-id", header_value);
    }
    if let Ok(header_value) = metadata.model_id.parse() {
        request
            .headers_mut()
            .insert("x-tensorzero-model-id", header_value);
    }

    Ok(next.run(request).await)
}
