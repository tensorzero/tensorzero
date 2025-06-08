use serde_json::Value;
use axum::http::StatusCode;
use axum::middleware::Next;
use axum::response::{Response, IntoResponse};
use axum::extract::{State, Request};
use axum::body::{to_bytes, Body};
use std::sync::{Arc, RwLock};
use std::collections::HashMap;

pub type APIConfig = HashMap<String, String>;

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
    api_keys: Arc<RwLock<HashMap<String, APIConfig>>>
}

impl Auth {
    pub fn new(api_keys: HashMap<String, APIConfig>) -> Self {
        Self { api_keys: Arc::new(RwLock::new(api_keys)) }
    }

    pub fn update_api_keys(&self, api_key: &str, api_config: APIConfig) {
        let mut api_keys = self.api_keys.write().unwrap();
        api_keys.insert(api_key.to_string(), api_config);
    }

    pub fn delete_api_key(&self, api_key: &str) {
        let mut api_keys = self.api_keys.write().unwrap();
        api_keys.remove(api_key);
    }

    pub fn validate_api_key(&self, api_key: &str) -> Result<APIConfig, StatusCode> {
        let api_keys = self.api_keys.read().unwrap();
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
        None => return Err(auth_error_response(
            StatusCode::UNAUTHORIZED, 
            "missing_authorization", 
            "Missing authorization header"
        )),
    };

    let api_config = auth.validate_api_key(key);
    if api_config.is_err() {
        return Err(auth_error_response(
            StatusCode::UNAUTHORIZED, 
            "invalid_api_key", 
            "Invalid API key"
        ));
    }

    // Parse the JSON body and replace the "model" field with the value from api_config if present
    let mut val: Value = match serde_json::from_slice(&bytes) {
        Ok(v) => v,
        Err(_) => return Err(auth_error_response(
            StatusCode::BAD_REQUEST, 
            "invalid_request_body", 
            "Invalid request body"
        )),
    };

    let model = match val.get("model").and_then(|v| v.as_str()) {
        Some(m) => m,
        None => return Err(auth_error_response(
            StatusCode::BAD_REQUEST, 
            "invalid_request_body", 
            "Missing model name in request body"
        )),
    };

    let api_config = api_config.unwrap();
    let model_value = match api_config.get(model) {
        Some(v) => v,
        None => return Err(auth_error_response(
            StatusCode::NOT_FOUND, 
            "model_not_found", 
            "Model not found in API key"
        )),
    };

    // Replace the "model" field in the request body with the value from api_config
    val["model"] = format!("tensorzero::model_name::{}", model_value.clone()).into();

    // Re-serialize the modified JSON body for the downstream handler
    let bytes = match serde_json::to_vec(&val) {
        Ok(b) => b,
        Err(_) => return Err(auth_error_response(
            StatusCode::INTERNAL_SERVER_ERROR, 
            "serialization_error", 
            "Failed to serialize modified request body"
        )),
    };

    let request = Request::from_parts(parts, Body::from(bytes));

    Ok(next.run(request).await)
}