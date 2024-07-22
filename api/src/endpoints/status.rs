use axum::debug_handler;
use axum::response::Json;
use serde_json::json;

/// A handler for a simple liveness check
#[debug_handler]
pub async fn status_handler() -> Json<serde_json::Value> {
    Json(json!({ "status": "ok" }))
}
