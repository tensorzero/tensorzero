use crate::api_util::{AppState, AppStateData};
use axum::debug_handler;
use axum::extract::State;
use axum::http::StatusCode;
use axum::response::Json;
use serde_json::{json, Value};

/// A handler for a simple liveness check
#[debug_handler]
pub async fn status_handler() -> Json<Value> {
    Json(json!({ "status": "ok" }))
}

pub async fn health_handler(
    State(AppStateData {
        clickhouse_connection_info,
        ..
    }): AppState,
) -> Result<Json<Value>, (StatusCode, Json<Value>)> {
    if clickhouse_connection_info.health().await.is_err() {
        return Err((
            StatusCode::SERVICE_UNAVAILABLE,
            Json(json!({
                "api": "ok",
                "clickhouse": "error"
            })),
        ));
    }
    Ok(Json(json!({ "api": "ok", "clickhouse": "ok" })))
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use crate::{config_parser::Config, testing::get_unit_test_app_state_data};

    use super::*;

    #[tokio::test]
    async fn test_health_handler() {
        let config = Config {
            models: HashMap::new(),
            functions: HashMap::new(),
            metrics: None,
        };
        let app_state_data = get_unit_test_app_state_data(config.clone(), Some(true));
        let response = health_handler(State(app_state_data)).await;
        assert!(response.is_ok());
        let app_state_data = get_unit_test_app_state_data(config.clone(), Some(false));
        let response = health_handler(State(app_state_data)).await;
        assert!(response.is_err());
    }
}
