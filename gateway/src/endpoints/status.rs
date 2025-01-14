use crate::gateway_util::{AppState, AppStateData};
use axum::debug_handler;
use axum::extract::State;
use axum::http::StatusCode;
use axum::response::Json;
use serde_json::{json, Value};

const TENSORZERO_VERSION: &str = "2025.01.2";

/// A handler for a simple liveness check
#[debug_handler]
pub async fn status_handler() -> Json<Value> {
    Json(json!({ "status": "ok", "version": TENSORZERO_VERSION }))
}

/// A handler for a health check that includes availability of related services (for now, ClickHouse)
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
                "gateway": "ok",
                "clickhouse": "error"
            })),
        ));
    }
    Ok(Json(json!({ "gateway": "ok", "clickhouse": "ok" })))
}

#[cfg(test)]
mod tests {

    use crate::config_parser::Config;
    use crate::testing::get_unit_test_app_state_data;

    use super::*;

    #[tokio::test]
    async fn test_health_handler() {
        let config = Box::leak(Box::new(Config::default()));
        let app_state_data = get_unit_test_app_state_data(config, true);
        let response = health_handler(State(app_state_data)).await;
        assert!(response.is_ok());
        let response_value = response.unwrap();
        assert_eq!(response_value.get("gateway").unwrap(), "ok");
        assert_eq!(response_value.get("clickhouse").unwrap(), "ok");

        let app_state_data = get_unit_test_app_state_data(config, false);
        let response = health_handler(State(app_state_data)).await;
        assert!(response.is_err());
        let (status_code, error_json) = response.unwrap_err();
        assert_eq!(status_code, StatusCode::SERVICE_UNAVAILABLE);
        assert_eq!(error_json.get("gateway").unwrap(), "ok");
        assert_eq!(error_json.get("clickhouse").unwrap(), "error");
    }

    #[tokio::test]
    async fn test_status_handler() {
        let response = status_handler().await;
        assert_eq!(response.get("version").unwrap(), TENSORZERO_VERSION);
    }
}
