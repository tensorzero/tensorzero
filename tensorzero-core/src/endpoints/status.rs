use crate::{
    db::HealthCheckable,
    utils::gateway::{AppState, AppStateData},
};
use axum::debug_handler;
use axum::extract::State;
use axum::http::StatusCode;
use axum::response::Json;
use futures::join;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

pub const TENSORZERO_VERSION: &str = env!("CARGO_PKG_VERSION");

/// A handler for a simple liveness check
#[debug_handler]
pub async fn status_handler() -> Json<StatusResponse> {
    Json(StatusResponse {
        status: "ok".to_string(),
        version: TENSORZERO_VERSION.to_string(),
    })
}

#[derive(Debug, Deserialize, Serialize)]
pub struct StatusResponse {
    pub status: String,
    pub version: String,
}

/// A handler for a health check that includes availability of related services (for now, ClickHouse)
pub async fn health_handler(
    State(AppStateData {
        clickhouse_connection_info,
        postgres_connection_info,
        ..
    }): AppState,
) -> Result<Json<Value>, (StatusCode, Json<Value>)> {
    let (clickhouse_result, postgres_result) = join!(
        clickhouse_connection_info.health(),
        postgres_connection_info.health(),
    );

    if clickhouse_result.is_ok() && postgres_result.is_ok() {
        return Ok(Json(json!({
            "gateway": "ok",
            "clickhouse": "ok",
            "postgres": "ok",
        })));
    }

    Err((
        StatusCode::SERVICE_UNAVAILABLE,
        Json(json!({
            "gateway": "ok",
            "clickhouse": if clickhouse_result.is_ok() { "ok" } else { "error" },
            "postgres": if postgres_result.is_ok() { "ok" } else { "error" },
        })),
    ))
}

#[cfg(test)]
mod tests {

    use std::sync::Arc;

    use crate::config::Config;
    use crate::db::clickhouse::clickhouse_client::FakeClickHouseClient;
    use crate::testing::get_unit_test_gateway_handle_with_options;
    use crate::utils::gateway::GatewayHandleTestOptions;

    use super::*;

    #[tokio::test]
    async fn test_health_handler() {
        let config = Arc::new(Config::default());
        let fake = FakeClickHouseClient::new(/* healthy= */ true);
        let gateway_handle = get_unit_test_gateway_handle_with_options(
            config.clone(),
            GatewayHandleTestOptions {
                clickhouse_client: Arc::new(fake),
                postgres_healthy: true,
            },
        );
        let response = health_handler(State(gateway_handle.app_state.clone())).await;
        assert!(response.is_ok());
        let response_value = response.unwrap();
        assert_eq!(response_value.get("gateway").unwrap(), "ok");
        assert_eq!(response_value.get("clickhouse").unwrap(), "ok");
        assert_eq!(response_value.get("postgres").unwrap(), "ok");
    }

    #[tokio::test]
    async fn should_report_error_for_unhealthy_clickhouse() {
        let config = Arc::new(Config::default());
        let fake = FakeClickHouseClient::new(/* healthy= */ false);
        let gateway_handle = get_unit_test_gateway_handle_with_options(
            config,
            GatewayHandleTestOptions {
                clickhouse_client: Arc::new(fake),
                postgres_healthy: true,
            },
        );
        let response = health_handler(State(gateway_handle.app_state.clone())).await;
        assert!(response.is_err());
        let (status_code, error_json) = response.unwrap_err();
        assert_eq!(status_code, StatusCode::SERVICE_UNAVAILABLE);
        assert_eq!(error_json.get("gateway").unwrap(), "ok");
        assert_eq!(error_json.get("clickhouse").unwrap(), "error");
    }

    #[tokio::test]
    async fn should_report_error_for_unhealthy_postgres() {
        let config = Arc::new(Config::default());
        let fake = FakeClickHouseClient::new(/* healthy= */ true);
        let gateway_handle = get_unit_test_gateway_handle_with_options(
            config,
            GatewayHandleTestOptions {
                clickhouse_client: Arc::new(fake),
                postgres_healthy: false,
            },
        );
        let response = health_handler(State(gateway_handle.app_state.clone())).await;
        assert!(response.is_err());
        let (status_code, error_json) = response.unwrap_err();
        assert_eq!(status_code, StatusCode::SERVICE_UNAVAILABLE);
        assert_eq!(error_json.get("gateway").unwrap(), "ok");
        assert_eq!(error_json.get("postgres").unwrap(), "error");
    }

    #[tokio::test]
    async fn test_status_handler() {
        let response = status_handler().await;
        assert_eq!(response.version, TENSORZERO_VERSION);
    }
}
