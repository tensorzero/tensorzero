use crate::{
    db::HealthCheckable,
    utils::gateway::{AppState, AppStateData},
};
use axum::extract::State;
use axum::http::StatusCode;
use axum::response::Json;
use futures::join;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

pub const TENSORZERO_VERSION: &str = env!("CARGO_PKG_VERSION");

#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Debug, Deserialize, Serialize)]
#[cfg_attr(feature = "ts-bindings", ts(export))]
pub struct StatusResponse {
    pub status: String,
    pub version: String,
    pub config_hash: String,
}

/// A handler for a simple liveness check
#[expect(clippy::unused_async)]
pub async fn status_handler(State(app_state): AppState) -> Json<StatusResponse> {
    Json(StatusResponse {
        status: "ok".to_string(),
        version: TENSORZERO_VERSION.to_string(),
        config_hash: app_state.config.hash.to_string(),
    })
}

/// A handler for a health check that includes availability of related services
pub async fn health_handler(
    State(AppStateData {
        clickhouse_connection_info,
        postgres_connection_info,
        valkey_connection_info,
        ..
    }): AppState,
) -> Result<Json<Value>, (StatusCode, Json<Value>)> {
    let (clickhouse_result, postgres_result, valkey_result) = join!(
        clickhouse_connection_info.health(),
        postgres_connection_info.health(),
        valkey_connection_info.health(),
    );

    if clickhouse_result.is_ok() && postgres_result.is_ok() && valkey_result.is_ok() {
        return Ok(Json(json!({
            "gateway": "ok",
            "clickhouse": "ok",
            "postgres": "ok",
            "valkey": "ok",
        })));
    }

    Err((
        StatusCode::SERVICE_UNAVAILABLE,
        Json(json!({
            "gateway": "ok",
            "clickhouse": if clickhouse_result.is_ok() { "ok" } else { "error" },
            "postgres": if postgres_result.is_ok() { "ok" } else { "error" },
            "valkey": if valkey_result.is_ok() { "ok" } else { "error" },
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
        assert!(response.is_ok(), "health check should pass");
        let response_value = response.unwrap();
        assert_eq!(response_value.get("gateway").unwrap(), "ok");
        assert_eq!(response_value.get("clickhouse").unwrap(), "ok");
        assert_eq!(response_value.get("postgres").unwrap(), "ok");
        assert_eq!(response_value.get("valkey").unwrap(), "ok");
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
        let config = Arc::new(Config::default());
        let gateway_handle = get_unit_test_gateway_handle_with_options(
            config.clone(),
            GatewayHandleTestOptions {
                clickhouse_client: Arc::new(FakeClickHouseClient::new(true)),
                postgres_healthy: true,
            },
        );
        let response = status_handler(State(gateway_handle.app_state.clone())).await;
        assert_eq!(response.version, TENSORZERO_VERSION);
        assert!(!response.config_hash.is_empty());
    }
}
