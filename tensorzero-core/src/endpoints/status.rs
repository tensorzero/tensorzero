use crate::{
    db::HealthCheckable,
    db::clickhouse::{ClickHouseConnectionInfo, clickhouse_client::ClickHouseClientType},
    db::postgres::PostgresConnectionInfo,
    db::valkey::ValkeyConnectionInfo,
    utils::gateway::{AppState, AppStateData},
};
use axum::extract::State;
use axum::http::StatusCode;
use axum::response::Json;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value, json};

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
        valkey_cache_connection_info,
        ..
    }): AppState,
) -> Result<Json<Value>, (StatusCode, Json<Value>)> {
    health_check_inner(
        &clickhouse_connection_info,
        &postgres_connection_info,
        &valkey_connection_info,
        &valkey_cache_connection_info,
    )
    .await
}

fn is_clickhouse_enabled(info: &ClickHouseConnectionInfo) -> bool {
    info.client_type() != ClickHouseClientType::Disabled
}

fn is_postgres_enabled(info: &PostgresConnectionInfo) -> bool {
    !matches!(info, PostgresConnectionInfo::Disabled)
}

fn is_valkey_enabled(info: &ValkeyConnectionInfo) -> bool {
    !matches!(info, ValkeyConnectionInfo::Disabled)
}

async fn health_check_inner(
    clickhouse_connection_info: &ClickHouseConnectionInfo,
    postgres_connection_info: &PostgresConnectionInfo,
    valkey_connection_info: &ValkeyConnectionInfo,
    valkey_cache_connection_info: &ValkeyConnectionInfo,
) -> Result<Json<Value>, (StatusCode, Json<Value>)> {
    let clickhouse_enabled = is_clickhouse_enabled(clickhouse_connection_info);
    let postgres_enabled = is_postgres_enabled(postgres_connection_info);
    let valkey_enabled = is_valkey_enabled(valkey_connection_info);
    let valkey_cache_enabled = is_valkey_enabled(valkey_cache_connection_info);

    let (clickhouse_result, postgres_result, valkey_result, valkey_cache_result) = tokio::join!(
        clickhouse_connection_info.health(),
        postgres_connection_info.health(),
        valkey_connection_info.health(),
        valkey_cache_connection_info.health(),
    );

    let all_ok = clickhouse_result.is_ok()
        && postgres_result.is_ok()
        && valkey_result.is_ok()
        && valkey_cache_result.is_ok();

    let mut response = Map::new();
    response.insert("gateway".to_string(), json!("ok"));

    if clickhouse_enabled {
        response.insert(
            "clickhouse".to_string(),
            json!(if clickhouse_result.is_ok() {
                "ok"
            } else {
                "error"
            }),
        );
    }
    if postgres_enabled {
        response.insert(
            "postgres".to_string(),
            json!(if postgres_result.is_ok() {
                "ok"
            } else {
                "error"
            }),
        );
    }
    if valkey_enabled {
        response.insert(
            "valkey".to_string(),
            json!(if valkey_result.is_ok() { "ok" } else { "error" }),
        );
    }
    if valkey_cache_enabled {
        response.insert(
            "valkey_cache".to_string(),
            json!(if valkey_cache_result.is_ok() {
                "ok"
            } else {
                "error"
            }),
        );
    }

    if all_ok {
        Ok(Json(Value::Object(response)))
    } else {
        Err((
            StatusCode::SERVICE_UNAVAILABLE,
            Json(Value::Object(response)),
        ))
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use crate::db::clickhouse::clickhouse_client::MockClickHouseClient;
    use crate::error::{Error, ErrorDetails};

    use super::*;

    fn mock_healthy_clickhouse() -> ClickHouseConnectionInfo {
        let mut mock = MockClickHouseClient::new();
        mock.expect_health().returning(|| Ok(()));
        mock.expect_client_type()
            .returning(|| ClickHouseClientType::Production);
        ClickHouseConnectionInfo::new_mock(Arc::new(mock))
    }

    fn mock_unhealthy_clickhouse() -> ClickHouseConnectionInfo {
        let mut mock = MockClickHouseClient::new();
        mock.expect_health().returning(|| {
            Err(Error::new(ErrorDetails::InternalError {
                message: "unhealthy".to_string(),
            }))
        });
        mock.expect_client_type()
            .returning(|| ClickHouseClientType::Production);
        ClickHouseConnectionInfo::new_mock(Arc::new(mock))
    }

    #[tokio::test]
    async fn test_health_handler_all_enabled_and_healthy() {
        let clickhouse = mock_healthy_clickhouse();
        let postgres = PostgresConnectionInfo::new_mock(true);
        let valkey = ValkeyConnectionInfo::Disabled;
        let valkey_cache = ValkeyConnectionInfo::Disabled;

        let response = health_check_inner(&clickhouse, &postgres, &valkey, &valkey_cache).await;
        assert!(response.is_ok(), "health check should pass");
        let response_value = response.unwrap();
        assert_eq!(response_value.get("gateway").unwrap(), "ok");
        assert_eq!(response_value.get("clickhouse").unwrap(), "ok");
        assert_eq!(response_value.get("postgres").unwrap(), "ok");
        assert!(
            response_value.get("valkey").is_none(),
            "disabled valkey should not appear"
        );
        assert!(
            response_value.get("valkey_cache").is_none(),
            "disabled valkey_cache should not appear"
        );
    }

    #[tokio::test]
    async fn test_health_handler_all_disabled() {
        let clickhouse = ClickHouseConnectionInfo::new_disabled();
        let postgres = PostgresConnectionInfo::Disabled;
        let valkey = ValkeyConnectionInfo::Disabled;
        let valkey_cache = ValkeyConnectionInfo::Disabled;

        let response = health_check_inner(&clickhouse, &postgres, &valkey, &valkey_cache).await;
        assert!(response.is_ok(), "health check should pass");
        let response_value = response.unwrap();
        assert_eq!(response_value.get("gateway").unwrap(), "ok");
        assert!(
            response_value.get("clickhouse").is_none(),
            "disabled clickhouse should not appear"
        );
        assert!(
            response_value.get("postgres").is_none(),
            "disabled postgres should not appear"
        );
        assert!(
            response_value.get("valkey").is_none(),
            "disabled valkey should not appear"
        );
        assert!(
            response_value.get("valkey_cache").is_none(),
            "disabled valkey_cache should not appear"
        );
    }

    #[tokio::test]
    async fn should_report_error_for_unhealthy_clickhouse() {
        let clickhouse = mock_unhealthy_clickhouse();
        let postgres = PostgresConnectionInfo::new_mock(true);
        let valkey = ValkeyConnectionInfo::Disabled;
        let valkey_cache = ValkeyConnectionInfo::Disabled;

        let response = health_check_inner(&clickhouse, &postgres, &valkey, &valkey_cache).await;
        assert!(response.is_err());
        let (status_code, error_json) = response.unwrap_err();
        assert_eq!(status_code, StatusCode::SERVICE_UNAVAILABLE);
        assert_eq!(error_json.get("gateway").unwrap(), "ok");
        assert_eq!(error_json.get("clickhouse").unwrap(), "error");
    }

    #[tokio::test]
    async fn should_report_error_for_unhealthy_postgres() {
        let clickhouse = mock_healthy_clickhouse();
        let postgres = PostgresConnectionInfo::new_mock(false);
        let valkey = ValkeyConnectionInfo::Disabled;
        let valkey_cache = ValkeyConnectionInfo::Disabled;

        let response = health_check_inner(&clickhouse, &postgres, &valkey, &valkey_cache).await;
        assert!(response.is_err());
        let (status_code, error_json) = response.unwrap_err();
        assert_eq!(status_code, StatusCode::SERVICE_UNAVAILABLE);
        assert_eq!(error_json.get("gateway").unwrap(), "ok");
        assert_eq!(error_json.get("postgres").unwrap(), "error");
    }
}
