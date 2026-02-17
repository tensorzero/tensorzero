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

async fn health_check_inner(
    clickhouse_connection_info: &(dyn HealthCheckable + Sync),
    postgres_connection_info: &(dyn HealthCheckable + Sync),
    valkey_connection_info: &(dyn HealthCheckable + Sync),
    valkey_cache_connection_info: &(dyn HealthCheckable + Sync),
) -> Result<Json<Value>, (StatusCode, Json<Value>)> {
    let (clickhouse_result, postgres_result, valkey_result, valkey_cache_result) = join!(
        clickhouse_connection_info.health(),
        postgres_connection_info.health(),
        valkey_connection_info.health(),
        valkey_cache_connection_info.health(),
    );

    if clickhouse_result.is_ok()
        && postgres_result.is_ok()
        && valkey_result.is_ok()
        && valkey_cache_result.is_ok()
    {
        return Ok(Json(json!({
            "gateway": "ok",
            "clickhouse": "ok",
            "postgres": "ok",
            "valkey": "ok",
            "valkey_cache": "ok",
        })));
    }

    Err((
        StatusCode::SERVICE_UNAVAILABLE,
        Json(json!({
            "gateway": "ok",
            "clickhouse": if clickhouse_result.is_ok() { "ok" } else { "error" },
            "postgres": if postgres_result.is_ok() { "ok" } else { "error" },
            "valkey": if valkey_result.is_ok() { "ok" } else { "error" },
            "valkey_cache": if valkey_cache_result.is_ok() { "ok" } else { "error" },
        })),
    ))
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use crate::config::Config;
    use crate::db::clickhouse::ClickHouseConnectionInfo;
    use crate::db::clickhouse::clickhouse_client::FakeClickHouseClient;
    use crate::db::postgres::PostgresConnectionInfo;
    use crate::db::valkey::ValkeyConnectionInfo;

    use super::*;

    #[tokio::test]
    async fn test_health_handler() {
        let clickhouse = ClickHouseConnectionInfo::new_mock(Arc::new(FakeClickHouseClient::new(
            /* healthy= */ true,
        )));
        let postgres = PostgresConnectionInfo::new_mock(/* healthy= */ true);
        let valkey = ValkeyConnectionInfo::Disabled;

        let response = health_check_inner(&clickhouse, &postgres, &valkey, &valkey).await;
        assert!(response.is_ok(), "health check should pass");
        let response_value = response.unwrap();
        assert_eq!(response_value.get("gateway").unwrap(), "ok");
        assert_eq!(response_value.get("clickhouse").unwrap(), "ok");
        assert_eq!(response_value.get("postgres").unwrap(), "ok");
        assert_eq!(response_value.get("valkey").unwrap(), "ok");
        assert_eq!(response_value.get("valkey_cache").unwrap(), "ok");
    }

    #[tokio::test]
    async fn should_report_error_for_unhealthy_clickhouse() {
        let clickhouse = ClickHouseConnectionInfo::new_mock(Arc::new(FakeClickHouseClient::new(
            /* healthy= */ false,
        )));
        let postgres = PostgresConnectionInfo::new_mock(/* healthy= */ true);
        let valkey = ValkeyConnectionInfo::Disabled;

        let response = health_check_inner(&clickhouse, &postgres, &valkey, &valkey).await;
        assert!(response.is_err());
        let (status_code, error_json) = response.unwrap_err();
        assert_eq!(status_code, StatusCode::SERVICE_UNAVAILABLE);
        assert_eq!(error_json.get("gateway").unwrap(), "ok");
        assert_eq!(error_json.get("clickhouse").unwrap(), "error");
    }

    #[tokio::test]
    async fn should_report_error_for_unhealthy_postgres() {
        let clickhouse = ClickHouseConnectionInfo::new_mock(Arc::new(FakeClickHouseClient::new(
            /* healthy= */ true,
        )));
        let postgres = PostgresConnectionInfo::new_mock(/* healthy= */ false);
        let valkey = ValkeyConnectionInfo::Disabled;

        let response = health_check_inner(&clickhouse, &postgres, &valkey, &valkey).await;
        assert!(response.is_err());
        let (status_code, error_json) = response.unwrap_err();
        assert_eq!(status_code, StatusCode::SERVICE_UNAVAILABLE);
        assert_eq!(error_json.get("gateway").unwrap(), "ok");
        assert_eq!(error_json.get("postgres").unwrap(), "error");
    }

    #[test]
    fn test_status_response() {
        let config = Config::default();
        let response = StatusResponse {
            status: "ok".to_string(),
            version: TENSORZERO_VERSION.to_string(),
            config_hash: config.hash.to_string(),
        };
        assert_eq!(response.version, TENSORZERO_VERSION);
        assert!(!response.config_hash.is_empty());
    }
}
