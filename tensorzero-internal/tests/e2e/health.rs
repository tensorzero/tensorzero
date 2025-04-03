use reqwest::{Client, StatusCode};
use serde_json::Value;
use tensorzero_internal::endpoints::status::TENSORZERO_VERSION;

use crate::common::get_gateway_endpoint;

#[tokio::test]
async fn test_health_handler() {
    let client = Client::new();
    let response = client.get(get_gateway_endpoint("/health")).send().await;
    assert!(response.is_ok());
    let response = response.unwrap();
    assert_eq!(response.status(), StatusCode::OK);

    let version = response
        .headers()
        .get("x-tensorzero-gateway-version")
        .unwrap();
    assert_eq!(version.to_str().unwrap(), TENSORZERO_VERSION);

    let response_value = response.json::<Value>().await;
    assert!(response_value.is_ok());
    let response_value = response_value.unwrap();
    assert_eq!(response_value.get("gateway").unwrap(), "ok");
    assert_eq!(response_value.get("clickhouse").unwrap(), "ok");
}
