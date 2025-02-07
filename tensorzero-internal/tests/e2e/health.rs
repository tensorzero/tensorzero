use reqwest::{Client, StatusCode};
use serde_json::Value;

use crate::common::get_gateway_endpoint;

#[tokio::test]
async fn test_health_handler() {
    let client = Client::new();
    let response = client.get(get_gateway_endpoint("/health")).send().await;
    assert!(response.is_ok());
    let response = response.unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let response_value = response.json::<Value>().await;
    assert!(response_value.is_ok());
    let response_value = response_value.unwrap();
    assert_eq!(response_value.get("gateway").unwrap(), "ok");
    assert_eq!(response_value.get("clickhouse").unwrap(), "ok");
}
