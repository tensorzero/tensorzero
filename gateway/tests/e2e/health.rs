use reqwest::{Client, StatusCode};
use serde_json::Value;

// TODO (#74): make this endpoint configurable with main.rs
const HEALTH_URL: &str = "http://localhost:3000/health";

#[tokio::test]
async fn test_health_handler() {
    let client = Client::new();
    let response = client.get(HEALTH_URL).send().await;
    assert!(response.is_ok());
    let response = response.unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let response_value = response.json::<Value>().await;
    assert!(response_value.is_ok());
    let response_value = response_value.unwrap();
    assert_eq!(response_value.get("gateway").unwrap(), "ok");
    assert_eq!(response_value.get("clickhouse").unwrap(), "ok");
}
