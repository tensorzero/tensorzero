use reqwest::Client;
use serde_json::Value;

use crate::common::get_gateway_endpoint;

#[tokio::test]
async fn test_404_get() {
    let client = Client::new();

    // Make GET request to non-existent endpoint
    let response = client
        .get(get_gateway_endpoint("/non/existent/path"))
        .send()
        .await
        .unwrap();

    assert_eq!(response.status().as_u16(), 404);

    let response_json = response.json::<Value>().await.unwrap();
    let error_msg = response_json.get("error").and_then(Value::as_str).unwrap();

    assert!(error_msg.contains("GET"));
    assert!(error_msg.contains("/non/existent/path"));
}

#[tokio::test]
async fn test_404_post() {
    let client = Client::new();

    // Make POST request to non-existent endpoint with JSON body
    let json_body = serde_json::json!({
        "message": "Hello world",
        "number": 42,
        "active": true
    });

    let response = client
        .post(get_gateway_endpoint("/non/existent/path"))
        .json(&json_body)
        .send()
        .await
        .unwrap();

    assert_eq!(response.status().as_u16(), 404);

    let response_json = response.json::<Value>().await.unwrap();
    let error_msg = response_json.get("error").and_then(Value::as_str).unwrap();

    assert!(error_msg.contains("POST"));
    assert!(error_msg.contains("/non/existent/path"));
}
