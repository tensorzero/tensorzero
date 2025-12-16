//! E2E tests for the config snapshot endpoints.

use reqwest::{Client, StatusCode};
use serde_json::Value;

use crate::common::get_gateway_endpoint;

#[tokio::test(flavor = "multi_thread")]
async fn test_get_live_config() {
    let http_client = Client::new();
    let url = get_gateway_endpoint("/internal/config");

    let resp = http_client.get(url).send().await.unwrap();
    let status = resp.status();
    let body = resp.text().await.unwrap();

    assert!(
        status.is_success(),
        "get_live_config request failed: status={status}, body={body}"
    );

    let response: Value = serde_json::from_str(&body).unwrap();

    // Verify response structure
    assert!(
        response.get("config").is_some(),
        "Response should have 'config' field"
    );
    assert!(
        response.get("hash").is_some(),
        "Response should have 'hash' field"
    );
    assert!(
        response.get("extra_templates").is_some(),
        "Response should have 'extra_templates' field"
    );

    // Verify hash is a non-empty string
    let hash = response["hash"].as_str().unwrap();
    assert!(!hash.is_empty(), "Hash should not be empty");
}

#[tokio::test(flavor = "multi_thread")]
async fn test_get_config_by_hash() {
    let http_client = Client::new();

    // First get the live config to obtain the current hash
    let live_url = get_gateway_endpoint("/internal/config");
    let live_resp = http_client.get(live_url).send().await.unwrap();
    assert!(
        live_resp.status().is_success(),
        "get_live_config request failed"
    );

    let live_config: Value = live_resp.json().await.unwrap();
    let hash = live_config["hash"].as_str().unwrap();

    // Now fetch the config by hash
    let url = get_gateway_endpoint(&format!("/internal/config/{hash}"));
    let resp = http_client.get(url).send().await.unwrap();
    let status = resp.status();
    let body = resp.text().await.unwrap();

    assert!(
        status.is_success(),
        "get_config_by_hash request failed: status={status}, body={body}"
    );

    let response: Value = serde_json::from_str(&body).unwrap();

    // Verify the returned hash matches
    assert_eq!(
        response["hash"].as_str().unwrap(),
        hash,
        "Returned hash should match requested hash"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn test_get_config_by_nonexistent_hash() {
    let http_client = Client::new();

    // Use a hash that definitely doesn't exist
    let nonexistent_hash = "12345678901234567890";
    let url = get_gateway_endpoint(&format!("/internal/config/{nonexistent_hash}"));

    let resp = http_client.get(url).send().await.unwrap();

    assert_eq!(
        resp.status(),
        StatusCode::NOT_FOUND,
        "Expected 404 for nonexistent hash"
    );
}
