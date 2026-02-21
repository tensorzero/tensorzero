//! E2E tests for the UI config endpoints.

use reqwest::{Client, StatusCode};
use serde_json::Value;

use crate::common::get_gateway_endpoint;

#[tokio::test(flavor = "multi_thread")]
async fn test_get_ui_config_by_hash() {
    let http_client = Client::new();

    // First get the live UI config to obtain the current hash
    let live_url = get_gateway_endpoint("/internal/ui_config");
    let live_resp = http_client.get(live_url).send().await.unwrap();
    assert!(
        live_resp.status().is_success(),
        "GET /internal/ui_config request failed"
    );

    let live_config: Value = live_resp.json().await.unwrap();
    let hash = live_config["config_hash"]
        .as_str()
        .expect("Response should have `config_hash` field");

    // Now fetch the UI config by hash
    let url = get_gateway_endpoint(&format!("/internal/ui_config/{hash}"));
    let resp = http_client.get(url).send().await.unwrap();
    let status = resp.status();
    let body = resp.text().await.unwrap();

    assert!(
        status.is_success(),
        "GET /internal/ui_config/{{hash}} request failed: status={status}, body={body}"
    );

    let response: Value = serde_json::from_str(&body).unwrap();

    // Verify the returned hash matches
    assert_eq!(
        response["config_hash"].as_str().unwrap(),
        hash,
        "Returned config_hash should match requested hash"
    );

    // Verify response has expected fields
    assert!(
        response.get("functions").is_some(),
        "Response should have 'functions' field"
    );
    assert!(
        response.get("metrics").is_some(),
        "Response should have 'metrics' field"
    );
    assert!(
        response.get("tools").is_some(),
        "Response should have 'tools' field"
    );
    assert!(
        response.get("evaluations").is_some(),
        "Response should have 'evaluations' field"
    );
    assert!(
        response.get("model_names").is_some(),
        "Response should have 'model_names' field"
    );

    // Verify the snapshot has the same function, metric, tool, and evaluation keys as the live config.
    // We compare keys rather than full equality because serialization order of internal arrays
    // (e.g. fallback_variants) can differ between the live config and snapshot-loaded config.
    let live_fn_keys = sorted_keys(live_config["functions"].as_object().unwrap());
    let snap_fn_keys = sorted_keys(response["functions"].as_object().unwrap());
    assert_eq!(
        snap_fn_keys, live_fn_keys,
        "Snapshot should have the same function names as live config"
    );

    let live_metric_keys = sorted_keys(live_config["metrics"].as_object().unwrap());
    let snap_metric_keys = sorted_keys(response["metrics"].as_object().unwrap());
    assert_eq!(
        snap_metric_keys, live_metric_keys,
        "Snapshot should have the same metric names as live config"
    );

    let live_tool_keys = sorted_keys(live_config["tools"].as_object().unwrap());
    let snap_tool_keys = sorted_keys(response["tools"].as_object().unwrap());
    assert_eq!(
        snap_tool_keys, live_tool_keys,
        "Snapshot should have the same tool names as live config"
    );

    let live_eval_keys = sorted_keys(live_config["evaluations"].as_object().unwrap());
    let snap_eval_keys = sorted_keys(response["evaluations"].as_object().unwrap());
    assert_eq!(
        snap_eval_keys, live_eval_keys,
        "Snapshot should have the same evaluation names as live config"
    );
}

fn sorted_keys(obj: &serde_json::Map<String, Value>) -> Vec<&String> {
    let mut keys: Vec<&String> = obj.keys().collect();
    keys.sort();
    keys
}

#[tokio::test(flavor = "multi_thread")]
async fn test_get_ui_config_by_nonexistent_hash() {
    let http_client = Client::new();

    // Use a hash that definitely doesn't exist
    let nonexistent_hash = "12345678901234567890";
    let url = get_gateway_endpoint(&format!("/internal/ui_config/{nonexistent_hash}"));

    let resp = http_client.get(url).send().await.unwrap();

    assert_eq!(
        resp.status(),
        StatusCode::NOT_FOUND,
        "Expected 404 for nonexistent hash"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn test_get_ui_config_by_invalid_hash() {
    let http_client = Client::new();

    // Use a non-numeric string that won't parse as a SnapshotHash
    let invalid_hash = "not-a-number";
    let url = get_gateway_endpoint(&format!("/internal/ui_config/{invalid_hash}"));

    let resp = http_client.get(url).send().await.unwrap();

    assert_eq!(
        resp.status(),
        StatusCode::NOT_FOUND,
        "Expected 404 for invalid hash"
    );
}
