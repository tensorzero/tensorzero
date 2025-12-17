//! E2E tests for the config snapshot endpoints.

use reqwest::{Client, StatusCode};
use serde_json::Value;
use std::collections::HashMap;
use std::time::Duration;
use tensorzero::{Client as TensorZeroClient, ClientExt, WriteConfigRequest};
use tensorzero_core::config::stored::StoredConfig;
use uuid::Uuid;

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

// ============================================================================
// Tests using Rust client (both embedded and HTTP gateway modes)
// ============================================================================

/// Test writing a config via the Rust client
async fn test_write_config_impl(client: TensorZeroClient) {
    let id = Uuid::now_v7();

    // Create a minimal config with a unique metric
    let config_toml = format!(
        r#"
[metrics.client_test_metric_{id}]
type = "boolean"
level = "inference"
optimize = "max"
"#
    );

    let stored_config: StoredConfig = toml::from_str(&config_toml).unwrap();

    let mut tags = HashMap::new();
    tags.insert("env".to_string(), "test".to_string());
    tags.insert("version".to_string(), "1.0".to_string());

    let mut extra_templates = HashMap::new();
    extra_templates.insert("test_template".to_string(), "Hello {{name}}!".to_string());

    let request = WriteConfigRequest {
        config: stored_config,
        extra_templates: extra_templates.clone(),
        tags: tags.clone(),
    };

    // Write the config
    let response = client.write_config(request).await;
    assert!(
        response.is_ok(),
        "write_config should succeed: {:?}",
        response.err()
    );

    let write_response = response.unwrap();
    assert!(!write_response.hash.is_empty(), "Hash should not be empty");

    // Wait for ClickHouse to commit
    tokio::time::sleep(Duration::from_millis(500)).await;

    // Verify we can retrieve the config by hash
    let get_response = client.get_config_snapshot(Some(&write_response.hash)).await;
    assert!(
        get_response.is_ok(),
        "get_config_snapshot should succeed: {:?}",
        get_response.err()
    );

    let config_snapshot = get_response.unwrap();
    assert_eq!(config_snapshot.hash, write_response.hash);
    assert_eq!(
        config_snapshot.extra_templates.get("test_template"),
        Some(&"Hello {{name}}!".to_string())
    );
    assert_eq!(config_snapshot.tags.get("env"), Some(&"test".to_string()));
    assert_eq!(
        config_snapshot.tags.get("version"),
        Some(&"1.0".to_string())
    );
}

tensorzero::make_gateway_test_functions!(test_write_config_impl);

/// Test tag merging when writing the same config twice via the Rust client
async fn test_write_config_tag_merging_impl(client: TensorZeroClient) {
    let id = Uuid::now_v7();

    // Create a config
    let config_toml = format!(
        r#"
[metrics.client_tag_merge_metric_{id}]
type = "boolean"
level = "inference"
optimize = "max"
"#
    );

    let stored_config: StoredConfig = toml::from_str(&config_toml).unwrap();

    // First write with initial tags
    let mut tags1 = HashMap::new();
    tags1.insert("key1".to_string(), "value1".to_string());
    tags1.insert("key2".to_string(), "original".to_string());

    let request1 = WriteConfigRequest {
        config: stored_config.clone(),
        extra_templates: HashMap::new(),
        tags: tags1,
    };

    let response1 = client.write_config(request1).await.unwrap();
    let hash = response1.hash.clone();

    // Wait for ClickHouse to commit
    tokio::time::sleep(Duration::from_millis(500)).await;

    // Second write with different tags (should merge)
    let mut tags2 = HashMap::new();
    tags2.insert("key2".to_string(), "updated".to_string()); // Update existing
    tags2.insert("key3".to_string(), "new".to_string()); // Add new

    let request2 = WriteConfigRequest {
        config: stored_config,
        extra_templates: HashMap::new(),
        tags: tags2,
    };

    let response2 = client.write_config(request2).await.unwrap();
    assert_eq!(response2.hash, hash, "Same config should produce same hash");

    // Wait for ClickHouse to commit
    tokio::time::sleep(Duration::from_millis(500)).await;

    // Verify tags were merged
    let config_snapshot = client.get_config_snapshot(Some(&hash)).await.unwrap();

    assert_eq!(
        config_snapshot.tags.get("key1"),
        Some(&"value1".to_string()),
        "key1 should be preserved from first write"
    );
    assert_eq!(
        config_snapshot.tags.get("key2"),
        Some(&"updated".to_string()),
        "key2 should be updated from second write"
    );
    assert_eq!(
        config_snapshot.tags.get("key3"),
        Some(&"new".to_string()),
        "key3 should be added from second write"
    );
}

tensorzero::make_gateway_test_functions!(test_write_config_tag_merging_impl);
