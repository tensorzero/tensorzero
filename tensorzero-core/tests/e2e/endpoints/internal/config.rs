//! E2E tests for the config snapshot endpoints.

use reqwest::{Client, StatusCode};
use serde_json::Value;
use std::collections::HashMap;
use std::time::Duration;
use tensorzero_core::config::UninitializedConfig;
use tensorzero_core::endpoints::internal::config::{
    GetConfigResponse, WriteConfigRequest, WriteConfigResponse,
};
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

#[tokio::test(flavor = "multi_thread")]
async fn test_write_config_endpoint() {
    let http_client = Client::new();
    let id = Uuid::now_v7();

    let config_toml = format!(
        r#"
[metrics.client_test_metric_{id}]
type = "boolean"
level = "inference"
optimize = "max"
"#
    );

    let stored_config: UninitializedConfig = toml::from_str(&config_toml).unwrap();

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

    let url = get_gateway_endpoint("/internal/config");
    let response = http_client.post(url).json(&request).send().await.unwrap();
    assert!(
        response.status().is_success(),
        "POST /internal/config should succeed for a valid config snapshot: status={}",
        response.status()
    );
    let write_response: WriteConfigResponse = response
        .json()
        .await
        .expect("Response should parse as WriteConfigResponse correctly");
    assert!(
        !write_response.hash.is_empty(),
        "write_config endpoint should return a non-empty hash"
    );

    tokio::time::sleep(Duration::from_millis(500)).await;

    let get_url = get_gateway_endpoint(&format!("/internal/config/{}", write_response.hash));
    let get_response = http_client.get(get_url).send().await.unwrap();
    assert!(
        get_response.status().is_success(),
        "GET /internal/config/{{hash}} should succeed for a freshly written hash: status={}",
        get_response.status()
    );
    let config_snapshot: GetConfigResponse = get_response
        .json()
        .await
        .expect("Response should parse as GetConfigResponse correctly");
    assert_eq!(
        config_snapshot.hash, write_response.hash,
        "Fetched config hash should match the hash returned by POST /internal/config"
    );
    assert_eq!(
        config_snapshot.extra_templates.get("test_template"),
        Some(&"Hello {{name}}!".to_string()),
        "Fetched config snapshot should preserve submitted extra templates"
    );
    assert_eq!(
        config_snapshot.tags.get("env"),
        Some(&"test".to_string()),
        "Fetched config snapshot should preserve submitted `env` tag"
    );
    assert_eq!(
        config_snapshot.tags.get("version"),
        Some(&"1.0".to_string()),
        "Fetched config snapshot should preserve submitted `version` tag"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn test_write_config_tag_merging_endpoint() {
    let http_client = Client::new();
    let id = Uuid::now_v7();

    let config_toml = format!(
        r#"
[metrics.client_tag_merge_metric_{id}]
type = "boolean"
level = "inference"
optimize = "max"
"#
    );

    let stored_config: UninitializedConfig = toml::from_str(&config_toml).unwrap();

    let mut tags1 = HashMap::new();
    tags1.insert("key1".to_string(), "value1".to_string());
    tags1.insert("key2".to_string(), "original".to_string());

    let request1 = WriteConfigRequest {
        config: stored_config.clone(),
        extra_templates: HashMap::new(),
        tags: tags1,
    };

    let write_url = get_gateway_endpoint("/internal/config");
    let response1 = http_client
        .post(write_url.clone())
        .json(&request1)
        .send()
        .await
        .unwrap();
    assert!(
        response1.status().is_success(),
        "First POST /internal/config should succeed: status={}",
        response1.status()
    );
    let write_response1: WriteConfigResponse = response1
        .json()
        .await
        .expect("Response should parse as WriteConfigResponse correctly");
    let hash = write_response1.hash.clone();

    tokio::time::sleep(Duration::from_millis(500)).await;

    let mut tags2 = HashMap::new();
    tags2.insert("key2".to_string(), "updated".to_string());
    tags2.insert("key3".to_string(), "new".to_string());

    let request2 = WriteConfigRequest {
        config: stored_config,
        extra_templates: HashMap::new(),
        tags: tags2,
    };

    let response2 = http_client
        .post(write_url)
        .json(&request2)
        .send()
        .await
        .unwrap();
    assert!(
        response2.status().is_success(),
        "Second POST /internal/config should succeed for duplicate config content: status={}",
        response2.status()
    );
    let write_response2: WriteConfigResponse = response2
        .json()
        .await
        .expect("Response should parse as WriteConfigResponse correctly");

    assert_eq!(
        write_response2.hash, hash,
        "Writing identical config content should return the same hash"
    );

    tokio::time::sleep(Duration::from_millis(500)).await;

    let get_url = get_gateway_endpoint(&format!("/internal/config/{hash}"));
    let get_response = http_client.get(get_url).send().await.unwrap();
    assert!(
        get_response.status().is_success(),
        "GET /internal/config/{{hash}} should succeed after tag merging writes: status={}",
        get_response.status()
    );

    let config_snapshot: GetConfigResponse = get_response
        .json()
        .await
        .expect("Response should parse as GetConfigResponse correctly");

    assert_eq!(
        config_snapshot.tags.get("key1"),
        Some(&"value1".to_string()),
        "Tag merge should preserve keys from the first write that were not overwritten"
    );
    assert_eq!(
        config_snapshot.tags.get("key2"),
        Some(&"updated".to_string()),
        "Tag merge should overwrite key2 with the value from the second write"
    );
    assert_eq!(
        config_snapshot.tags.get("key3"),
        Some(&"new".to_string()),
        "Tag merge should add brand new keys from the second write"
    );
}
