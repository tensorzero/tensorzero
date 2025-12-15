//! E2E tests for the `/internal/action` endpoint.
//!
//! These tests verify that the action endpoint correctly executes inference
//! using historical config snapshots loaded from ClickHouse.

use reqwest::{Client, StatusCode};
use std::collections::HashMap;
use std::time::Duration;
use tensorzero_core::config::snapshot::{ConfigSnapshot, SnapshotHash};
use tensorzero_core::config::write_config_snapshot;
use tensorzero_core::db::clickhouse::test_helpers::get_clickhouse;
use uuid::Uuid;

use crate::common::get_gateway_endpoint;

/// Test that `/internal/action` can execute inference using a historical config
/// that has a function not present in the running gateway config.
#[tokio::test(flavor = "multi_thread")]
async fn test_action_with_historical_config() {
    let clickhouse = get_clickhouse().await;
    let id = Uuid::now_v7();

    // Create a historical config with a unique function that doesn't exist in the running gateway.
    // Functions require variants (not a direct `model` field).
    let historical_config = format!(
        r#"
[models.action_test_model_{id}]
routing = ["provider"]

[models.action_test_model_{id}.providers.provider]
type = "dummy"
model_name = "test"

[functions.historical_only_func_{id}]
type = "chat"

[functions.historical_only_func_{id}.variants.baseline]
type = "chat_completion"
model = "action_test_model_{id}"
"#
    );

    let snapshot =
        ConfigSnapshot::new_from_toml_string(&historical_config, HashMap::new()).unwrap();
    let snapshot_hash = snapshot.hash.clone();

    // Write the historical snapshot to ClickHouse
    write_config_snapshot(&clickhouse, snapshot).await.unwrap();
    tokio::time::sleep(Duration::from_millis(500)).await;

    // Call /internal/action with the historical snapshot hash
    let http_client = Client::new();
    let url = get_gateway_endpoint("/internal/action");

    let response = http_client
        .post(url)
        .json(&serde_json::json!({
            "snapshot_hash": snapshot_hash.to_string(),
            "type": "inference",
            "function_name": format!("historical_only_func_{id}"),
            "input": {
                "messages": [{"role": "user", "content": [{"type": "text", "text": "test"}]}]
            }
        }))
        .send()
        .await
        .unwrap();

    let status = response.status();
    let body = response.text().await.unwrap();

    assert!(
        status.is_success(),
        "Action with historical config should succeed: status={status}, body={body}"
    );
}

/// Test that `/internal/action` returns 404 for a non-existent snapshot hash.
#[tokio::test(flavor = "multi_thread")]
async fn test_action_nonexistent_snapshot_hash() {
    let http_client = Client::new();
    let url = get_gateway_endpoint("/internal/action");

    // Use a properly formatted hash that simply doesn't exist in the database
    let nonexistent_hash = SnapshotHash::new_test();

    let response = http_client
        .post(url)
        .json(&serde_json::json!({
            "snapshot_hash": nonexistent_hash.to_string(),
            "type": "inference",
            "function_name": "any_function",
            "input": {
                "messages": [{"role": "user", "content": [{"type": "text", "text": "test"}]}]
            }
        }))
        .send()
        .await
        .unwrap();

    assert_eq!(
        response.status(),
        StatusCode::NOT_FOUND,
        "Should return 404 for non-existent snapshot hash"
    );
}

/// Test that `/internal/action` rejects streaming requests.
#[tokio::test(flavor = "multi_thread")]
async fn test_action_streaming_rejected() {
    let clickhouse = get_clickhouse().await;
    let id = Uuid::now_v7();

    // Create a minimal historical config with proper variant structure
    let historical_config = format!(
        r#"
[models.action_test_model_{id}]
routing = ["provider"]

[models.action_test_model_{id}.providers.provider]
type = "dummy"
model_name = "test"

[functions.stream_test_func_{id}]
type = "chat"

[functions.stream_test_func_{id}.variants.baseline]
type = "chat_completion"
model = "action_test_model_{id}"
"#
    );

    let snapshot =
        ConfigSnapshot::new_from_toml_string(&historical_config, HashMap::new()).unwrap();
    let snapshot_hash = snapshot.hash.clone();

    write_config_snapshot(&clickhouse, snapshot).await.unwrap();
    tokio::time::sleep(Duration::from_millis(500)).await;

    let http_client = Client::new();
    let url = get_gateway_endpoint("/internal/action");

    let response = http_client
        .post(url)
        .json(&serde_json::json!({
            "snapshot_hash": snapshot_hash.to_string(),
            "type": "inference",
            "function_name": format!("stream_test_func_{id}"),
            "stream": true,
            "input": {
                "messages": [{"role": "user", "content": [{"type": "text", "text": "test"}]}]
            }
        }))
        .send()
        .await
        .unwrap();

    assert_eq!(
        response.status(),
        StatusCode::BAD_REQUEST,
        "Streaming requests should be rejected with 400 Bad Request"
    );
}
