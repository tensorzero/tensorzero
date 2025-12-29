//! E2E tests for `include_raw_usage` relay passthrough.
//!
//! Tests that when an edge gateway requests raw_usage, it is passed through
//! to the downstream gateway and returned in the response.
//!
//! NOTE: These tests require a relay configuration to be set up.
//! The relay config should point the edge gateway to a downstream gateway.

use futures::StreamExt;
use reqwest::Client;
use reqwest_eventsource::{Event, RequestBuilderExt};
use serde_json::{Value, json};
use uuid::Uuid;

/// Test that relay passthrough works for include_raw_usage (non-streaming)
///
/// This test is marked as ignored because it requires:
/// 1. A running downstream gateway at localhost:3000
/// 2. An edge gateway at localhost:3001 that relays to the downstream gateway
/// 3. The downstream gateway to have the `weather_helper` function configured
///
/// To run this test:
/// 1. Start a downstream gateway: `cargo run-e2e`
/// 2. Start an edge gateway with relay config pointing to localhost:3000
/// 3. Run: `cargo test e2e_test_raw_usage_relay_non_streaming -- --ignored`
#[tokio::test]
#[ignore = "Requires running edge and downstream gateways"]
async fn e2e_test_raw_usage_relay_non_streaming() {
    let edge_gateway_url = std::env::var("TENSORZERO_EDGE_GATEWAY_URL")
        .unwrap_or_else(|_| "http://localhost:3001".to_string());

    let episode_id = Uuid::now_v7();
    let random_suffix = Uuid::now_v7();

    // Make request through edge gateway with include_raw_usage
    let payload = json!({
        "function_name": "weather_helper",
        "variant_name": "openai",
        "episode_id": episode_id,
        "input": {
            "system": {"assistant_name": "RelayBot"},
            "messages": [
                {
                    "role": "user",
                    "content": format!("Relay test message: {random_suffix}")
                }
            ]
        },
        "stream": false,
        "include_raw_usage": true
    });

    let response = Client::new()
        .post(format!("{edge_gateway_url}/inference"))
        .json(&payload)
        .send()
        .await
        .unwrap();

    assert!(
        response.status().is_success(),
        "Relay request should succeed"
    );

    let response_json: Value = response.json().await.unwrap();

    // Check raw_usage exists and is an array
    let raw_usage = response_json
        .get("raw_usage")
        .expect("Relay response should include raw_usage when requested");
    assert!(raw_usage.is_array(), "raw_usage should be an array");

    let raw_usage_array = raw_usage.as_array().unwrap();
    assert!(
        !raw_usage_array.is_empty(),
        "raw_usage should have entries from downstream gateway"
    );

    // Verify structure of entries
    for entry in raw_usage_array {
        assert!(
            entry.get("model_inference_id").is_some(),
            "raw_usage entry should have model_inference_id"
        );
        assert!(
            entry.get("provider_type").is_some(),
            "raw_usage entry should have provider_type"
        );
        assert!(
            entry.get("api_type").is_some(),
            "raw_usage entry should have api_type"
        );
    }
}

/// Test that relay does NOT return raw_usage when not requested
#[tokio::test]
#[ignore = "Requires running edge and downstream gateways"]
async fn e2e_test_raw_usage_relay_not_requested() {
    let edge_gateway_url = std::env::var("TENSORZERO_EDGE_GATEWAY_URL")
        .unwrap_or_else(|_| "http://localhost:3001".to_string());

    let episode_id = Uuid::now_v7();
    let random_suffix = Uuid::now_v7();

    // Make request through edge gateway WITHOUT include_raw_usage
    let payload = json!({
        "function_name": "weather_helper",
        "variant_name": "openai",
        "episode_id": episode_id,
        "input": {
            "system": {"assistant_name": "RelayBot"},
            "messages": [
                {
                    "role": "user",
                    "content": format!("Relay test no raw_usage: {random_suffix}")
                }
            ]
        },
        "stream": false,
        "include_raw_usage": false
    });

    let response = Client::new()
        .post(format!("{edge_gateway_url}/inference"))
        .json(&payload)
        .send()
        .await
        .unwrap();

    assert!(
        response.status().is_success(),
        "Relay request should succeed"
    );

    let response_json: Value = response.json().await.unwrap();

    // raw_usage should NOT be present when not requested
    assert!(
        response_json.get("raw_usage").is_none(),
        "raw_usage should NOT be present when not requested"
    );
}

/// Test relay streaming with include_raw_usage
#[tokio::test]
#[ignore = "Requires running edge and downstream gateways"]
async fn e2e_test_raw_usage_relay_streaming() {
    let edge_gateway_url = std::env::var("TENSORZERO_EDGE_GATEWAY_URL")
        .unwrap_or_else(|_| "http://localhost:3001".to_string());

    let episode_id = Uuid::now_v7();
    let random_suffix = Uuid::now_v7();

    let payload = json!({
        "function_name": "weather_helper",
        "variant_name": "openai",
        "episode_id": episode_id,
        "input": {
            "system": {"assistant_name": "RelayBot"},
            "messages": [
                {
                    "role": "user",
                    "content": format!("Relay streaming test: {random_suffix}")
                }
            ]
        },
        "stream": true,
        "include_raw_usage": true
    });

    let mut chunks = Client::new()
        .post(format!("{edge_gateway_url}/inference"))
        .json(&payload)
        .eventsource()
        .unwrap();

    let mut found_raw_usage = false;

    while let Some(chunk) = chunks.next().await {
        let chunk = chunk.unwrap();
        let Event::Message(chunk) = chunk else {
            continue;
        };
        if chunk.data == "[DONE]" {
            break;
        }

        let chunk_json: Value = serde_json::from_str(&chunk.data).unwrap();
        if chunk_json.get("raw_usage").is_some() {
            found_raw_usage = true;
        }
    }

    assert!(
        found_raw_usage,
        "Streaming relay response should include raw_usage when requested"
    );
}
