//! E2E tests for `include_raw_usage` cache behavior.
//!
//! Tests that cache hits are correctly excluded from raw_usage.

use futures::StreamExt;
use reqwest::{Client, StatusCode};
use reqwest_eventsource::{Event, RequestBuilderExt};
use serde_json::{Value, json};
use uuid::Uuid;

use crate::common::get_gateway_endpoint;

/// Makes an inference request and returns the raw_usage array length
async fn make_request_and_get_raw_usage_count(
    function_name: &str,
    variant_name: &str,
    input_text: &str,
    stream: bool,
) -> usize {
    let episode_id = Uuid::now_v7();

    let payload = json!({
        "function_name": function_name,
        "variant_name": variant_name,
        "episode_id": episode_id,
        "input": {
            "system": {"assistant_name": "TestBot"},
            "messages": [
                {
                    "role": "user",
                    "content": input_text
                }
            ]
        },
        "stream": stream,
        "include_raw_usage": true,
        "cache_options": {
            "enabled": "on",
            "lookback_s": 60
        }
    });

    if stream {
        let mut chunks = Client::new()
            .post(get_gateway_endpoint("/inference"))
            .json(&payload)
            .eventsource()
            .unwrap();

        let mut raw_usage_count = 0;

        while let Some(chunk) = chunks.next().await {
            let chunk = chunk.unwrap();
            let Event::Message(chunk) = chunk else {
                continue;
            };
            if chunk.data == "[DONE]" {
                break;
            }

            let chunk_json: Value = serde_json::from_str(&chunk.data).unwrap();
            // Check for raw_usage nested inside usage
            if let Some(usage) = chunk_json.get("usage")
                && let Some(raw_usage) = usage.get("raw_usage")
                && let Some(array) = raw_usage.as_array()
            {
                raw_usage_count = array.len();
            }
        }

        raw_usage_count
    } else {
        let response = Client::new()
            .post(get_gateway_endpoint("/inference"))
            .json(&payload)
            .send()
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);

        let response_json: Value = response.json().await.unwrap();

        // Get raw_usage nested inside usage
        response_json
            .get("usage")
            .and_then(|usage| usage.get("raw_usage"))
            .and_then(|v| v.as_array())
            .map(|a| a.len())
            .unwrap_or(0)
    }
}

#[tokio::test]
async fn e2e_test_raw_usage_cache_behavior_non_streaming() {
    // Use a unique input to ensure cache miss on first request
    let unique_input = format!("What is 2+2? Cache test: {}", Uuid::now_v7());

    // First request: should be a cache miss, raw_usage should have entries
    let first_count =
        make_request_and_get_raw_usage_count("weather_helper", "openai", &unique_input, false)
            .await;

    assert!(
        first_count > 0,
        "First request (cache miss) should have at least one raw_usage entry, got {first_count}"
    );

    // Wait a moment for cache write to complete
    tokio::time::sleep(std::time::Duration::from_millis(500)).await;

    // Second request with same input: should be a cache hit
    // Cache hits should be EXCLUDED from raw_usage
    let second_count =
        make_request_and_get_raw_usage_count("weather_helper", "openai", &unique_input, false)
            .await;

    assert_eq!(
        second_count, 0,
        "Second request (cache hit) should have zero raw_usage entries (cached responses excluded), got {second_count}"
    );
}

#[tokio::test]
async fn e2e_test_raw_usage_cache_behavior_streaming() {
    // Use a unique input to ensure cache miss on first request
    let unique_input = format!("What is 3+3? Cache streaming test: {}", Uuid::now_v7());

    // First request: should be a cache miss
    let first_count =
        make_request_and_get_raw_usage_count("weather_helper", "openai", &unique_input, true).await;

    // Note: For streaming, raw_usage comes from previous_model_inference_results,
    // which excludes the current streaming inference. For a simple variant with
    // only one model inference, the first streaming request may have 0 raw_usage
    // entries because the only inference is the one currently streaming.
    // This is expected behavior - we're testing that cache hits don't add entries.

    // Wait for cache write
    tokio::time::sleep(std::time::Duration::from_millis(500)).await;

    // Second request: should be a cache hit
    let second_count =
        make_request_and_get_raw_usage_count("weather_helper", "openai", &unique_input, true).await;

    // For streaming cache hit, raw_usage should still not include cached entries
    assert!(
        second_count <= first_count,
        "Cache hit should not increase raw_usage count"
    );
}

#[tokio::test]
async fn e2e_test_raw_usage_cache_disabled() {
    // Use a unique input
    let unique_input = format!("What is 4+4? Cache disabled test: {}", Uuid::now_v7());

    let episode_id = Uuid::now_v7();

    // Request with cache disabled
    let payload = json!({
        "function_name": "weather_helper",
        "variant_name": "openai",
        "episode_id": episode_id,
        "input": {
            "system": {"assistant_name": "TestBot"},
            "messages": [
                {
                    "role": "user",
                    "content": unique_input
                }
            ]
        },
        "stream": false,
        "include_raw_usage": true,
        "cache_options": {
            "enabled": "off"
        }
    });

    let response = Client::new()
        .post(get_gateway_endpoint("/inference"))
        .json(&payload)
        .send()
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);

    let response_json: Value = response.json().await.unwrap();

    // With cache disabled, raw_usage should still work (nested inside usage)
    let usage = response_json.get("usage");
    assert!(usage.is_some(), "usage should be present");

    let raw_usage = usage.unwrap().get("raw_usage");
    assert!(
        raw_usage.is_some(),
        "raw_usage should be present inside usage even with cache disabled"
    );

    let raw_usage_array = raw_usage.unwrap().as_array().unwrap();
    assert!(
        !raw_usage_array.is_empty(),
        "raw_usage should have entries when cache is disabled (no cache hits possible)"
    );
}
