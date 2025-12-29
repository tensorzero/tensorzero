//! E2E tests for `include_raw_usage` cache behavior.
//!
//! Tests that cache hits are correctly excluded from raw_usage.

use futures::StreamExt;
use reqwest::{Client, StatusCode};
use reqwest_eventsource::{Event, RequestBuilderExt};
use serde_json::{Value, json};
use uuid::Uuid;

use crate::common::get_gateway_endpoint;

/// Makes an inference request and returns the raw_usage array (panics if not present)
async fn make_request_and_get_raw_usage(
    function_name: &str,
    variant_name: &str,
    input_text: &str,
    stream: bool,
) -> Vec<Value> {
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

        let mut raw_usage: Option<Vec<Value>> = None;

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
            // Note: Only the final empty usage chunk has raw_usage, not all chunks with usage
            if let Some(usage) = chunk_json.get("usage")
                && let Some(ru) = usage.get("raw_usage")
            {
                raw_usage = Some(ru.as_array().expect("raw_usage should be an array").clone());
            }
        }

        raw_usage.expect("Should have received a chunk with usage containing raw_usage")
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
        let usage = response_json.get("usage").expect("usage should be present");
        let raw_usage = usage
            .get("raw_usage")
            .expect("raw_usage should be present in usage when include_raw_usage is true");
        raw_usage
            .as_array()
            .expect("raw_usage should be an array")
            .clone()
    }
}

#[tokio::test]
async fn e2e_test_raw_usage_cache_behavior_non_streaming() {
    // Use a unique input to ensure cache miss on first request
    let unique_input = format!("What is 2+2? Cache test: {}", Uuid::now_v7());

    // First request: should be a cache miss, raw_usage should have entries
    let first_raw_usage =
        make_request_and_get_raw_usage("weather_helper", "openai", &unique_input, false).await;

    assert!(
        !first_raw_usage.is_empty(),
        "First request (cache miss) should have at least one raw_usage entry, got {first_raw_usage:?}"
    );

    // Wait a moment for cache write to complete
    tokio::time::sleep(std::time::Duration::from_millis(500)).await;

    // Second request with same input: should be a cache hit
    // Cache hits should be EXCLUDED from raw_usage, but field should still be present as empty array
    let second_raw_usage =
        make_request_and_get_raw_usage("weather_helper", "openai", &unique_input, false).await;

    assert!(
        second_raw_usage.is_empty(),
        "Second request (cache hit) should have empty raw_usage array (cached responses excluded), got {second_raw_usage:?}"
    );
}

#[tokio::test]
async fn e2e_test_raw_usage_cache_behavior_streaming() {
    // Use a unique input to ensure cache miss on first request
    let unique_input = format!("What is 3+3? Cache streaming test: {}", Uuid::now_v7());

    // First request: should be a cache miss
    // raw_usage should be present (even if potentially empty for simple streaming variants)
    let first_raw_usage =
        make_request_and_get_raw_usage("weather_helper", "openai", &unique_input, true).await;

    // Note: For streaming, raw_usage includes the current streaming inference's usage
    // when it completes. So we should have at least one entry.
    assert!(
        !first_raw_usage.is_empty(),
        "First streaming request (cache miss) should have at least one raw_usage entry"
    );

    // Wait for cache write
    tokio::time::sleep(std::time::Duration::from_millis(500)).await;

    // Second request: should be a cache hit
    // raw_usage should still be present but empty (field present as [])
    let second_raw_usage =
        make_request_and_get_raw_usage("weather_helper", "openai", &unique_input, true).await;

    assert!(
        second_raw_usage.is_empty(),
        "Second streaming request (cache hit) should have empty raw_usage array, got {second_raw_usage:?}"
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

#[tokio::test]
async fn e2e_test_raw_usage_cache_disabled_streaming() {
    // Use a unique input
    let unique_input = format!(
        "What is 5+5? Cache disabled streaming test: {}",
        Uuid::now_v7()
    );

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
        "stream": true,
        "include_raw_usage": true,
        "cache_options": {
            "enabled": "off"
        }
    });

    let mut chunks = Client::new()
        .post(get_gateway_endpoint("/inference"))
        .json(&payload)
        .eventsource()
        .unwrap();

    let mut raw_usage: Option<Vec<Value>> = None;

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
            && let Some(ru) = usage.get("raw_usage")
        {
            raw_usage = Some(ru.as_array().expect("raw_usage should be an array").clone());
        }
    }

    let raw_usage =
        raw_usage.expect("Should have received a chunk with usage containing raw_usage");

    // With cache disabled, raw_usage should have entries (no cache hits possible)
    assert!(
        !raw_usage.is_empty(),
        "raw_usage should have entries when cache is disabled (no cache hits possible)"
    );
}

// ============================================================================
// OpenAI-Compatible API Cache Tests
// ============================================================================

/// Helper to make OpenAI-compatible request and get tensorzero_raw_usage array
async fn make_openai_request_and_get_raw_usage(input_text: &str, stream: bool) -> Vec<Value> {
    let episode_id = Uuid::now_v7();

    let payload = json!({
        "model": "tensorzero::function_name::basic_test",
        "messages": [
            {
                "role": "system",
                "content": [{"type": "tensorzero::template", "name": "system", "arguments": {"assistant_name": "TestBot"}}]
            },
            {
                "role": "user",
                "content": input_text
            }
        ],
        "stream": stream,
        "tensorzero::episode_id": episode_id,
        "tensorzero::include_raw_usage": true,
        "tensorzero::cache_options": {
            "enabled": "on",
            "lookback_s": 60
        }
    });

    if stream {
        let mut chunks = Client::new()
            .post(get_gateway_endpoint("/openai/v1/chat/completions"))
            .json(&payload)
            .eventsource()
            .unwrap();

        let mut raw_usage: Option<Vec<Value>> = None;

        while let Some(chunk) = chunks.next().await {
            let chunk = chunk.unwrap();
            let Event::Message(chunk) = chunk else {
                continue;
            };
            if chunk.data == "[DONE]" {
                break;
            }

            let chunk_json: Value = serde_json::from_str(&chunk.data).unwrap();
            // Check for tensorzero_raw_usage nested inside usage
            // Note: Only the final usage chunk has tensorzero_raw_usage
            if let Some(usage) = chunk_json.get("usage")
                && let Some(ru) = usage.get("tensorzero_raw_usage")
            {
                raw_usage = Some(
                    ru.as_array()
                        .expect("tensorzero_raw_usage should be an array")
                        .clone(),
                );
            }
        }

        raw_usage.expect("Should have received a chunk with usage containing tensorzero_raw_usage")
    } else {
        let response = Client::new()
            .post(get_gateway_endpoint("/openai/v1/chat/completions"))
            .json(&payload)
            .send()
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);

        let response_json: Value = response.json().await.unwrap();

        // Get tensorzero_raw_usage nested inside usage
        let usage = response_json.get("usage").expect("usage should be present");
        let raw_usage = usage
            .get("tensorzero_raw_usage")
            .expect("tensorzero_raw_usage should be present in usage when tensorzero::include_raw_usage is true");
        raw_usage
            .as_array()
            .expect("tensorzero_raw_usage should be an array")
            .clone()
    }
}

#[tokio::test]
async fn e2e_test_raw_usage_cache_openai_compatible_non_streaming() {
    // Use a unique input to ensure cache miss on first request
    let unique_input = format!("What is 5+5? OpenAI cache test: {}", Uuid::now_v7());

    // First request: should be a cache miss
    let first_raw_usage = make_openai_request_and_get_raw_usage(&unique_input, false).await;

    assert!(
        !first_raw_usage.is_empty(),
        "First request (cache miss) should have at least one tensorzero_raw_usage entry"
    );

    // Wait for cache write
    tokio::time::sleep(std::time::Duration::from_millis(500)).await;

    // Second request: should be a cache hit
    // tensorzero_raw_usage should still be present but empty (field present as [])
    let second_raw_usage = make_openai_request_and_get_raw_usage(&unique_input, false).await;

    assert!(
        second_raw_usage.is_empty(),
        "Second request (cache hit) should have empty tensorzero_raw_usage array, got {second_raw_usage:?}"
    );
}

#[tokio::test]
async fn e2e_test_raw_usage_cache_openai_compatible_streaming() {
    // Use a unique input to ensure cache miss on first request
    let unique_input = format!(
        "What is 6+6? OpenAI streaming cache test: {}",
        Uuid::now_v7()
    );

    // First request: should be a cache miss
    let first_raw_usage = make_openai_request_and_get_raw_usage(&unique_input, true).await;

    assert!(
        !first_raw_usage.is_empty(),
        "First streaming request (cache miss) should have at least one tensorzero_raw_usage entry"
    );

    // Wait for cache write
    tokio::time::sleep(std::time::Duration::from_millis(500)).await;

    // Second request: should be a cache hit
    // tensorzero_raw_usage should still be present but empty (field present as [])
    let second_raw_usage = make_openai_request_and_get_raw_usage(&unique_input, true).await;

    assert!(
        second_raw_usage.is_empty(),
        "Second streaming request (cache hit) should have empty tensorzero_raw_usage array, got {second_raw_usage:?}"
    );
}
