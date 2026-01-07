//! E2E tests for `include_raw_usage` cache behavior.
//!
//! Tests that cache hits are correctly excluded from raw_usage.

use futures::StreamExt;
use reqwest::{Client, StatusCode};
use reqwest_eventsource::{Event, RequestBuilderExt};
use serde_json::{Value, json};
use uuid::Uuid;

use crate::common::get_gateway_endpoint;

fn assert_openai_chat_usage_details(entry: &Value) {
    let data = entry.get("data").unwrap_or_else(|| {
        panic!("raw_usage entry should include data field for chat completions: {entry:?}")
    });
    assert!(
        !data.is_null(),
        "raw_usage entry data should NOT be null for OpenAI chat completions. Entry: {entry:?}"
    );
    assert!(
        data.is_object(),
        "raw_usage entry data should be an object for chat completions"
    );
    assert!(
        data.get("total_tokens").is_some(),
        "raw_usage should include `total_tokens` for chat completions"
    );
    assert!(
        data.get("prompt_tokens_details")
            .and_then(|details| details.get("cached_tokens"))
            .is_some(),
        "raw_usage should include `prompt_tokens_details.cached_tokens` for chat completions"
    );
    assert!(
        data.get("completion_tokens_details")
            .and_then(|details| details.get("reasoning_tokens"))
            .is_some(),
        "raw_usage should include `completion_tokens_details.reasoning_tokens` for chat completions"
    );
}

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
            // Check for raw_usage at chunk level (sibling to usage)
            if let Some(ru) = chunk_json.get("raw_usage") {
                raw_usage = Some(ru.as_array().expect("raw_usage should be an array").clone());
            }
        }

        raw_usage.expect("Should have received a chunk with raw_usage")
    } else {
        let response = Client::new()
            .post(get_gateway_endpoint("/inference"))
            .json(&payload)
            .send()
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);

        let response_json: Value = response.json().await.unwrap();

        // Get raw_usage at response level (sibling to usage)
        let raw_usage = response_json
            .get("raw_usage")
            .expect("raw_usage should be present when include_raw_usage is true");
        raw_usage
            .as_array()
            .expect("raw_usage should be an array")
            .clone()
    }
}

#[tokio::test]
async fn test_raw_usage_cache_behavior_non_streaming() {
    // Use a fixed input for deterministic provider-proxy caching.
    // Gateway cache (ClickHouse) is fresh each CI run, so first request will be a cache miss.
    // See: https://github.com/tensorzero/tensorzero/issues/5380
    let unique_input = "What is 2+2? Cache test: test_raw_usage_cache_behavior_non_streaming_v1";

    // First request: should be a cache miss, raw_usage should have entries
    let first_raw_usage =
        make_request_and_get_raw_usage("weather_helper", "openai", unique_input, false).await;

    assert!(
        !first_raw_usage.is_empty(),
        "First request (cache miss) should have at least one raw_usage entry, got {first_raw_usage:?}"
    );
    assert_openai_chat_usage_details(&first_raw_usage[0]);

    // Wait a moment for cache write to complete
    tokio::time::sleep(std::time::Duration::from_millis(500)).await;

    // Second request with same input: should be a cache hit
    // Cache hits should be EXCLUDED from raw_usage, but field should still be present as empty array
    let second_raw_usage =
        make_request_and_get_raw_usage("weather_helper", "openai", unique_input, false).await;

    assert!(
        second_raw_usage.is_empty(),
        "Second request (cache hit) should have empty raw_usage array (cached responses excluded), got {second_raw_usage:?}"
    );
}

#[tokio::test]
async fn test_raw_usage_cache_behavior_streaming() {
    // Use a fixed input for deterministic provider-proxy caching.
    // Gateway cache (ClickHouse) is fresh each CI run, so first request will be a cache miss.
    // See: https://github.com/tensorzero/tensorzero/issues/5380
    let unique_input =
        "What is 3+3? Cache streaming test: test_raw_usage_cache_behavior_streaming_v1";

    // First request: should be a cache miss
    // raw_usage should be present (even if potentially empty for simple streaming variants)
    let first_raw_usage =
        make_request_and_get_raw_usage("weather_helper", "openai", unique_input, true).await;

    // Note: For streaming, raw_usage includes the current streaming inference's usage
    // when it completes. So we should have at least one entry.
    assert!(
        !first_raw_usage.is_empty(),
        "First streaming request (cache miss) should have at least one raw_usage entry"
    );
    assert_openai_chat_usage_details(&first_raw_usage[0]);

    // Wait for cache write
    tokio::time::sleep(std::time::Duration::from_millis(500)).await;

    // Second request: should be a cache hit
    // raw_usage should still be present but empty (field present as [])
    let second_raw_usage =
        make_request_and_get_raw_usage("weather_helper", "openai", unique_input, true).await;

    assert!(
        second_raw_usage.is_empty(),
        "Second streaming request (cache hit) should have empty raw_usage array, got {second_raw_usage:?}"
    );
}

#[tokio::test]
async fn test_raw_usage_cache_disabled() {
    // Use a fixed input for deterministic provider-proxy caching.
    // See: https://github.com/tensorzero/tensorzero/issues/5380
    let unique_input = "What is 4+4? Cache disabled test: test_raw_usage_cache_disabled_v1";

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

    // With cache disabled, raw_usage should still work (at response level, sibling to usage)
    let raw_usage = response_json.get("raw_usage");
    assert!(
        raw_usage.is_some(),
        "raw_usage should be present even with cache disabled"
    );

    let raw_usage_array = raw_usage.unwrap().as_array().unwrap();
    assert!(
        !raw_usage_array.is_empty(),
        "raw_usage should have entries when cache is disabled (no cache hits possible)"
    );
    assert_openai_chat_usage_details(&raw_usage_array[0]);
}

#[tokio::test]
async fn test_raw_usage_cache_disabled_streaming() {
    // Use a fixed input for deterministic provider-proxy caching.
    // See: https://github.com/tensorzero/tensorzero/issues/5380
    let unique_input =
        "What is 5+5? Cache disabled streaming test: test_raw_usage_cache_disabled_streaming_v1";

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
        // Check for raw_usage at chunk level (sibling to usage)
        if let Some(ru) = chunk_json.get("raw_usage") {
            raw_usage = Some(ru.as_array().expect("raw_usage should be an array").clone());
        }
    }

    let raw_usage = raw_usage.expect("Should have received a chunk with raw_usage");

    // With cache disabled, raw_usage should have entries (no cache hits possible)
    assert!(
        !raw_usage.is_empty(),
        "raw_usage should have entries when cache is disabled (no cache hits possible)"
    );
    assert_openai_chat_usage_details(&raw_usage[0]);
}

// ============================================================================
// OpenAI-Compatible API Cache Tests
// ============================================================================

/// Helper to make OpenAI-compatible request and get tensorzero_raw_usage array
async fn make_openai_request_and_get_raw_usage(input_text: &str, stream: bool) -> Vec<Value> {
    let episode_id = Uuid::now_v7();

    let payload = json!({
        "model": "tensorzero::model_name::openai::gpt-5-nano",
        "messages": [
            {
                "role": "user",
                "content": input_text
            }
        ],
        "params": {
            "chat_completion": {
                "reasoning_effort": "minimal"
            }
        },
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
            // Check for tensorzero_raw_usage at chunk level (sibling to usage)
            if let Some(ru) = chunk_json.get("tensorzero_raw_usage") {
                raw_usage = Some(
                    ru.as_array()
                        .expect("tensorzero_raw_usage should be an array")
                        .clone(),
                );
            }
        }

        raw_usage.expect("Should have received a chunk with tensorzero_raw_usage")
    } else {
        let response = Client::new()
            .post(get_gateway_endpoint("/openai/v1/chat/completions"))
            .json(&payload)
            .send()
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);

        let response_json: Value = response.json().await.unwrap();

        // Get tensorzero_raw_usage at response level (sibling to usage)
        let raw_usage = response_json.get("tensorzero_raw_usage").expect(
            "tensorzero_raw_usage should be present when tensorzero::include_raw_usage is true",
        );
        raw_usage
            .as_array()
            .expect("tensorzero_raw_usage should be an array")
            .clone()
    }
}

#[tokio::test]
async fn test_raw_usage_cache_openai_compatible_non_streaming() {
    // Use a fixed input for deterministic provider-proxy caching.
    // Gateway cache (ClickHouse) is fresh each CI run, so first request will be a cache miss.
    // See: https://github.com/tensorzero/tensorzero/issues/5380
    let unique_input =
        "What is 5+5? OpenAI cache test: test_raw_usage_cache_openai_compatible_non_streaming_v1";

    // First request: should be a cache miss
    let first_raw_usage = make_openai_request_and_get_raw_usage(unique_input, false).await;

    assert!(
        !first_raw_usage.is_empty(),
        "First request (cache miss) should have at least one tensorzero_raw_usage entry"
    );
    assert_openai_chat_usage_details(&first_raw_usage[0]);

    // Wait for cache write
    tokio::time::sleep(std::time::Duration::from_millis(500)).await;

    // Second request: should be a cache hit
    // tensorzero_raw_usage should still be present but empty (field present as [])
    let second_raw_usage = make_openai_request_and_get_raw_usage(unique_input, false).await;

    assert!(
        second_raw_usage.is_empty(),
        "Second request (cache hit) should have empty tensorzero_raw_usage array, got {second_raw_usage:?}"
    );
}

#[tokio::test]
async fn test_raw_usage_cache_openai_compatible_streaming() {
    // Use a fixed input for deterministic provider-proxy caching.
    // Gateway cache (ClickHouse) is fresh each CI run, so first request will be a cache miss.
    // See: https://github.com/tensorzero/tensorzero/issues/5380
    let unique_input = "What is 6+6? OpenAI streaming cache test: test_raw_usage_cache_openai_compatible_streaming_v1";

    // First request: should be a cache miss
    let first_raw_usage = make_openai_request_and_get_raw_usage(unique_input, true).await;

    assert!(
        !first_raw_usage.is_empty(),
        "First streaming request (cache miss) should have at least one tensorzero_raw_usage entry"
    );
    assert_openai_chat_usage_details(&first_raw_usage[0]);

    // Wait for cache write
    tokio::time::sleep(std::time::Duration::from_millis(500)).await;

    // Second request: should be a cache hit
    // tensorzero_raw_usage should still be present but empty (field present as [])
    let second_raw_usage = make_openai_request_and_get_raw_usage(unique_input, true).await;

    assert!(
        second_raw_usage.is_empty(),
        "Second streaming request (cache hit) should have empty tensorzero_raw_usage array, got {second_raw_usage:?}"
    );
}
