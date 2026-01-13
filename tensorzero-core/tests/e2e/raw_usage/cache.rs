//! E2E tests for `include_raw_usage` cache behavior.
//!
//! Tests that cache hits are correctly excluded from raw_usage.
//!
//! The TensorZero native API tests use embedded gateway with unique databases
//! for test isolation. The OpenAI-compatible tests use the shared HTTP gateway.

use futures::StreamExt;
use reqwest::{Client, StatusCode};
use reqwest_eventsource::{Event, RequestBuilderExt};
use serde_json::{Map, Value, json};
use tensorzero::test_helpers::{
    make_embedded_gateway_e2e_with_unique_db, start_http_gateway_with_unique_db,
};
use tensorzero::{
    CacheParamsOptions, ClientInferenceParams, InferenceOutput, Input, InputMessage,
    InputMessageContent,
};
use tensorzero_core::cache::CacheEnabledMode;
use tensorzero_core::inference::types::usage::RawUsageEntry;
use tensorzero_core::inference::types::{Arguments, Role, System, Text};
use uuid::Uuid;

fn assert_raw_usage_entry(entry: &RawUsageEntry) {
    assert!(
        !entry.data.is_null(),
        "raw_usage entry should have non-null data for OpenAI chat completions"
    );
    assert!(
        entry.data.get("total_tokens").is_some(),
        "raw_usage should include `total_tokens` for chat completions"
    );
    assert!(
        entry
            .data
            .get("prompt_tokens_details")
            .and_then(|d: &Value| d.get("cached_tokens"))
            .is_some(),
        "raw_usage should include `prompt_tokens_details.cached_tokens` for chat completions"
    );
    assert!(
        entry
            .data
            .get("completion_tokens_details")
            .and_then(|d: &Value| d.get("reasoning_tokens"))
            .is_some(),
        "raw_usage should include `completion_tokens_details.reasoning_tokens` for chat completions"
    );
}

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

/// Tests cache behavior with embedded gateway (unique database for test isolation)
#[tokio::test(flavor = "multi_thread")]
async fn test_raw_usage_cache_behavior_non_streaming() {
    // Use embedded gateway with unique database for test isolation
    let client = make_embedded_gateway_e2e_with_unique_db("cache_non_streaming").await;

    // Fixed input for deterministic provider-proxy caching
    let input = "raw_usage_non_streaming: What is 2+2?";

    // First request: should be a cache miss, raw_usage should have entries
    let first_response = client
        .inference(ClientInferenceParams {
            function_name: Some("weather_helper".to_string()),
            variant_name: Some("openai".to_string()),
            episode_id: Some(Uuid::now_v7()),
            input: Input {
                system: Some(System::Template(Arguments({
                    let mut args = Map::new();
                    args.insert("assistant_name".to_string(), json!("TestBot"));
                    args
                }))),
                messages: vec![InputMessage {
                    role: Role::User,
                    content: vec![InputMessageContent::Text(Text {
                        text: input.to_string(),
                    })],
                }],
            },
            stream: Some(false),
            include_raw_usage: true,
            cache_options: CacheParamsOptions {
                enabled: CacheEnabledMode::On,
                max_age_s: Some(60),
            },
            ..Default::default()
        })
        .await
        .unwrap();

    let InferenceOutput::NonStreaming(first_response) = first_response else {
        panic!("Expected non-streaming response");
    };

    let first_raw_usage = first_response
        .raw_usage()
        .expect("raw_usage should be present when include_raw_usage is true");
    assert!(
        !first_raw_usage.is_empty(),
        "First request (cache miss) should have at least one raw_usage entry"
    );
    assert_raw_usage_entry(&first_raw_usage[0]);

    // Wait for cache write to complete
    tokio::time::sleep(std::time::Duration::from_millis(500)).await;

    // Second request with same input: should be a cache hit
    let second_response = client
        .inference(ClientInferenceParams {
            function_name: Some("weather_helper".to_string()),
            variant_name: Some("openai".to_string()),
            episode_id: Some(Uuid::now_v7()),
            input: Input {
                system: Some(System::Template(Arguments({
                    let mut args = Map::new();
                    args.insert("assistant_name".to_string(), json!("TestBot"));
                    args
                }))),
                messages: vec![InputMessage {
                    role: Role::User,
                    content: vec![InputMessageContent::Text(Text {
                        text: input.to_string(),
                    })],
                }],
            },
            stream: Some(false),
            include_raw_usage: true,
            cache_options: CacheParamsOptions {
                enabled: CacheEnabledMode::On,
                max_age_s: Some(60),
            },
            ..Default::default()
        })
        .await
        .unwrap();

    let InferenceOutput::NonStreaming(second_response) = second_response else {
        panic!("Expected non-streaming response");
    };

    let second_raw_usage = second_response
        .raw_usage()
        .expect("raw_usage should be present when include_raw_usage is true");
    assert!(
        second_raw_usage.is_empty(),
        "Second request (cache hit) should have empty raw_usage array, got {second_raw_usage:?}"
    );
}

/// Tests streaming cache behavior with embedded gateway (unique database for test isolation)
#[tokio::test(flavor = "multi_thread")]
async fn test_raw_usage_cache_behavior_streaming() {
    // Use embedded gateway with unique database for test isolation
    let client = make_embedded_gateway_e2e_with_unique_db("cache_streaming").await;

    // Fixed input for deterministic provider-proxy caching
    let input = "raw_usage_streaming: What is 3+3?";

    // Helper to make streaming request and collect raw_usage
    async fn make_streaming_request(
        client: &tensorzero::Client,
        input: &str,
    ) -> Vec<RawUsageEntry> {
        let response = client
            .inference(ClientInferenceParams {
                function_name: Some("weather_helper".to_string()),
                variant_name: Some("openai".to_string()),
                episode_id: Some(Uuid::now_v7()),
                input: Input {
                    system: Some(System::Template(Arguments({
                        let mut args = Map::new();
                        args.insert("assistant_name".to_string(), json!("TestBot"));
                        args
                    }))),
                    messages: vec![InputMessage {
                        role: Role::User,
                        content: vec![InputMessageContent::Text(Text {
                            text: input.to_string(),
                        })],
                    }],
                },
                stream: Some(true),
                include_raw_usage: true,
                cache_options: CacheParamsOptions {
                    enabled: CacheEnabledMode::On,
                    max_age_s: Some(60),
                },
                ..Default::default()
            })
            .await
            .unwrap();

        let InferenceOutput::Streaming(mut stream) = response else {
            panic!("Expected streaming response");
        };

        let mut raw_usage_entries = Vec::new();
        while let Some(chunk) = stream.next().await {
            let chunk = chunk.unwrap();
            let entries = match &chunk {
                tensorzero::InferenceResponseChunk::Chat(c) => c.raw_usage.as_ref(),
                tensorzero::InferenceResponseChunk::Json(j) => j.raw_usage.as_ref(),
            };
            if let Some(entries) = entries {
                raw_usage_entries.extend(entries.iter().cloned());
            }
        }
        raw_usage_entries
    }

    // First request: should be a cache miss
    let first_raw_usage = make_streaming_request(&client, input).await;
    assert!(
        !first_raw_usage.is_empty(),
        "First streaming request (cache miss) should have at least one raw_usage entry"
    );
    assert_raw_usage_entry(&first_raw_usage[0]);

    // Wait for cache write
    tokio::time::sleep(std::time::Duration::from_millis(500)).await;

    // Second request: should be a cache hit
    let second_raw_usage = make_streaming_request(&client, input).await;
    assert!(
        second_raw_usage.is_empty(),
        "Second streaming request (cache hit) should have empty raw_usage array, got {second_raw_usage:?}"
    );
}

/// Tests raw_usage with cache disabled using embedded gateway
#[tokio::test(flavor = "multi_thread")]
async fn test_raw_usage_cache_disabled() {
    // Use embedded gateway with unique database for test isolation
    let client = make_embedded_gateway_e2e_with_unique_db("cache_disabled").await;

    let input = "raw_usage_cache_disabled: What is 4+4?";

    let response = client
        .inference(ClientInferenceParams {
            function_name: Some("weather_helper".to_string()),
            variant_name: Some("openai".to_string()),
            episode_id: Some(Uuid::now_v7()),
            input: Input {
                system: Some(System::Template(Arguments({
                    let mut args = Map::new();
                    args.insert("assistant_name".to_string(), json!("TestBot"));
                    args
                }))),
                messages: vec![InputMessage {
                    role: Role::User,
                    content: vec![InputMessageContent::Text(Text {
                        text: input.to_string(),
                    })],
                }],
            },
            stream: Some(false),
            include_raw_usage: true,
            cache_options: CacheParamsOptions {
                enabled: CacheEnabledMode::Off,
                max_age_s: None,
            },
            ..Default::default()
        })
        .await
        .unwrap();

    let InferenceOutput::NonStreaming(response) = response else {
        panic!("Expected non-streaming response");
    };

    let raw_usage = response
        .raw_usage()
        .expect("raw_usage should be present when include_raw_usage is true");
    assert!(
        !raw_usage.is_empty(),
        "raw_usage should have entries when cache is disabled (no cache hits possible)"
    );
    assert_raw_usage_entry(&raw_usage[0]);
}

/// Tests streaming raw_usage with cache disabled using embedded gateway
#[tokio::test(flavor = "multi_thread")]
async fn test_raw_usage_cache_disabled_streaming() {
    // Use embedded gateway with unique database for test isolation
    let client = make_embedded_gateway_e2e_with_unique_db("cache_disabled_streaming").await;

    let input = "raw_usage_cache_disabled_streaming: What is 5+5?";

    let response = client
        .inference(ClientInferenceParams {
            function_name: Some("weather_helper".to_string()),
            variant_name: Some("openai".to_string()),
            episode_id: Some(Uuid::now_v7()),
            input: Input {
                system: Some(System::Template(Arguments({
                    let mut args = Map::new();
                    args.insert("assistant_name".to_string(), json!("TestBot"));
                    args
                }))),
                messages: vec![InputMessage {
                    role: Role::User,
                    content: vec![InputMessageContent::Text(Text {
                        text: input.to_string(),
                    })],
                }],
            },
            stream: Some(true),
            include_raw_usage: true,
            cache_options: CacheParamsOptions {
                enabled: CacheEnabledMode::Off,
                max_age_s: None,
            },
            ..Default::default()
        })
        .await
        .unwrap();

    let InferenceOutput::Streaming(mut stream) = response else {
        panic!("Expected streaming response");
    };

    let mut raw_usage_entries = Vec::new();
    while let Some(chunk) = stream.next().await {
        let chunk = chunk.unwrap();
        let entries = match &chunk {
            tensorzero::InferenceResponseChunk::Chat(c) => c.raw_usage.as_ref(),
            tensorzero::InferenceResponseChunk::Json(j) => j.raw_usage.as_ref(),
        };
        if let Some(entries) = entries {
            raw_usage_entries.extend(entries.iter().cloned());
        }
    }

    // With cache disabled, raw_usage should have entries (no cache hits possible)
    assert!(
        !raw_usage_entries.is_empty(),
        "raw_usage should have entries when cache is disabled (no cache hits possible)"
    );
    assert_raw_usage_entry(&raw_usage_entries[0]);
}

// ============================================================================
// OpenAI-Compatible API Cache Tests (with unique HTTP gateway per test)
// ============================================================================

/// Helper to make OpenAI-compatible request and get tensorzero_raw_usage array
async fn make_openai_request_to_gateway(
    base_url: &str,
    input_text: &str,
    stream: bool,
) -> Vec<Value> {
    let episode_id = Uuid::now_v7();

    let payload = json!({
        "model": "tensorzero::function_name::weather_helper",
        "messages": [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "tensorzero::arguments": {
                            "assistant_name": "TestBot"
                        }
                    }
                ]
            },
            {
                "role": "user",
                "content": input_text
            }
        ],
        "stream": stream,
        "tensorzero::variant_name": "openai",
        "tensorzero::episode_id": episode_id,
        "tensorzero::include_raw_usage": true,
        "tensorzero::cache_options": {
            "enabled": "on",
            "max_age_s": 60
        }
    });

    let url = format!("{base_url}/openai/v1/chat/completions");

    if stream {
        let mut chunks = Client::new()
            .post(&url)
            .json(&payload)
            .eventsource()
            .unwrap();

        // Collect raw_usage entries from all chunks, similar to native API
        let mut raw_usage_entries: Vec<Value> = Vec::new();

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
            if let Some(ru) = chunk_json.get("tensorzero_raw_usage")
                && let Some(arr) = ru.as_array()
            {
                raw_usage_entries.extend(arr.iter().cloned());
            }
        }

        raw_usage_entries
    } else {
        let response = Client::new()
            .post(&url)
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

/// Tests OpenAI-compatible cache behavior with HTTP gateway (unique database for test isolation)
#[tokio::test(flavor = "multi_thread")]
async fn test_raw_usage_cache_openai_compatible_non_streaming() {
    // Start HTTP gateway with unique database
    let (base_url, _shutdown_handle) =
        start_http_gateway_with_unique_db("openai_cache_non_streaming").await;

    let input = "raw_usage_openai_non_streaming: What is 5+5?";

    // First request: should be a cache miss
    let first_raw_usage = make_openai_request_to_gateway(&base_url, input, false).await;

    assert!(
        !first_raw_usage.is_empty(),
        "First request (cache miss) should have at least one tensorzero_raw_usage entry"
    );
    assert_openai_chat_usage_details(&first_raw_usage[0]);

    // Wait for cache write
    tokio::time::sleep(std::time::Duration::from_millis(500)).await;

    // Second request: should be a cache hit
    // tensorzero_raw_usage should still be present but empty (field present as [])
    let second_raw_usage = make_openai_request_to_gateway(&base_url, input, false).await;

    assert!(
        second_raw_usage.is_empty(),
        "Second request (cache hit) should have empty tensorzero_raw_usage array, got {second_raw_usage:?}"
    );
}

/// Tests OpenAI-compatible streaming cache behavior with HTTP gateway (unique database for test isolation)
#[tokio::test(flavor = "multi_thread")]
async fn test_raw_usage_cache_openai_compatible_streaming() {
    // Start HTTP gateway with unique database
    let (base_url, _shutdown_handle) =
        start_http_gateway_with_unique_db("openai_cache_streaming").await;

    let input = "raw_usage_openai_streaming: What is 6+6?";

    // First request: should be a cache miss
    let first_raw_usage = make_openai_request_to_gateway(&base_url, input, true).await;

    assert!(
        !first_raw_usage.is_empty(),
        "First streaming request (cache miss) should have at least one tensorzero_raw_usage entry"
    );
    assert_openai_chat_usage_details(&first_raw_usage[0]);

    // Wait for cache write
    tokio::time::sleep(std::time::Duration::from_millis(500)).await;

    // Second request: should be a cache hit
    // tensorzero_raw_usage should still be present but empty (field present as [])
    let second_raw_usage = make_openai_request_to_gateway(&base_url, input, true).await;

    assert!(
        second_raw_usage.is_empty(),
        "Second streaming request (cache hit) should have empty tensorzero_raw_usage array, got {second_raw_usage:?}"
    );
}
