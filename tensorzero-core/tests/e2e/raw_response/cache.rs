//! E2E tests for `include_raw_response` cache behavior.
//!
//! Tests that cache hits result in empty raw_response arrays.
//!
//! The TensorZero native API tests use embedded gateway with unique databases
//! for test isolation. The OpenAI-compatible tests use the shared HTTP gateway.

use futures::StreamExt;
use reqwest::{Client, StatusCode};
use reqwest_sse_stream::into_sse_stream;
use serde_json::{Map, Value, json};
use tensorzero::test_helpers::{
    make_embedded_gateway_e2e_with_unique_db, start_http_gateway_with_unique_db,
};
use tensorzero::{
    CacheParamsOptions, ClientInferenceParams, InferenceOutput, Input, InputMessage,
    InputMessageContent,
};
use tensorzero_core::cache::CacheEnabledMode;
use tensorzero_core::inference::types::usage::{ApiType, RawResponseEntry};
use tensorzero_core::inference::types::{Arguments, Role, System, Text};
use uuid::Uuid;

fn assert_raw_response_entry(entry: &RawResponseEntry) {
    assert!(
        !entry.model_inference_id.is_nil(),
        "raw_response entry should have valid model_inference_id"
    );
    assert!(
        !entry.provider_type.is_empty(),
        "raw_response entry should have non-empty provider_type"
    );
    assert!(
        matches!(
            entry.api_type,
            ApiType::ChatCompletions | ApiType::Responses | ApiType::Embeddings
        ),
        "raw_response entry should have valid api_type"
    );
    assert!(
        !entry.data.is_empty(),
        "raw_response entry data should not be empty"
    );
}

fn assert_openai_chat_response_details(entry: &Value) {
    assert!(
        entry.get("model_inference_id").is_some(),
        "raw_response entry should have model_inference_id"
    );
    assert!(
        entry.get("provider_type").is_some(),
        "raw_response entry should have provider_type"
    );
    assert!(
        entry.get("api_type").is_some(),
        "raw_response entry should have api_type"
    );
    let data = entry
        .get("data")
        .expect("raw_response entry should have data field");
    assert!(
        data.is_string(),
        "raw_response entry data should be a string"
    );
}

/// Tests cache behavior with embedded gateway (unique database for test isolation)
#[tokio::test(flavor = "multi_thread")]
async fn test_raw_response_cache_behavior_non_streaming() {
    // Use embedded gateway with unique database for test isolation
    let client = make_embedded_gateway_e2e_with_unique_db("raw_response_cache_non_streaming").await;

    // Fixed input for deterministic provider-proxy caching
    let input = "raw_response_non_streaming: What is 2+2?";

    // First request: should be a cache miss, raw_response should have entries
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
            include_raw_response: true,
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

    let first_raw_response = first_response
        .raw_response()
        .expect("raw_response should be present when include_raw_response is true");
    assert!(
        !first_raw_response.is_empty(),
        "First request (cache miss) should have at least one raw_response entry"
    );
    assert_raw_response_entry(&first_raw_response[0]);

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
            include_raw_response: true,
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

    let second_raw_response = second_response
        .raw_response()
        .expect("raw_response should be present when include_raw_response is true");
    assert!(
        second_raw_response.is_empty(),
        "Second request (cache hit) should have empty raw_response array, got {second_raw_response:?}"
    );
}

/// Tests streaming cache behavior with embedded gateway (unique database for test isolation)
#[tokio::test(flavor = "multi_thread")]
async fn test_raw_response_cache_behavior_streaming() {
    // Use embedded gateway with unique database for test isolation
    let client = make_embedded_gateway_e2e_with_unique_db("raw_response_cache_streaming").await;

    // Fixed input for deterministic provider-proxy caching
    let input = "raw_response_streaming: What is 3+3?";

    // Helper to make streaming request and collect raw_response entries
    async fn make_streaming_request(
        client: &tensorzero::Client,
        input: &str,
    ) -> (Vec<RawResponseEntry>, bool) {
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
                include_raw_response: true,
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

        let mut raw_response_entries = Vec::new();
        let mut found_raw_chunk = false;
        while let Some(chunk) = stream.next().await {
            let chunk = chunk.unwrap();

            // Check for raw_chunk
            let has_raw_chunk = match &chunk {
                tensorzero::InferenceResponseChunk::Chat(c) => c.raw_chunk.is_some(),
                tensorzero::InferenceResponseChunk::Json(j) => j.raw_chunk.is_some(),
            };
            if has_raw_chunk {
                found_raw_chunk = true;
            }

            // Check for raw_response entries
            let entries = match &chunk {
                tensorzero::InferenceResponseChunk::Chat(c) => c.raw_response.as_ref(),
                tensorzero::InferenceResponseChunk::Json(j) => j.raw_response.as_ref(),
            };
            if let Some(entries) = entries {
                raw_response_entries.extend(entries.iter().cloned());
            }
        }
        (raw_response_entries, found_raw_chunk)
    }

    // First request: should be a cache miss
    let (_first_raw_response, first_found_raw_chunk) = make_streaming_request(&client, input).await;

    // For single inference, raw_response array should be empty (no previous inferences)
    // but raw_chunk should be present
    assert!(
        first_found_raw_chunk,
        "First streaming request (cache miss) should have raw_chunk"
    );

    // Wait for cache write
    tokio::time::sleep(std::time::Duration::from_millis(500)).await;

    // Second request: should be a cache hit
    let (second_raw_response, _second_found_raw_chunk) =
        make_streaming_request(&client, input).await;

    // Cache hit should have empty raw_response array
    assert!(
        second_raw_response.is_empty(),
        "Second streaming request (cache hit) should have empty raw_response array, got {second_raw_response:?}"
    );
}

/// Tests raw_response with cache disabled using embedded gateway
#[tokio::test(flavor = "multi_thread")]
async fn test_raw_response_cache_disabled_non_streaming() {
    // Use embedded gateway with unique database for test isolation
    let client =
        make_embedded_gateway_e2e_with_unique_db("raw_response_cache_disabled_non_streaming").await;

    let input = "raw_response_cache_disabled: What is 4+4?";

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
            include_raw_response: true,
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

    let raw_response = response
        .raw_response()
        .expect("raw_response should be present when include_raw_response is true");
    assert!(
        !raw_response.is_empty(),
        "raw_response should have entries when cache is disabled (no cache hits possible)"
    );
    assert_raw_response_entry(&raw_response[0]);
}

/// Tests streaming raw_response with cache disabled using embedded gateway
#[tokio::test(flavor = "multi_thread")]
async fn test_raw_response_cache_disabled_streaming() {
    // Use embedded gateway with unique database for test isolation
    let client =
        make_embedded_gateway_e2e_with_unique_db("raw_response_cache_disabled_streaming").await;

    let input = "raw_response_cache_disabled_streaming: What is 5+5?";

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
            include_raw_response: true,
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

    let mut found_raw_chunk = false;
    while let Some(chunk) = stream.next().await {
        let chunk = chunk.unwrap();
        let has_raw_chunk = match &chunk {
            tensorzero::InferenceResponseChunk::Chat(c) => c.raw_chunk.is_some(),
            tensorzero::InferenceResponseChunk::Json(j) => j.raw_chunk.is_some(),
        };
        if has_raw_chunk {
            found_raw_chunk = true;
        }
    }

    // With cache disabled, raw_chunk should have entries (no cache hits possible)
    assert!(
        found_raw_chunk,
        "raw_chunk should be present when cache is disabled (no cache hits possible)"
    );
}

// ============================================================================
// OpenAI-Compatible API Cache Tests (with unique HTTP gateway per test)
// ============================================================================

/// Helper to make OpenAI-compatible request and get tensorzero_raw_response array
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
        "tensorzero::include_raw_response": true,
        "tensorzero::cache_options": {
            "enabled": "on",
            "max_age_s": 60
        }
    });

    let url = format!("{base_url}/openai/v1/chat/completions");

    if stream {
        let mut chunks = into_sse_stream(Client::new().post(&url).json(&payload))
            .await
            .unwrap();

        // Collect raw_response entries from all chunks
        let mut raw_response_entries: Vec<Value> = Vec::new();

        while let Some(chunk) = chunks.next().await {
            let sse = chunk.unwrap();
            let Some(data) = sse.data else { continue };
            if data == "[DONE]" {
                break;
            }

            let chunk_json: Value = serde_json::from_str(&data).unwrap();
            // Check for tensorzero_raw_response at chunk level
            if let Some(rr) = chunk_json.get("tensorzero_raw_response")
                && let Some(arr) = rr.as_array()
            {
                raw_response_entries.extend(arr.iter().cloned());
            }
        }

        raw_response_entries
    } else {
        let response = Client::new()
            .post(&url)
            .json(&payload)
            .send()
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);

        let response_json: Value = response.json().await.unwrap();

        // Get tensorzero_raw_response at response level
        let raw_response = response_json.get("tensorzero_raw_response").expect(
            "tensorzero_raw_response should be present when tensorzero::include_raw_response is true",
        );
        raw_response
            .as_array()
            .expect("tensorzero_raw_response should be an array")
            .clone()
    }
}

/// Tests OpenAI-compatible cache behavior with HTTP gateway (unique database for test isolation)
#[tokio::test(flavor = "multi_thread")]
async fn test_raw_response_cache_openai_compatible_non_streaming() {
    // Start HTTP gateway with unique database
    let (base_url, _shutdown_handle) =
        start_http_gateway_with_unique_db("raw_response_openai_cache_non_streaming").await;

    let input = "raw_response_openai_non_streaming: What is 5+5?";

    // First request: should be a cache miss
    let first_raw_response = make_openai_request_to_gateway(&base_url, input, false).await;

    assert!(
        !first_raw_response.is_empty(),
        "First request (cache miss) should have at least one tensorzero_raw_response entry"
    );
    assert_openai_chat_response_details(&first_raw_response[0]);

    // Wait for cache write
    tokio::time::sleep(std::time::Duration::from_millis(500)).await;

    // Second request: should be a cache hit
    // tensorzero_raw_response should still be present but empty (field present as [])
    let second_raw_response = make_openai_request_to_gateway(&base_url, input, false).await;

    assert!(
        second_raw_response.is_empty(),
        "Second request (cache hit) should have empty tensorzero_raw_response array, got {second_raw_response:?}"
    );
}

/// Tests OpenAI-compatible streaming cache behavior with HTTP gateway (unique database for test isolation)
#[tokio::test(flavor = "multi_thread")]
async fn test_raw_response_cache_openai_compatible_streaming() {
    // Start HTTP gateway with unique database
    let (base_url, _shutdown_handle) =
        start_http_gateway_with_unique_db("raw_response_openai_cache_streaming").await;

    let input = "raw_response_openai_streaming: What is 6+6?";

    // First request: should be a cache miss
    // For single inference streaming, raw_response array should be empty
    // (no previous inferences), but the response should succeed
    let _first_raw_response = make_openai_request_to_gateway(&base_url, input, true).await;

    // Wait for cache write
    tokio::time::sleep(std::time::Duration::from_millis(500)).await;

    // Second request: should be a cache hit
    // tensorzero_raw_response should still be present but empty
    let second_raw_response = make_openai_request_to_gateway(&base_url, input, true).await;

    assert!(
        second_raw_response.is_empty(),
        "Second streaming request (cache hit) should have empty tensorzero_raw_response array, got {second_raw_response:?}"
    );
}
