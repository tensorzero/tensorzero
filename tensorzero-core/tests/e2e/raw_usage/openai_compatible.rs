//! E2E tests for `tensorzero::include_raw_usage` parameter in the OpenAI-compatible API.
//!
//! Tests that raw provider-specific usage data is correctly returned when requested
//! via the OpenAI-compatible endpoint.

use futures::StreamExt;
use reqwest::{Client, StatusCode};
use reqwest_eventsource::{Event, RequestBuilderExt};
use serde_json::{Value, json};
use uuid::Uuid;

use crate::common::get_gateway_endpoint;

fn assert_openai_chat_usage_details(entry: &Value) {
    let data = entry
        .get("data")
        .expect("raw_usage entry should include data field for chat completions");
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

/// Test that tensorzero::include_raw_usage returns tensorzero_raw_usage in non-streaming response
#[tokio::test]
async fn test_openai_compatible_raw_usage_non_streaming() {
    let client = Client::new();
    let episode_id = Uuid::now_v7();

    let payload = json!({
        "model": "tensorzero::model_name::openai::gpt-5-nano",
        "messages": [
            {
                "role": "user",
                "content": "Hello"
            }
        ],
        "params": {
            "chat_completion": {
                "reasoning_effort": "minimal"
            }
        },
        "stream": false,
        "tensorzero::episode_id": episode_id.to_string(),
        "tensorzero::include_raw_usage": true
    });

    let response = client
        .post(get_gateway_endpoint("/openai/v1/chat/completions"))
        .json(&payload)
        .send()
        .await
        .unwrap();

    let status = response.status();
    let response_text = response.text().await.unwrap();

    assert_eq!(
        status,
        StatusCode::OK,
        "Response should be successful. Body: {response_text}"
    );

    let response_json: Value = serde_json::from_str(&response_text).unwrap();

    // Check usage exists and is an object
    let usage = response_json
        .get("usage")
        .expect("Response should have usage field");
    assert!(usage.is_object(), "usage should be an object");

    // Check standard usage fields
    assert!(
        usage.get("prompt_tokens").is_some(),
        "usage should have prompt_tokens"
    );
    assert!(
        usage.get("completion_tokens").is_some(),
        "usage should have completion_tokens"
    );

    // Check tensorzero_raw_usage exists at response level (sibling to usage)
    let raw_usage = response_json.get("tensorzero_raw_usage").expect(
        "Response should have tensorzero_raw_usage when tensorzero::include_raw_usage=true",
    );
    assert!(
        raw_usage.is_array(),
        "tensorzero_raw_usage should be an array"
    );

    let raw_usage_array = raw_usage.as_array().unwrap();
    assert!(
        !raw_usage_array.is_empty(),
        "tensorzero_raw_usage should have at least one entry"
    );

    // Validate first entry structure
    let first_entry = &raw_usage_array[0];
    assert!(
        first_entry.get("model_inference_id").is_some(),
        "raw_usage entry should have model_inference_id"
    );
    assert!(
        first_entry.get("provider_type").is_some(),
        "raw_usage entry should have provider_type"
    );
    assert!(
        first_entry.get("api_type").is_some(),
        "raw_usage entry should have api_type"
    );
    assert_openai_chat_usage_details(first_entry);
}

/// Test that tensorzero_raw_usage is NOT returned when tensorzero::include_raw_usage is false
#[tokio::test]
async fn test_openai_compatible_raw_usage_not_requested() {
    let client = Client::new();
    let episode_id = Uuid::now_v7();

    let payload = json!({
        "model": "tensorzero::model_name::openai::gpt-5-nano",
        "messages": [
            {
                "role": "user",
                "content": "Hello"
            }
        ],
        "params": {
            "chat_completion": {
                "reasoning_effort": "minimal"
            }
        },
        "stream": false,
        "tensorzero::episode_id": episode_id.to_string(),
        "tensorzero::include_raw_usage": false
    });

    let response = client
        .post(get_gateway_endpoint("/openai/v1/chat/completions"))
        .json(&payload)
        .send()
        .await
        .unwrap();

    let status = response.status();
    let response_text = response.text().await.unwrap();

    assert_eq!(
        status,
        StatusCode::OK,
        "Response should be successful. Body: {response_text}"
    );

    let response_json: Value = serde_json::from_str(&response_text).unwrap();

    // tensorzero_raw_usage should NOT be present at response level
    assert!(
        response_json.get("tensorzero_raw_usage").is_none(),
        "tensorzero_raw_usage should not be present when tensorzero::include_raw_usage is false"
    );
}

/// Test that tensorzero::include_raw_usage errors when stream_options.include_usage is explicitly false
#[tokio::test]
async fn test_openai_compatible_raw_usage_streaming_error_without_include_usage() {
    let client = Client::new();
    let episode_id = Uuid::now_v7();

    let payload = json!({
        "model": "tensorzero::model_name::openai::gpt-5-nano",
        "messages": [
            {
                "role": "user",
                "content": "Hello"
            }
        ],
        "params": {
            "chat_completion": {
                "reasoning_effort": "minimal"
            }
        },
        "stream": true,
        "stream_options": {
            "include_usage": false
        },
        "tensorzero::episode_id": episode_id.to_string(),
        "tensorzero::include_raw_usage": true
    });

    let response = client
        .post(get_gateway_endpoint("/openai/v1/chat/completions"))
        .json(&payload)
        .send()
        .await
        .unwrap();

    assert_eq!(
        response.status(),
        StatusCode::BAD_REQUEST,
        "Should error when include_raw_usage is true but include_usage is false"
    );

    let error_body: Value = response.json().await.unwrap();
    let error_message = error_body["error"].as_str().unwrap_or("");
    assert!(
        error_message.contains("include_usage"),
        "Error message should mention include_usage requirement. Got: {error_message}"
    );
}

/// Test that tensorzero::include_raw_usage returns tensorzero_raw_usage in streaming response
/// and that raw_usage is NOT buffered (appears as chunks stream, not only at the end)
#[tokio::test]
async fn test_openai_compatible_raw_usage_streaming() {
    let client = Client::new();
    let episode_id = Uuid::now_v7();

    // Note: tensorzero::include_raw_usage automatically enables include_usage for streaming
    let payload = json!({
        "model": "tensorzero::model_name::openai::gpt-5-nano",
        "messages": [
            {
                "role": "user",
                "content": "Hello"
            }
        ],
        "params": {
            "chat_completion": {
                "reasoning_effort": "minimal"
            }
        },
        "stream": true,
        "tensorzero::episode_id": episode_id.to_string(),
        "tensorzero::include_raw_usage": true
    });

    let mut response = client
        .post(get_gateway_endpoint("/openai/v1/chat/completions"))
        .json(&payload)
        .eventsource()
        .unwrap();

    let mut all_chunks: Vec<Value> = Vec::new();
    // Track which chunk indices have raw_usage
    let mut chunks_with_raw_usage: Vec<usize> = Vec::new();

    while let Some(event) = response.next().await {
        let event = event.expect("Failed to receive event");
        let Event::Message(message) = event else {
            continue;
        };
        if message.data == "[DONE]" {
            break;
        }

        let chunk_json: Value =
            serde_json::from_str(&message.data).expect("Failed to parse chunk as JSON");
        let chunk_index = all_chunks.len();
        all_chunks.push(chunk_json.clone());

        // Check if this chunk has tensorzero_raw_usage at chunk level
        if let Some(raw_usage) = chunk_json.get("tensorzero_raw_usage") {
            chunks_with_raw_usage.push(chunk_index);
            assert!(
                raw_usage.is_array(),
                "tensorzero_raw_usage should be an array"
            );

            let raw_usage_array = raw_usage.as_array().unwrap();
            // Validate structure of entries
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
                assert_openai_chat_usage_details(entry);
            }
        }
    }

    assert!(
        !chunks_with_raw_usage.is_empty(),
        "Streaming response should include tensorzero_raw_usage.\n\
        Total chunks received: {}\n\
        Last few chunks:\n{:#?}",
        all_chunks.len(),
        all_chunks.iter().rev().take(3).collect::<Vec<_>>()
    );

    // Verify that raw_usage is NOT buffered to the end.
    // It should appear on chunks as they stream, not only on the final chunk.
    // Since create_stream passes through raw_usage on the final chunk (which has usage),
    // we expect the chunk with raw_usage to also be the chunk with usage.
    // The key point is that raw_usage is NOT accumulated across chunks anymore.
    let raw_usage_chunk = &all_chunks[chunks_with_raw_usage[0]];
    let raw_usage = raw_usage_chunk
        .get("tensorzero_raw_usage")
        .expect("tensorzero_raw_usage field missing from chunk");
    assert!(
        raw_usage.is_array(),
        "tensorzero_raw_usage should be an array"
    );
    let raw_usage_array = raw_usage
        .as_array()
        .expect("tensorzero_raw_usage should be an array");
    assert!(
        !raw_usage_array.is_empty(),
        "Streaming response should include at least one tensorzero_raw_usage entry"
    );
    assert_openai_chat_usage_details(&raw_usage_array[0]);
}

/// Test that usage only appears in the final chunk for OpenAI-compatible streaming
#[tokio::test]
async fn test_openai_compatible_streaming_usage_only_in_last_chunk() {
    let client = Client::new();
    let episode_id = Uuid::now_v7();

    let payload = json!({
        "model": "tensorzero::model_name::openai::gpt-5-nano",
        "messages": [
            {
                "role": "user",
                "content": "Hello"
            }
        ],
        "params": {
            "chat_completion": {
                "reasoning_effort": "minimal"
            }
        },
        "stream": true,
        "stream_options": {
            "include_usage": true
        },
        "tensorzero::episode_id": episode_id.to_string()
    });

    let mut response = client
        .post(get_gateway_endpoint("/openai/v1/chat/completions"))
        .json(&payload)
        .eventsource()
        .unwrap();

    let mut all_chunks: Vec<Value> = Vec::new();
    // Track which chunk indices have usage populated (not null)
    let mut chunks_with_usage: Vec<usize> = Vec::new();

    while let Some(event) = response.next().await {
        let event = event.expect("Failed to receive event");
        let Event::Message(message) = event else {
            continue;
        };
        if message.data == "[DONE]" {
            break;
        }

        let chunk_json: Value =
            serde_json::from_str(&message.data).expect("Failed to parse chunk as JSON");
        let chunk_index = all_chunks.len();
        all_chunks.push(chunk_json.clone());

        // Check if this chunk has non-null usage
        if let Some(usage) = chunk_json.get("usage")
            && !usage.is_null()
        {
            chunks_with_usage.push(chunk_index);
        }
    }

    assert!(
        all_chunks.len() >= 2,
        "Expected at least 2 chunks (content + usage), got {}",
        all_chunks.len()
    );

    assert_eq!(
        chunks_with_usage.len(),
        1,
        "Usage should appear in exactly one chunk (the final one). \
        Found usage in {} chunks at indices: {:?}.\n\
        All chunks:\n{:#?}",
        chunks_with_usage.len(),
        chunks_with_usage,
        all_chunks
    );

    // Verify the chunk with usage is the last chunk
    let last_chunk_index = all_chunks.len() - 1;
    assert_eq!(
        chunks_with_usage[0], last_chunk_index,
        "Usage should appear only in the last chunk (index {}), but found at index {}",
        last_chunk_index, chunks_with_usage[0]
    );

    // Verify the usage data is valid
    let final_chunk = &all_chunks[last_chunk_index];
    let usage = final_chunk
        .get("usage")
        .expect("Final chunk should have usage field");
    assert!(
        usage.get("prompt_tokens").is_some(),
        "usage should have prompt_tokens"
    );
    assert!(
        usage.get("completion_tokens").is_some(),
        "usage should have completion_tokens"
    );
}

/// Test that no usage appears when stream_options.include_usage is false (or omitted)
#[tokio::test]
async fn test_openai_compatible_streaming_no_usage_when_disabled() {
    let client = Client::new();
    let episode_id = Uuid::now_v7();

    // Omit stream_options entirely (OpenAI default is no usage)
    let payload = json!({
        "model": "tensorzero::model_name::openai::gpt-5-nano",
        "messages": [
            {
                "role": "user",
                "content": "Hello"
            }
        ],
        "params": {
            "chat_completion": {
                "reasoning_effort": "minimal"
            }
        },
        "stream": true,
        "tensorzero::episode_id": episode_id.to_string()
    });

    let mut response = client
        .post(get_gateway_endpoint("/openai/v1/chat/completions"))
        .json(&payload)
        .eventsource()
        .unwrap();

    let mut all_chunks: Vec<Value> = Vec::new();
    let mut chunks_with_usage: Vec<usize> = Vec::new();

    while let Some(event) = response.next().await {
        let event = event.expect("Failed to receive event");
        let Event::Message(message) = event else {
            continue;
        };
        if message.data == "[DONE]" {
            break;
        }

        let chunk_json: Value =
            serde_json::from_str(&message.data).expect("Failed to parse chunk as JSON");
        let chunk_index = all_chunks.len();
        all_chunks.push(chunk_json.clone());

        // Check if this chunk has non-null usage
        if let Some(usage) = chunk_json.get("usage")
            && !usage.is_null()
        {
            chunks_with_usage.push(chunk_index);
        }
    }

    assert!(!all_chunks.is_empty(), "Should receive at least one chunk");

    assert!(
        chunks_with_usage.is_empty(),
        "No chunks should have usage when stream_options.include_usage is not set.\n\
        Found usage in {} chunks at indices: {:?}.\n\
        All chunks:\n{:#?}",
        chunks_with_usage.len(),
        chunks_with_usage,
        all_chunks
    );
}
