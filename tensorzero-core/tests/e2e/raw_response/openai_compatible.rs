//! E2E tests for `tensorzero::include_raw_response` parameter in the OpenAI-compatible API.
//!
//! Tests that raw provider-specific response data is correctly returned when requested
//! via the OpenAI-compatible endpoint.
//!
//! These tests use HTTP gateways with unique databases for test isolation.

use futures::StreamExt;
use reqwest::{Client, StatusCode};
use reqwest_sse_stream::{Event, RequestBuilderExt};
use serde_json::{Value, json};
use tensorzero::test_helpers::make_http_gateway_with_unique_db;
use uuid::Uuid;

fn assert_raw_response_entry_structure(entry: &Value) {
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

    // Verify api_type is valid
    let api_type = entry.get("api_type").unwrap().as_str().unwrap();
    assert!(
        ["chat_completions", "responses", "embeddings"].contains(&api_type),
        "api_type should be 'chat_completions', 'responses', or 'embeddings', got: {api_type}"
    );

    // Verify data is a string (raw response)
    let data = entry
        .get("data")
        .expect("raw_response entry should have data");
    assert!(
        data.is_string(),
        "raw_response entry data should be a string, got: {data:?}"
    );
}

/// Test that tensorzero::include_raw_response returns tensorzero_raw_response in non-streaming response
#[tokio::test(flavor = "multi_thread")]
async fn test_openai_compatible_raw_response_non_streaming() {
    let (base_url, _shutdown_handle) =
        make_http_gateway_with_unique_db("raw_response_openai_non_streaming").await;
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
        "tensorzero::include_raw_response": true
    });

    let response = client
        .post(format!("{base_url}/openai/v1/chat/completions"))
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

    // Check tensorzero_raw_response exists at response level
    let raw_response = response_json.get("tensorzero_raw_response").expect(
        "Response should have tensorzero_raw_response when tensorzero::include_raw_response=true",
    );
    assert!(
        raw_response.is_array(),
        "tensorzero_raw_response should be an array"
    );

    let raw_response_array = raw_response.as_array().unwrap();
    assert!(
        !raw_response_array.is_empty(),
        "tensorzero_raw_response should have at least one entry"
    );

    // Validate first entry structure
    let first_entry = &raw_response_array[0];
    assert_raw_response_entry_structure(first_entry);

    // For OpenAI chat completions, api_type should be "chat_completions"
    let api_type = first_entry.get("api_type").unwrap().as_str().unwrap();
    assert_eq!(
        api_type, "chat_completions",
        "OpenAI chat completions should have api_type 'chat_completions'"
    );
}

/// Test that tensorzero_raw_response is NOT returned when tensorzero::include_raw_response is false
#[tokio::test(flavor = "multi_thread")]
async fn test_openai_compatible_raw_response_not_requested() {
    let (base_url, _shutdown_handle) =
        make_http_gateway_with_unique_db("raw_response_openai_not_requested").await;
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
        "tensorzero::episode_id": episode_id.to_string()
        // Note: tensorzero::include_raw_response is NOT set (defaults to false)
    });

    let response = client
        .post(format!("{base_url}/openai/v1/chat/completions"))
        .json(&payload)
        .send()
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);

    let response_json: Value = response.json().await.unwrap();

    // tensorzero_raw_response should NOT be present
    assert!(
        response_json.get("tensorzero_raw_response").is_none(),
        "tensorzero_raw_response should not be present when not requested"
    );
}

/// Test that tensorzero::include_raw_response returns tensorzero_raw_chunk in streaming response
#[tokio::test(flavor = "multi_thread")]
async fn test_openai_compatible_raw_response_streaming() {
    let (base_url, _shutdown_handle) =
        make_http_gateway_with_unique_db("raw_response_openai_streaming").await;
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
        "tensorzero::episode_id": episode_id.to_string(),
        "tensorzero::include_raw_response": true
    });

    let mut chunks = client
        .post(format!("{base_url}/openai/v1/chat/completions"))
        .json(&payload)
        .eventsource()
        .await
        .expect("Failed to create eventsource for streaming request");

    let mut found_raw_chunk = false;
    let mut content_chunks_count: usize = 0;
    let mut chunks_with_raw_chunk: usize = 0;
    let mut all_chunks: Vec<Value> = Vec::new();

    while let Some(chunk) = chunks.next().await {
        let chunk = chunk.expect("Failed to receive chunk from stream");
        let Event::Message(chunk) = chunk else {
            continue;
        };
        if chunk.data == "[DONE]" {
            break;
        }

        let chunk_json: Value =
            serde_json::from_str(&chunk.data).expect("Failed to parse chunk as JSON");

        all_chunks.push(chunk_json.clone());

        // Count content chunks (chunks with delta content)
        if chunk_json.get("choices").is_some() {
            content_chunks_count += 1;
        }

        // Check for tensorzero_raw_chunk
        if let Some(raw_chunk) = chunk_json.get("tensorzero_raw_chunk") {
            found_raw_chunk = true;
            chunks_with_raw_chunk += 1;
            assert!(
                raw_chunk.is_string(),
                "tensorzero_raw_chunk should be a string, got: {raw_chunk:?}"
            );
        }
    }

    assert!(
        found_raw_chunk,
        "Streaming response should include tensorzero_raw_chunk in at least one chunk.\n\
        Total chunks received: {}\n\
        Content chunks: {}\n\
        Last few chunks:\n{:#?}",
        all_chunks.len(),
        content_chunks_count,
        all_chunks.iter().rev().take(3).collect::<Vec<_>>()
    );

    // Most content chunks should have raw_chunk (allow first/last to not have it)
    assert!(
        chunks_with_raw_chunk >= content_chunks_count.saturating_sub(2),
        "Most content chunks should have tensorzero_raw_chunk: {chunks_with_raw_chunk} of {content_chunks_count}"
    );
}

/// Test that tensorzero_raw_chunk is NOT returned when tensorzero::include_raw_response is false in streaming
#[tokio::test(flavor = "multi_thread")]
async fn test_openai_compatible_raw_response_streaming_not_requested() {
    let (base_url, _shutdown_handle) =
        make_http_gateway_with_unique_db("raw_response_openai_streaming_not_requested").await;
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
        "tensorzero::episode_id": episode_id.to_string()
        // Note: tensorzero::include_raw_response is NOT set (defaults to false)
    });

    let mut chunks = client
        .post(format!("{base_url}/openai/v1/chat/completions"))
        .json(&payload)
        .eventsource()
        .await
        .expect("Failed to create eventsource for streaming request");

    while let Some(chunk) = chunks.next().await {
        let chunk = chunk.expect("Failed to receive chunk from stream");
        let Event::Message(chunk) = chunk else {
            continue;
        };
        if chunk.data == "[DONE]" {
            break;
        }

        let chunk_json: Value =
            serde_json::from_str(&chunk.data).expect("Failed to parse chunk as JSON");

        // tensorzero_raw_chunk should NOT be present
        assert!(
            chunk_json.get("tensorzero_raw_chunk").is_none(),
            "tensorzero_raw_chunk should not be present when not requested"
        );

        // tensorzero_raw_response should also NOT be present
        assert!(
            chunk_json.get("tensorzero_raw_response").is_none(),
            "tensorzero_raw_response should not be present when not requested"
        );
    }
}

// =============================================================================
// Error Tests with raw_response
// =============================================================================

/// Test that errors include raw_response when tensorzero::include_raw_response is true (non-streaming)
#[tokio::test(flavor = "multi_thread")]
async fn test_openai_compatible_raw_response_error_non_streaming() {
    let (base_url, _shutdown_handle) =
        make_http_gateway_with_unique_db("raw_response_openai_error_non_streaming").await;
    let client = Client::new();
    let episode_id = Uuid::now_v7();

    // Use error_with_raw_response model via shorthand
    let payload = json!({
        "model": "tensorzero::model_name::dummy::error_with_raw_response",
        "messages": [{"role": "user", "content": "Hello"}],
        "stream": false,
        "tensorzero::episode_id": episode_id.to_string(),
        "tensorzero::include_raw_response": true
    });

    let response = client
        .post(format!("{base_url}/openai/v1/chat/completions"))
        .json(&payload)
        .send()
        .await
        .unwrap();

    // Should be an error status
    let status = response.status();
    assert!(
        !status.is_success(),
        "Response should be an error, got status: {status}"
    );

    let body: Value = response.json().await.unwrap();

    // Should have raw_response in error response
    let raw_response = body
        .get("raw_response")
        .expect("Error should include raw_response when tensorzero::include_raw_response=true");
    assert!(raw_response.is_array(), "raw_response should be an array");

    let entries = raw_response.as_array().unwrap();
    assert!(
        !entries.is_empty(),
        "Should have at least one raw_response entry"
    );

    // Verify entry contains error data
    let entry = &entries[0];
    assert_eq!(
        entry.get("provider_type").and_then(|v| v.as_str()),
        Some("dummy"),
        "Provider type should be 'dummy'"
    );
    assert_eq!(
        entry.get("api_type").and_then(|v| v.as_str()),
        Some("chat_completions"),
        "API type should be 'chat_completions'"
    );

    let data = entry.get("data").and_then(|d| d.as_str()).unwrap();
    assert!(
        data.contains("test_error"),
        "raw_response data should contain error info"
    );
}

/// Test that errors include raw_response when tensorzero::include_raw_response is true (streaming)
#[tokio::test(flavor = "multi_thread")]
async fn test_openai_compatible_raw_response_error_streaming() {
    let (base_url, _shutdown_handle) =
        make_http_gateway_with_unique_db("raw_response_openai_error_streaming").await;
    let client = Client::new();
    let episode_id = Uuid::now_v7();

    // Use error_with_raw_response model - errors happen before streaming starts
    // so we get a regular HTTP error response, not SSE
    let payload = json!({
        "model": "tensorzero::model_name::dummy::error_with_raw_response",
        "messages": [{"role": "user", "content": "Hello"}],
        "stream": true,
        "tensorzero::episode_id": episode_id.to_string(),
        "tensorzero::include_raw_response": true
    });

    let response = client
        .post(format!("{base_url}/openai/v1/chat/completions"))
        .json(&payload)
        .send()
        .await
        .unwrap();

    // Should be an error status (not SSE)
    let status = response.status();
    assert!(
        !status.is_success(),
        "Response should be an error, got status: {status}"
    );

    let body: Value = response.json().await.unwrap();

    // raw_response should be included in the error
    let raw_response = body
        .get("raw_response")
        .expect("Streaming error should include raw_response");
    assert!(raw_response.is_array(), "raw_response should be an array");

    let entries = raw_response.as_array().unwrap();
    assert!(
        !entries.is_empty(),
        "Error should have raw_response entries"
    );

    // Verify provider is dummy
    let entry = &entries[0];
    assert_eq!(
        entry.get("provider_type").and_then(|v| v.as_str()),
        Some("dummy"),
        "Provider type should be 'dummy'"
    );
}
