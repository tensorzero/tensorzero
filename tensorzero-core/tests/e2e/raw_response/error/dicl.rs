//! E2E tests for `raw_response` in DICL error scenarios.
//!
//! Tests DICL-specific error propagation:
//! - Embedding partial failure: first embedding provider fails, second succeeds
//! - Embedding total failure: all embedding providers fail
//! - LLM failure: embedding succeeds, LLM fails

use futures::StreamExt;
use reqwest::{Client, StatusCode};
use reqwest_sse_stream::{Event, RequestBuilderExt};
use serde_json::{Value, json};
use uuid::Uuid;

use super::assert_error_raw_response_entry;
use crate::common::get_gateway_endpoint;
use crate::raw_response::assert_raw_response_entry;

// =============================================================================
// Embedding Partial Failure (first provider fails, second succeeds)
// =============================================================================

/// DICL with embedding fallback: first embedding provider errors with raw_response,
/// second succeeds. LLM also succeeds.
/// Expects raw_response with: 1 embedding error + 1 embedding success + 1 LLM success.
#[tokio::test(flavor = "multi_thread")]
async fn test_dicl_embedding_partial_fail_non_streaming() {
    let episode_id = Uuid::now_v7();
    let payload = json!({
        "function_name": "basic_test",
        "variant_name": "dicl_embedding_partial_fail",
        "episode_id": episode_id,
        "input": {
            "system": {"assistant_name": "TestBot"},
            "messages": [{"role": "user", "content": "Hello"}]
        },
        "stream": false,
        "include_raw_response": true
    });

    let response = Client::new()
        .post(get_gateway_endpoint("/inference"))
        .json(&payload)
        .send()
        .await
        .unwrap();

    assert_eq!(
        response.status(),
        StatusCode::OK,
        "Response should be successful (embedding fallback + LLM succeed)"
    );

    let response_json: Value = response.json().await.unwrap();

    let raw_response = response_json
        .get("raw_response")
        .expect("Response should have raw_response when include_raw_response=true");
    let raw_response_array = raw_response.as_array().unwrap();
    assert_eq!(
        raw_response_array.len(),
        3,
        "raw_response should have 3 entries: 1 embedding error + 1 embedding success + 1 LLM success, got {raw_response_array:#?}"
    );

    // First entry: failed embedding provider (error entry — no model_inference_id)
    let error_entry = &raw_response_array[0];
    assert_error_raw_response_entry(error_entry);
    assert_eq!(
        error_entry["provider_type"].as_str().unwrap(),
        "dummy",
        "Failed embedding provider type should be `dummy`"
    );
    assert_eq!(
        error_entry["api_type"].as_str().unwrap(),
        "embeddings",
        "Failed embedding entry should have api_type `embeddings`"
    );
    assert_eq!(
        error_entry["data"].as_str().unwrap(),
        "dummy error raw response",
        "Failed embedding provider data should be `dummy error raw response`"
    );

    // Second entry: successful embedding provider
    let embedding_success = &raw_response_array[1];
    assert_raw_response_entry(embedding_success);
    assert_eq!(
        embedding_success["api_type"].as_str().unwrap(),
        "embeddings",
        "Second entry should have api_type `embeddings`"
    );

    // Third entry: successful LLM
    let llm_success = &raw_response_array[2];
    assert_raw_response_entry(llm_success);
    assert_eq!(
        llm_success["api_type"].as_str().unwrap(),
        "chat_completions",
        "Third entry should have api_type `chat_completions`"
    );
}

/// Streaming version of embedding partial failure test.
#[tokio::test(flavor = "multi_thread")]
async fn test_dicl_embedding_partial_fail_streaming() {
    let episode_id = Uuid::now_v7();
    let payload = json!({
        "function_name": "basic_test",
        "variant_name": "dicl_embedding_partial_fail",
        "episode_id": episode_id,
        "input": {
            "system": {"assistant_name": "TestBot"},
            "messages": [{"role": "user", "content": "Hello"}]
        },
        "stream": true,
        "include_raw_response": true
    });

    let mut chunks = Client::new()
        .post(get_gateway_endpoint("/inference"))
        .json(&payload)
        .eventsource()
        .await
        .expect("Failed to create eventsource");

    let mut found_raw_response = false;
    let mut found_raw_chunk = false;
    let mut embedding_error_count = 0;
    let mut embedding_success_count = 0;

    while let Some(chunk) = chunks.next().await {
        let chunk = chunk.expect("Failed to receive chunk");
        let Event::Message(chunk) = chunk else {
            continue;
        };
        if chunk.data == "[DONE]" {
            break;
        }

        let chunk_json: Value =
            serde_json::from_str(&chunk.data).expect("Failed to parse chunk JSON");

        // Check for raw_response (embedding entries from previous model inferences)
        if let Some(raw_response) = chunk_json.get("raw_response") {
            found_raw_response = true;
            if let Some(arr) = raw_response.as_array() {
                for entry in arr {
                    let api_type = entry["api_type"].as_str().unwrap();
                    if api_type == "embeddings" {
                        if entry.get("model_inference_id").is_none() {
                            embedding_error_count += 1;
                        } else {
                            embedding_success_count += 1;
                        }
                    }
                }
            }
        }

        // Check for raw_chunk (LLM streaming data)
        if chunk_json.get("raw_chunk").is_some() {
            found_raw_chunk = true;
        }
    }

    assert!(
        found_raw_response,
        "Streaming response should include raw_response for embedding inferences"
    );
    assert!(
        found_raw_chunk,
        "Streaming response should include raw_chunk for LLM streaming"
    );
    assert_eq!(
        embedding_error_count, 1,
        "Should have 1 embedding error entry in raw_response"
    );
    assert_eq!(
        embedding_success_count, 1,
        "Should have 1 embedding success entry in raw_response"
    );
}

// =============================================================================
// Embedding Total Failure (all embedding providers fail)
// =============================================================================

/// DICL where all embedding providers fail.
/// Expects error response with raw_response entries from both failed embedding providers.
#[tokio::test(flavor = "multi_thread")]
async fn test_dicl_embedding_all_fail_non_streaming() {
    let episode_id = Uuid::now_v7();
    let payload = json!({
        "function_name": "basic_test",
        "variant_name": "dicl_embedding_all_fail",
        "episode_id": episode_id,
        "input": {
            "system": {"assistant_name": "TestBot"},
            "messages": [{"role": "user", "content": "Hello"}]
        },
        "stream": false,
        "include_raw_response": true
    });

    let response = Client::new()
        .post(get_gateway_endpoint("/inference"))
        .json(&payload)
        .send()
        .await
        .unwrap();

    assert!(
        !response.status().is_success(),
        "Response should be an error (all embedding providers failed)"
    );

    let response_json: Value = response.json().await.unwrap();

    let raw_response = response_json
        .get("raw_response")
        .expect("Error response should have raw_response when include_raw_response=true");
    let raw_response_array = raw_response.as_array().unwrap();
    assert_eq!(
        raw_response_array.len(),
        2,
        "raw_response should have 2 entries from 2 failed embedding providers"
    );

    for entry in raw_response_array {
        assert_error_raw_response_entry(entry);
        assert_eq!(
            entry["provider_type"].as_str().unwrap(),
            "dummy",
            "Failed provider type should be `dummy`"
        );
        assert_eq!(
            entry["api_type"].as_str().unwrap(),
            "embeddings",
            "Failed entries should have api_type `embeddings`"
        );
        assert_eq!(
            entry["data"].as_str().unwrap(),
            "dummy error raw response",
            "Failed provider data should be `dummy error raw response`"
        );
    }
}

/// Streaming version of all-embedding-fail test.
/// Pre-stream error: returns non-200 HTTP response (not SSE).
#[tokio::test(flavor = "multi_thread")]
async fn test_dicl_embedding_all_fail_streaming() {
    let episode_id = Uuid::now_v7();
    let payload = json!({
        "function_name": "basic_test",
        "variant_name": "dicl_embedding_all_fail",
        "episode_id": episode_id,
        "input": {
            "system": {"assistant_name": "TestBot"},
            "messages": [{"role": "user", "content": "Hello"}]
        },
        "stream": true,
        "include_raw_response": true
    });

    let response = Client::new()
        .post(get_gateway_endpoint("/inference"))
        .json(&payload)
        .send()
        .await
        .unwrap();

    assert!(
        !response.status().is_success(),
        "Response should be an error (all embedding providers failed)"
    );

    let response_json: Value = response.json().await.unwrap();

    let raw_response = response_json
        .get("raw_response")
        .expect("Error response should have raw_response when include_raw_response=true");
    let raw_response_array = raw_response.as_array().unwrap();
    assert_eq!(
        raw_response_array.len(),
        2,
        "raw_response should have 2 entries from 2 failed embedding providers"
    );

    for entry in raw_response_array {
        assert_error_raw_response_entry(entry);
        assert_eq!(
            entry["api_type"].as_str().unwrap(),
            "embeddings",
            "Failed entries should have api_type `embeddings`"
        );
    }
}

// =============================================================================
// LLM Failure (embedding succeeds, LLM fails)
// =============================================================================

/// DICL where embedding succeeds but LLM fails.
/// Expects error response with raw_response entry from the failed LLM provider.
#[tokio::test(flavor = "multi_thread")]
async fn test_dicl_llm_fail_non_streaming() {
    let episode_id = Uuid::now_v7();
    let payload = json!({
        "function_name": "basic_test",
        "variant_name": "dicl_llm_fail",
        "episode_id": episode_id,
        "input": {
            "system": {"assistant_name": "TestBot"},
            "messages": [{"role": "user", "content": "Hello"}]
        },
        "stream": false,
        "include_raw_response": true
    });

    let response = Client::new()
        .post(get_gateway_endpoint("/inference"))
        .json(&payload)
        .send()
        .await
        .unwrap();

    assert!(
        !response.status().is_success(),
        "Response should be an error (LLM provider failed)"
    );

    let response_json: Value = response.json().await.unwrap();

    let raw_response = response_json
        .get("raw_response")
        .expect("Error response should have raw_response when include_raw_response=true");
    let raw_response_array = raw_response.as_array().unwrap();
    assert_eq!(
        raw_response_array.len(),
        1,
        "raw_response should have 1 entry from the failed LLM provider"
    );

    let entry = &raw_response_array[0];
    assert_error_raw_response_entry(entry);
    assert_eq!(
        entry["provider_type"].as_str().unwrap(),
        "dummy",
        "Failed provider type should be `dummy`"
    );
    assert_eq!(
        entry["api_type"].as_str().unwrap(),
        "chat_completions",
        "Failed LLM entry should have api_type `chat_completions`"
    );
    assert_eq!(
        entry["data"].as_str().unwrap(),
        "dummy error raw response",
        "Failed LLM provider data should be `dummy error raw response`"
    );
}

/// Streaming version of LLM failure test.
/// Pre-stream error: returns non-200 HTTP response (not SSE).
#[tokio::test(flavor = "multi_thread")]
async fn test_dicl_llm_fail_streaming() {
    let episode_id = Uuid::now_v7();
    let payload = json!({
        "function_name": "basic_test",
        "variant_name": "dicl_llm_fail",
        "episode_id": episode_id,
        "input": {
            "system": {"assistant_name": "TestBot"},
            "messages": [{"role": "user", "content": "Hello"}]
        },
        "stream": true,
        "include_raw_response": true
    });

    let response = Client::new()
        .post(get_gateway_endpoint("/inference"))
        .json(&payload)
        .send()
        .await
        .unwrap();

    assert!(
        !response.status().is_success(),
        "Response should be an error (LLM provider failed)"
    );

    let response_json: Value = response.json().await.unwrap();

    let raw_response = response_json
        .get("raw_response")
        .expect("Error response should have raw_response when include_raw_response=true");
    let raw_response_array = raw_response.as_array().unwrap();
    assert_eq!(
        raw_response_array.len(),
        1,
        "raw_response should have 1 entry from the failed LLM provider"
    );

    let entry = &raw_response_array[0];
    assert_error_raw_response_entry(entry);
    assert_eq!(
        entry["api_type"].as_str().unwrap(),
        "chat_completions",
        "Failed LLM entry should have api_type `chat_completions`"
    );
}
