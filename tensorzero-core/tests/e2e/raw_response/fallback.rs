//! E2E tests for `raw_response` in fallback scenarios.
//!
//! Tests that raw provider-specific response data from failed providers/variants
//! is correctly included in success responses when `include_raw_response=true`.
//!
//! Organized by fallback level:
//! - Model fallback: A model has multiple providers, first provider(s) fail, later one succeeds
//! - Variant fallback: A function has multiple variants, first variant(s) fail, later one succeeds

use futures::StreamExt;
use reqwest::{Client, StatusCode};
use reqwest_sse_stream::{Event, RequestBuilderExt};
use serde_json::{Value, json};
use uuid::Uuid;

use super::assert_raw_response_entry;
use super::error::assert_error_raw_response_entry;
use crate::common::get_gateway_endpoint;

// =============================================================================
// Model Fallback Tests
// =============================================================================

/// Model fallback: error provider (with raw response) -> good provider, non-streaming.
/// Expects `raw_response` to contain entries from both the failed and successful providers.
#[tokio::test]
async fn test_model_fallback_fail_success_non_streaming() {
    let episode_id = Uuid::now_v7();
    let payload = json!({
        "function_name": "basic_test",
        "variant_name": "model_fallback_with_raw_response",
        "episode_id": episode_id,
        "input": {
            "system": {"assistant_name": "AskJeeves"},
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
        "Response should be successful (fallback to good provider)"
    );

    let response_json: Value = response.json().await.unwrap();

    let raw_response = response_json
        .get("raw_response")
        .expect("Response should have raw_response when include_raw_response=true");
    let raw_response_array = raw_response.as_array().unwrap();
    assert_eq!(
        raw_response_array.len(),
        2,
        "raw_response should have 2 entries: 1 from failed provider + 1 from successful provider"
    );

    // First entry: failed provider (error entry — no model_inference_id)
    let error_entry = &raw_response_array[0];
    assert_error_raw_response_entry(error_entry);
    assert_eq!(
        error_entry.get("provider_type").unwrap().as_str().unwrap(),
        "dummy",
        "Failed provider type should be 'dummy'"
    );
    assert_eq!(
        error_entry.get("data").unwrap().as_str().unwrap(),
        "dummy error raw response",
        "Failed provider data should be 'dummy error raw response'"
    );

    // Second entry: successful provider (success entry — has model_inference_id)
    let success_entry = &raw_response_array[1];
    assert_raw_response_entry(success_entry);
    assert_eq!(
        success_entry
            .get("provider_type")
            .unwrap()
            .as_str()
            .unwrap(),
        "dummy",
        "Successful provider_type should be the actual provider type"
    );
}

/// Model fallback: error provider (with raw response) -> good provider, streaming.
/// Expects failed provider entries in a chunk, then streaming chunks from good provider.
#[tokio::test]
async fn test_model_fallback_fail_success_streaming() {
    let episode_id = Uuid::now_v7();
    let payload = json!({
        "function_name": "basic_test",
        "variant_name": "model_fallback_with_raw_response",
        "episode_id": episode_id,
        "input": {
            "system": {"assistant_name": "AskJeeves"},
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

    let mut found_failed_raw_response = false;
    let mut found_raw_chunk = false;
    let mut chunk_count = 0;

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
        chunk_count += 1;

        // Check for raw_response entries (from failed provider)
        if let Some(raw_response) = chunk_json.get("raw_response") {
            let raw_response_array = raw_response.as_array().unwrap();
            for entry in raw_response_array {
                assert_error_raw_response_entry(entry);
                assert_eq!(
                    entry.get("data").unwrap().as_str().unwrap(),
                    "dummy error raw response",
                    "Failed provider data should be 'dummy error raw response'"
                );
            }
            found_failed_raw_response = true;
        }

        // Check for raw_chunk (from successful provider streaming)
        if chunk_json.get("raw_chunk").is_some() {
            found_raw_chunk = true;
        }
    }

    assert!(chunk_count > 0, "Should have received at least one chunk");
    assert!(
        found_failed_raw_response,
        "Should have received raw_response entries from failed provider in a streaming chunk"
    );
    assert!(
        found_raw_chunk,
        "Should have received raw_chunk from successful provider streaming"
    );
}

/// Model fallback: both providers error (with raw response), non-streaming.
/// Expects error response (non-200) with `raw_response` entries from both failed providers.
#[tokio::test]
async fn test_model_fallback_both_fail_non_streaming() {
    let episode_id = Uuid::now_v7();
    let payload = json!({
        "function_name": "basic_test",
        "variant_name": "model_both_error_with_raw_response",
        "episode_id": episode_id,
        "input": {
            "system": {"assistant_name": "AskJeeves"},
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
        response.status().is_server_error() || response.status().is_client_error(),
        "Response should be an error (all providers failed)"
    );

    let response_json: Value = response.json().await.unwrap();

    let raw_response = response_json
        .get("raw_response")
        .expect("Error response should have raw_response when include_raw_response=true");
    let raw_response_array = raw_response.as_array().unwrap();
    assert_eq!(
        raw_response_array.len(),
        2,
        "raw_response should have 2 entries from 2 failed providers"
    );

    // Both entries should be error entries
    for entry in raw_response_array {
        assert_error_raw_response_entry(entry);
        assert_eq!(
            entry.get("data").unwrap().as_str().unwrap(),
            "dummy error raw response",
            "Failed provider data should be 'dummy error raw response'"
        );
    }
}

/// Model fallback: both providers error (with raw response), streaming (pre-stream error).
/// Expects error response with `raw_response` entries from both failed providers.
#[tokio::test]
async fn test_model_fallback_both_fail_streaming() {
    let episode_id = Uuid::now_v7();
    let payload = json!({
        "function_name": "basic_test",
        "variant_name": "model_both_error_with_raw_response",
        "episode_id": episode_id,
        "input": {
            "system": {"assistant_name": "AskJeeves"},
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

    // Pre-stream errors return a non-200 HTTP response (not SSE)
    assert!(
        response.status().is_server_error() || response.status().is_client_error(),
        "Response should be an error (all providers failed)"
    );

    let response_json: Value = response.json().await.unwrap();

    let raw_response = response_json
        .get("raw_response")
        .expect("Error response should have raw_response when include_raw_response=true");
    let raw_response_array = raw_response.as_array().unwrap();
    assert_eq!(
        raw_response_array.len(),
        2,
        "raw_response should have 2 entries from 2 failed providers"
    );

    for entry in raw_response_array {
        assert_error_raw_response_entry(entry);
        assert_eq!(
            entry.get("data").unwrap().as_str().unwrap(),
            "dummy error raw response",
            "Failed provider data should be 'dummy error raw response'"
        );
    }
}

// =============================================================================
// Variant Fallback Tests
// =============================================================================

/// Variant fallback: error variant -> good variant, non-streaming.
/// Expects `raw_response` entries from the failed variant + successful variant.
#[tokio::test]
async fn test_variant_fallback_fail_success_non_streaming() {
    let episode_id = Uuid::now_v7();
    let payload = json!({
        "function_name": "raw_response_variant_fallback",
        "episode_id": episode_id,
        "input": {
            "system": {"assistant_name": "AskJeeves"},
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
        "Response should be successful (fallback to good variant)"
    );

    let response_json: Value = response.json().await.unwrap();

    let raw_response = response_json
        .get("raw_response")
        .expect("Response should have raw_response when include_raw_response=true");
    let raw_response_array = raw_response.as_array().unwrap();
    assert!(
        raw_response_array.len() >= 2,
        "raw_response should have at least 2 entries: error variant + successful variant, got {}",
        raw_response_array.len()
    );

    // First entry: failed variant (error entry — no model_inference_id)
    let error_entry = &raw_response_array[0];
    assert_error_raw_response_entry(error_entry);
    assert_eq!(
        error_entry.get("data").unwrap().as_str().unwrap(),
        "dummy error raw response",
        "Failed variant data should be 'dummy error raw response'"
    );

    // Last entry: successful variant (success entry — has model_inference_id)
    let success_entry = raw_response_array.last().unwrap();
    assert_raw_response_entry(success_entry);
}

/// Variant fallback: error variant -> good variant, streaming.
/// Expects error entries from failed variant in a chunk, then streaming chunks from good variant.
#[tokio::test]
async fn test_variant_fallback_fail_success_streaming() {
    let episode_id = Uuid::now_v7();
    let payload = json!({
        "function_name": "raw_response_variant_fallback",
        "episode_id": episode_id,
        "input": {
            "system": {"assistant_name": "AskJeeves"},
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

    let mut found_failed_raw_response = false;
    let mut found_raw_chunk = false;
    let mut chunk_count = 0;

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
        chunk_count += 1;

        // Check for raw_response entries (from failed variant)
        if let Some(raw_response) = chunk_json.get("raw_response") {
            let raw_response_array = raw_response.as_array().unwrap();
            for entry in raw_response_array {
                // Error entries from failed variant should not have model_inference_id
                if entry.get("model_inference_id").is_none() {
                    assert_error_raw_response_entry(entry);
                    assert_eq!(
                        entry.get("data").unwrap().as_str().unwrap(),
                        "dummy error raw response",
                        "Failed variant data should be 'dummy error raw response'"
                    );
                    found_failed_raw_response = true;
                }
            }
        }

        // Check for raw_chunk (from successful variant streaming)
        if chunk_json.get("raw_chunk").is_some() {
            found_raw_chunk = true;
        }
    }

    assert!(chunk_count > 0, "Should have received at least one chunk");
    assert!(
        found_failed_raw_response,
        "Should have received raw_response entries from failed variant in a streaming chunk"
    );
    assert!(
        found_raw_chunk,
        "Should have received raw_chunk from successful variant streaming"
    );
}

/// Variant fallback: both variants error, non-streaming.
/// Expects error response with `raw_response` entries from both failed variants.
#[tokio::test]
async fn test_variant_fallback_both_fail_non_streaming() {
    let episode_id = Uuid::now_v7();
    let payload = json!({
        "function_name": "raw_response_variant_both_fail",
        "episode_id": episode_id,
        "input": {
            "system": {"assistant_name": "AskJeeves"},
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
        response.status().is_server_error() || response.status().is_client_error(),
        "Response should be an error (all variants failed)"
    );

    let response_json: Value = response.json().await.unwrap();

    let raw_response = response_json
        .get("raw_response")
        .expect("Error response should have raw_response when include_raw_response=true");
    let raw_response_array = raw_response.as_array().unwrap();
    assert_eq!(
        raw_response_array.len(),
        2,
        "raw_response should have 2 entries from 2 failed variants"
    );

    for entry in raw_response_array {
        assert_error_raw_response_entry(entry);
        assert_eq!(
            entry.get("data").unwrap().as_str().unwrap(),
            "dummy error raw response",
            "Failed variant data should be 'dummy error raw response'"
        );
    }
}

/// Variant fallback: both variants error, streaming (pre-stream error).
/// Expects error response with `raw_response` entries from both failed variants.
#[tokio::test]
async fn test_variant_fallback_both_fail_streaming() {
    let episode_id = Uuid::now_v7();
    let payload = json!({
        "function_name": "raw_response_variant_both_fail",
        "episode_id": episode_id,
        "input": {
            "system": {"assistant_name": "AskJeeves"},
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

    // Pre-stream errors return a non-200 HTTP response (not SSE)
    assert!(
        response.status().is_server_error() || response.status().is_client_error(),
        "Response should be an error (all variants failed)"
    );

    let response_json: Value = response.json().await.unwrap();

    let raw_response = response_json
        .get("raw_response")
        .expect("Error response should have raw_response when include_raw_response=true");
    let raw_response_array = raw_response.as_array().unwrap();
    assert_eq!(
        raw_response_array.len(),
        2,
        "raw_response should have 2 entries from 2 failed variants"
    );

    for entry in raw_response_array {
        assert_error_raw_response_entry(entry);
        assert_eq!(
            entry.get("data").unwrap().as_str().unwrap(),
            "dummy error raw response",
            "Failed variant data should be 'dummy error raw response'"
        );
    }
}
