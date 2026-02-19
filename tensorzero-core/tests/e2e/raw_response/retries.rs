//! E2E tests for `raw_response` in retry scenarios.
//!
//! Tests that raw provider-specific response data from failed retry attempts
//! is correctly included in responses when `include_raw_response=true`.

use futures::StreamExt;
use reqwest::{Client, StatusCode};
use reqwest_sse_stream::{Event, RequestBuilderExt};
use serde_json::{Value, json};
use uuid::Uuid;

use super::assert_raw_response_entry;
use super::error::assert_error_raw_response_entry;
use crate::common::get_gateway_endpoint;

// =============================================================================
// Flaky model retry tests (fail then succeed)
// =============================================================================

/// Helper to make a warm-up call so the flaky model counter advances.
/// The flaky model fails on even-numbered calls (2, 4, 6, ...),
/// so we need to ensure the counter is at an even value before the real test call.
async fn warmup_flaky_retries() {
    let episode_id = Uuid::now_v7();
    let payload = json!({
        "function_name": "raw_response_retries_flaky",
        "variant_name": "test",
        "episode_id": episode_id,
        "input": {
            "system": {"assistant_name": "AskJeeves"},
            "messages": [{"role": "user", "content": "warmup"}]
        },
        "stream": false,
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
        "Warm-up call should succeed (counter at odd value)"
    );
}

/// Retries: flaky model fails then succeeds on retry, non-streaming.
/// The flaky model counter goes: warmup→1 (success), test call→2 (fail), retry→3 (success).
/// Expects `raw_response` to have error entry from failed attempt + success entry from retry.
#[tokio::test]
async fn test_retries_fail_then_succeed_non_streaming() {
    // Warm up to advance counter to 1
    warmup_flaky_retries().await;

    let episode_id = Uuid::now_v7();
    let payload = json!({
        "function_name": "raw_response_retries_flaky",
        "variant_name": "test",
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
        "Response should be successful (retry succeeds after initial failure)"
    );

    let response_json: Value = response.json().await.unwrap();

    let raw_response = response_json
        .get("raw_response")
        .expect("Response should have raw_response when include_raw_response=true");
    let raw_response_array = raw_response.as_array().unwrap();
    assert_eq!(
        raw_response_array.len(),
        2,
        "raw_response should have 2 entries: 1 from failed retry attempt + 1 from successful attempt"
    );

    // First entry: failed retry attempt (error entry — no model_inference_id)
    let error_entry = &raw_response_array[0];
    assert_error_raw_response_entry(error_entry);
    assert_eq!(
        error_entry.get("provider_type").unwrap().as_str().unwrap(),
        "dummy",
        "Failed retry attempt provider_type should be `dummy`"
    );
    assert_eq!(
        error_entry.get("data").unwrap().as_str().unwrap(),
        "dummy flaky error raw response",
        "Failed retry attempt data should be `dummy flaky error raw response`"
    );

    // Second entry: successful attempt (success entry — has model_inference_id)
    let success_entry = &raw_response_array[1];
    assert_raw_response_entry(success_entry);
    assert_eq!(
        success_entry
            .get("provider_type")
            .unwrap()
            .as_str()
            .unwrap(),
        "dummy",
        "Successful attempt provider_type should be `dummy`"
    );
}

/// Retries: flaky model fails then succeeds on retry, streaming.
/// Expects failed retry entries in `raw_response` chunk, then `raw_chunk` from successful streaming.
#[tokio::test]
async fn test_retries_fail_then_succeed_streaming() {
    // Warm up to advance counter to 1 (odd = success).
    // Both infer and infer_stream share the same FLAKY_COUNTERS,
    // so a non-streaming warmup works for the streaming test.
    warmup_flaky_retries().await;

    let episode_id = Uuid::now_v7();
    let payload = json!({
        "function_name": "raw_response_retries_flaky",
        "variant_name": "test",
        "episode_id": episode_id,
        "input": {
            "system": {"assistant_name": "AskJeeves"},
            "messages": [{"role": "user", "content": "Hello streaming"}]
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

        // Check for raw_response entries (from failed retry attempt)
        if let Some(raw_response) = chunk_json.get("raw_response") {
            let raw_response_array = raw_response.as_array().unwrap();
            for entry in raw_response_array {
                assert_error_raw_response_entry(entry);
                assert_eq!(
                    entry.get("provider_type").unwrap().as_str().unwrap(),
                    "dummy",
                    "Failed retry attempt provider_type should be `dummy`"
                );
                assert_eq!(
                    entry.get("data").unwrap().as_str().unwrap(),
                    "dummy flaky error raw response",
                    "Failed retry attempt data should be `dummy flaky error raw response`"
                );
            }
            found_failed_raw_response = true;
        }

        // Check for raw_chunk (from successful streaming attempt)
        if chunk_json.get("raw_chunk").is_some() {
            found_raw_chunk = true;
        }
    }

    assert!(chunk_count > 0, "Should have received at least one chunk");
    assert!(
        found_failed_raw_response,
        "Should have received raw_response entries from failed retry attempt"
    );
    assert!(
        found_raw_chunk,
        "Should have received raw_chunk from successful streaming attempt"
    );
}

// =============================================================================
// All retries fail tests
// =============================================================================

/// Retries: all attempts fail (error_with_raw_response always fails), non-streaming.
/// With 2 retries = 3 total attempts, expects error response with `raw_response` from all 3 attempts.
#[tokio::test]
async fn test_retries_all_fail_non_streaming() {
    let episode_id = Uuid::now_v7();
    let payload = json!({
        "function_name": "raw_response_retries_all_fail",
        "variant_name": "test",
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
        "Response should be an error (all retries failed)"
    );

    let response_json: Value = response.json().await.unwrap();

    let raw_response = response_json
        .get("raw_response")
        .expect("Error response should have raw_response when include_raw_response=true");
    let raw_response_array = raw_response.as_array().unwrap();
    assert_eq!(
        raw_response_array.len(),
        3,
        "raw_response should have 3 entries from 3 failed attempts (1 initial + 2 retries)"
    );

    // All entries should be error entries with raw response data
    for entry in raw_response_array {
        assert_error_raw_response_entry(entry);
        assert_eq!(
            entry.get("provider_type").unwrap().as_str().unwrap(),
            "dummy",
            "Failed attempt provider_type should be `dummy`"
        );
        assert_eq!(
            entry.get("data").unwrap().as_str().unwrap(),
            "dummy error raw response",
            "Failed attempt data should be `dummy error raw response`"
        );
    }
}

/// Retries: all attempts fail, streaming (pre-stream error).
/// Expects error response body (not SSE) with `raw_response` from all 3 attempts.
#[tokio::test]
async fn test_retries_all_fail_streaming() {
    let episode_id = Uuid::now_v7();
    let payload = json!({
        "function_name": "raw_response_retries_all_fail",
        "variant_name": "test",
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
        "Response should be an error (all retries failed)"
    );

    let response_json: Value = response.json().await.unwrap();

    let raw_response = response_json
        .get("raw_response")
        .expect("Error response should have raw_response when include_raw_response=true");
    let raw_response_array = raw_response.as_array().unwrap();
    assert_eq!(
        raw_response_array.len(),
        3,
        "raw_response should have 3 entries from 3 failed attempts (1 initial + 2 retries)"
    );

    for entry in raw_response_array {
        assert_error_raw_response_entry(entry);
        assert_eq!(
            entry.get("provider_type").unwrap().as_str().unwrap(),
            "dummy",
            "Failed attempt provider_type should be `dummy`"
        );
        assert_eq!(
            entry.get("data").unwrap().as_str().unwrap(),
            "dummy error raw response",
            "Failed attempt data should be `dummy error raw response`"
        );
    }
}

// =============================================================================
// Retries with provider fallback tests
// =============================================================================

/// Retries with provider fallback: model has error provider + good provider, non-streaming.
/// The model's provider fallback handles the error within a single attempt (no retry fires).
/// Expects both the failed provider entry + success entry in `raw_response`.
#[tokio::test]
async fn test_retries_with_provider_fallback_non_streaming() {
    let episode_id = Uuid::now_v7();
    let payload = json!({
        "function_name": "raw_response_retries_with_provider_fallback",
        "variant_name": "test",
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
        "Response should be successful (provider fallback succeeds)"
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
        "Failed provider type should be `dummy`"
    );
    assert_eq!(
        error_entry.get("data").unwrap().as_str().unwrap(),
        "dummy error raw response",
        "Failed provider data should be `dummy error raw response`"
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
        "Successful provider_type should be `dummy`"
    );
}

/// Retries with provider fallback: model has error provider + good provider, streaming.
/// Expects failed provider entries in `raw_response` chunk, then `raw_chunk` from good provider.
#[tokio::test]
async fn test_retries_with_provider_fallback_streaming() {
    let episode_id = Uuid::now_v7();
    let payload = json!({
        "function_name": "raw_response_retries_with_provider_fallback",
        "variant_name": "test",
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
                    entry.get("provider_type").unwrap().as_str().unwrap(),
                    "dummy",
                    "Failed provider_type should be `dummy`"
                );
                assert_eq!(
                    entry.get("data").unwrap().as_str().unwrap(),
                    "dummy error raw response",
                    "Failed provider data should be `dummy error raw response`"
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
