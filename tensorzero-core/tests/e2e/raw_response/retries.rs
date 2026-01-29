//! E2E tests for `include_raw_response` parameter with retry scenarios.
//!
//! Tests that raw_response data from failed retry attempts is correctly included
//! in the final response when `include_raw_response=true`.

use futures::StreamExt;
use reqwest::{Client, StatusCode};
use reqwest_eventsource::{Event, RequestBuilderExt};
use serde_json::{Value, json};
use uuid::Uuid;

use crate::common::get_gateway_endpoint;

// =============================================================================
// Retry Tests with Failed Attempt raw_response
// =============================================================================

/// Test that non-streaming responses include raw_response from failed retry attempts
#[tokio::test]
async fn e2e_test_raw_response_includes_failed_retry_attempts_non_streaming() {
    let episode_id = Uuid::now_v7();

    // Use flaky_with_raw_response model - fails on even calls with raw_response data
    // The variant has retries configured, so it will retry on failure
    let payload = json!({
        "function_name": "basic_test",
        "variant_name": "flaky_with_raw_response",
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
        "Response should be successful after retries"
    );

    let body: Value = response.json().await.unwrap();

    let raw_response = body
        .get("raw_response")
        .expect("Response should have raw_response");
    let entries = raw_response.as_array().unwrap();

    // Should have entries from BOTH failed attempt AND successful attempt
    // - Failed attempt: raw_response from error
    // - Successful attempt: raw_response from success
    assert!(
        entries.len() >= 2,
        "Should have raw_response from failed retry + successful attempt, got {}",
        entries.len()
    );

    // Verify we have error data from failed attempt
    let has_error_entry = entries.iter().any(|e| {
        e.get("data")
            .and_then(|d| d.as_str())
            .map(|s| s.contains("flaky_failure"))
            .unwrap_or(false)
    });
    assert!(
        has_error_entry,
        "Should include raw_response from failed retry attempt"
    );
}

/// Test that streaming responses include raw_response from failed retry attempts
#[tokio::test]
async fn e2e_test_raw_response_includes_failed_retry_attempts_streaming() {
    let episode_id = Uuid::now_v7();

    let payload = json!({
        "function_name": "basic_test",
        "variant_name": "flaky_with_raw_response",
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
        .expect("Failed to create eventsource");

    let mut raw_response_entries: Vec<Value> = Vec::new();
    let mut found_raw_chunk = false;

    while let Some(chunk) = chunks.next().await {
        let chunk = match chunk {
            Ok(c) => c,
            Err(_) => continue,
        };
        let Event::Message(message) = chunk else {
            continue;
        };
        if message.data == "[DONE]" {
            break;
        }

        let chunk_json: Value = serde_json::from_str(&message.data).unwrap();

        if let Some(raw_response) = chunk_json.get("raw_response")
            && let Some(arr) = raw_response.as_array()
        {
            raw_response_entries.extend(arr.clone());
        }

        if chunk_json.get("raw_chunk").is_some() {
            found_raw_chunk = true;
        }
    }

    assert!(
        found_raw_chunk,
        "Streaming should have raw_chunk from successful attempt"
    );

    // Should have raw_response entry from failed retry attempt
    let has_error_entry = raw_response_entries.iter().any(|e| {
        e.get("data")
            .and_then(|d| d.as_str())
            .map(|s| s.contains("flaky_failure"))
            .unwrap_or(false)
    });
    assert!(
        has_error_entry,
        "Should include raw_response from failed retry attempt in streaming"
    );
}
