//! E2E tests for `include_raw_response` parameter in error scenarios.
//!
//! Tests that raw provider-specific response data is correctly returned in errors
//! when `include_raw_response=true` is set.

use futures::StreamExt;
use reqwest::Client;
use reqwest_eventsource::{Event, RequestBuilderExt};
use serde_json::{Value, json};
use uuid::Uuid;

use crate::common::get_gateway_endpoint;

/// Helper to assert raw_response entry structure in errors
fn assert_error_raw_response_entry(entry: &Value) {
    assert!(
        entry.get("provider_type").is_some(),
        "raw_response entry should have provider_type"
    );
    assert!(
        entry.get("api_type").is_some(),
        "raw_response entry should have api_type"
    );
    assert!(
        entry.get("data").is_some(),
        "raw_response entry should have data field"
    );

    // Verify api_type is a valid value
    let api_type = entry.get("api_type").unwrap().as_str().unwrap();
    assert!(
        ["chat_completions", "responses", "embeddings"].contains(&api_type),
        "api_type should be a valid value, got: {api_type}"
    );

    // Verify data is a string (raw response from provider)
    assert!(
        entry.get("data").unwrap().is_string(),
        "data should be a string"
    );
}

// =============================================================================
// Direct Error Tests (non-relay)
// =============================================================================

/// Test that non-streaming errors include raw_response when include_raw_response=true
#[tokio::test]
async fn e2e_test_raw_response_direct_error_non_streaming() {
    let episode_id = Uuid::now_v7();

    // Use the error_with_raw_response variant which returns an error with raw_response data
    let payload = json!({
        "function_name": "basic_test",
        "variant_name": "error_with_raw_response",
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

    // Should be an error status
    assert!(
        !response.status().is_success(),
        "Response should be an error status"
    );

    let body: Value = response.json().await.unwrap();

    // Should have error field
    assert!(
        body.get("error").is_some(),
        "Response should have an error field"
    );

    // Should have raw_response in error when include_raw_response=true
    let raw_response = body
        .get("raw_response")
        .expect("Error response should include raw_response when include_raw_response=true");
    assert!(raw_response.is_array(), "raw_response should be an array");

    let entries = raw_response.as_array().unwrap();
    assert!(
        !entries.is_empty(),
        "Should have at least one raw_response entry"
    );

    // Validate entry structure
    for entry in entries {
        assert_error_raw_response_entry(entry);

        // Verify it contains our test error data
        let data = entry.get("data").and_then(|d| d.as_str()).unwrap();
        assert!(
            data.contains("test_error"),
            "raw_response data should contain error info, got: {data}"
        );

        // Provider should be dummy
        let provider_type = entry.get("provider_type").and_then(|v| v.as_str()).unwrap();
        assert_eq!(provider_type, "dummy", "Provider type should be 'dummy'");
    }
}

/// Test that streaming errors include raw_response when include_raw_response=true
#[tokio::test]
async fn e2e_test_raw_response_direct_error_streaming() {
    let episode_id = Uuid::now_v7();

    let payload = json!({
        "function_name": "basic_test",
        "variant_name": "error_with_raw_response",
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

    let mut found_error_with_raw_response = false;

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

        // Check for error with raw_response
        if chunk_json.get("error").is_some()
            && let Some(raw_response) = chunk_json.get("raw_response")
        {
            found_error_with_raw_response = true;
            assert!(raw_response.is_array(), "raw_response should be an array");

            let entries = raw_response.as_array().unwrap();
            assert!(
                !entries.is_empty(),
                "Error should have raw_response entries"
            );

            // Verify entry contains error data
            let has_error_data = entries.iter().any(|e| {
                e.get("data")
                    .and_then(|d| d.as_str())
                    .map(|s| s.contains("test_error"))
                    .unwrap_or(false)
            });
            assert!(has_error_data, "raw_response should contain error data");
        }
    }

    assert!(
        found_error_with_raw_response,
        "Streaming error should include raw_response"
    );
}
