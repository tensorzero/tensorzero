//! E2E tests for `include_raw_response` parameter with retry scenarios.
//!
//! Tests that raw_response data from failed retry attempts is correctly included
//! in the final response when `include_raw_response=true`.
//!
//! These tests use the `retry_once_with_raw_response` model which deterministically
//! fails on the first call for any given request and succeeds on retry.

use futures::StreamExt;
use serde_json::{Map, json};
use tensorzero::test_helpers::make_embedded_gateway_e2e_with_unique_db;
use tensorzero::{
    ClientInferenceParams, InferenceOutput, Input, InputMessage, InputMessageContent,
};
use tensorzero_core::inference::types::{Arguments, Role, System, Text};
use uuid::Uuid;

// =============================================================================
// Retry Tests with Failed Attempt raw_response
// =============================================================================

/// Test that non-streaming responses include raw_response from failed retry attempts
#[tokio::test(flavor = "multi_thread")]
async fn test_raw_response_includes_failed_retry_attempts_non_streaming() {
    let client = make_embedded_gateway_e2e_with_unique_db("raw_response_retry_non_streaming").await;

    // Use retry_once_with_raw_response model - always fails first, succeeds on retry
    let response = client
        .inference(ClientInferenceParams {
            function_name: Some("basic_test".to_string()),
            variant_name: Some("retry_once_with_raw_response".to_string()),
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
                        text: "Hello non-streaming".to_string(),
                    })],
                }],
            },
            stream: Some(false),
            include_raw_response: true,
            ..Default::default()
        })
        .await
        .unwrap();

    let InferenceOutput::NonStreaming(response) = response else {
        panic!("Expected non-streaming response");
    };

    let raw_response = response
        .raw_response()
        .expect("Response should have raw_response");

    // Should have entries from BOTH failed attempt AND successful attempt
    // - Failed attempt: raw_response from error
    // - Successful attempt: raw_response from success
    assert!(
        raw_response.len() >= 2,
        "Should have raw_response from failed retry + successful attempt, got {}",
        raw_response.len()
    );

    // Verify we have error data from failed attempt
    let has_error_entry = raw_response
        .iter()
        .any(|e| e.data.contains("retry_once_failure"));
    assert!(
        has_error_entry,
        "Should include raw_response from failed retry attempt"
    );
}

/// Test that streaming responses include raw_response from failed retry attempts
#[tokio::test(flavor = "multi_thread")]
async fn test_raw_response_includes_failed_retry_attempts_streaming() {
    let client = make_embedded_gateway_e2e_with_unique_db("raw_response_retry_streaming").await;

    // Use retry_once_with_raw_response model - always fails first, succeeds on retry
    let response = client
        .inference(ClientInferenceParams {
            function_name: Some("basic_test".to_string()),
            variant_name: Some("retry_once_with_raw_response".to_string()),
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
                        text: "Hello streaming".to_string(),
                    })],
                }],
            },
            stream: Some(true),
            include_raw_response: true,
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

    assert!(
        found_raw_chunk,
        "Streaming should have raw_chunk from successful attempt"
    );

    // Should have raw_response entry from failed retry attempt
    let has_error_entry = raw_response_entries
        .iter()
        .any(|e| e.data.contains("retry_once_failure"));
    assert!(
        has_error_entry,
        "Should include raw_response from failed retry attempt in streaming"
    );
}
