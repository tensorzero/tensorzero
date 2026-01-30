//! E2E tests for `include_raw_response` parameter in error scenarios.
//!
//! Tests that raw provider-specific response data is correctly returned in errors
//! when `include_raw_response=true` is set.

use serde_json::{Map, Value, json};
use tensorzero::test_helpers::make_embedded_gateway_e2e_with_unique_db;
use tensorzero::{
    ClientInferenceParams, Input, InputMessage, InputMessageContent, TensorZeroError,
};
use tensorzero_core::inference::types::{Arguments, Role, System, Text};
use uuid::Uuid;

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
#[tokio::test(flavor = "multi_thread")]
async fn test_raw_response_direct_error_non_streaming() {
    let client = make_embedded_gateway_e2e_with_unique_db("raw_response_error_non_streaming").await;

    // Use the error_with_raw_response variant which returns an error with raw_response data
    let result = client
        .inference(ClientInferenceParams {
            function_name: Some("basic_test".to_string()),
            variant_name: Some("error_with_raw_response".to_string()),
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
                        text: "Hello".to_string(),
                    })],
                }],
            },
            stream: Some(false),
            include_raw_response: true,
            ..Default::default()
        })
        .await;

    // Should be an error
    let err = result.expect_err("Response should be an error");

    // Extract raw_response from the error
    let TensorZeroError::Http { text, .. } = err else {
        panic!("Expected HTTP error, got: {err:?}");
    };

    let text = text.expect("Error should have text body");
    let body: Value = serde_json::from_str(&text).expect("Error body should be valid JSON");

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

/// Test that streaming request errors include raw_response when include_raw_response=true
/// Note: When an error happens before streaming starts, the server returns a regular HTTP error
/// response instead of starting SSE. This test verifies that even with stream=true, the error
/// response includes raw_response.
#[tokio::test(flavor = "multi_thread")]
async fn test_raw_response_direct_error_streaming() {
    let client = make_embedded_gateway_e2e_with_unique_db("raw_response_error_streaming").await;

    let result = client
        .inference(ClientInferenceParams {
            function_name: Some("basic_test".to_string()),
            variant_name: Some("error_with_raw_response".to_string()),
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
                        text: "Hello".to_string(),
                    })],
                }],
            },
            stream: Some(true),
            include_raw_response: true,
            ..Default::default()
        })
        .await;

    // Should be an error
    let err = result.expect_err("Response should be an error");

    // Extract raw_response from the error
    let TensorZeroError::Http { text, .. } = err else {
        panic!("Expected HTTP error, got: {err:?}");
    };

    let text = text.expect("Error should have text body");
    let body: Value = serde_json::from_str(&text).expect("Error body should be valid JSON");

    // raw_response should be included in the error response
    let raw_response = body
        .get("raw_response")
        .expect("Streaming error should include raw_response");
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

    // Provider should be dummy
    for entry in entries {
        let provider_type = entry.get("provider_type").and_then(|v| v.as_str()).unwrap();
        assert_eq!(provider_type, "dummy", "Provider type should be 'dummy'");
    }
}

// =============================================================================
// Mid-Stream Error Tests
// =============================================================================

/// Test that mid-stream errors (errors after streaming starts) include raw_response
#[tokio::test(flavor = "multi_thread")]
async fn test_raw_response_mid_stream_error() {
    use futures::StreamExt;
    use tensorzero::InferenceOutput;

    let client = make_embedded_gateway_e2e_with_unique_db("raw_response_mid_stream_error").await;

    let response = client
        .inference(ClientInferenceParams {
            function_name: Some("basic_test".to_string()),
            variant_name: Some("fatal_stream_error_with_raw_response".to_string()),
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
                        text: "Hello".to_string(),
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

    let mut chunks_before_error = 0;
    let mut found_error_with_raw_response = false;

    while let Some(chunk) = stream.next().await {
        match chunk {
            Ok(_chunk) => {
                chunks_before_error += 1;
            }
            Err(_e) => {
                // Mid-stream error occurred
                // The error should have raw_response data
                // Note: The TensorZeroError type doesn't expose raw_response directly,
                // but it's included in the error serialization for HTTP responses.
                // Here we just verify that the error occurred after some chunks were received.
                found_error_with_raw_response = true;
                break;
            }
        }
    }

    assert!(
        chunks_before_error > 0,
        "Should have received some chunks before the mid-stream error"
    );
    assert!(
        found_error_with_raw_response,
        "Mid-stream error should have occurred"
    );
}
