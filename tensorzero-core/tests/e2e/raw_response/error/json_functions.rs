//! JSON function (`type="json"`) error tests for `include_raw_response`.
//!
//! These tests use the `json_success` function with the `openai` variant,
//! triggering errors via `extra_body` with an invalid field.

use reqwest::Client;
use serde_json::{Value, json};
use uuid::Uuid;

use crate::common::get_gateway_endpoint;

use super::assert_error_raw_response_entry;

// =============================================================================
// T0 native /inference endpoint -- non-streaming -- include_raw_response=true
// =============================================================================

#[tokio::test(flavor = "multi_thread")]
async fn test_json_function_error_inference_non_streaming() {
    let episode_id = Uuid::now_v7();

    let payload = json!({
        "function_name": "json_success",
        "variant_name": "openai",
        "episode_id": episode_id,
        "input": {
            "system": {"assistant_name": "JsonBot"},
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "template", "name": "user", "arguments": {"country": "Japan"}}]
                }
            ]
        },
        "include_raw_response": true,
        "extra_body": [{"pointer": "/nonsense_field", "value": "garbage_12345"}]
    });

    let response = Client::new()
        .post(get_gateway_endpoint("/inference"))
        .json(&payload)
        .send()
        .await
        .unwrap();

    assert!(
        !response.status().is_success(),
        "Response should be an error, got status {}",
        response.status()
    );

    let response_text = response.text().await.unwrap();
    let response_json: Value =
        serde_json::from_str(&response_text).expect("Response should be valid JSON");

    // T0 native format: {"error": "..."}
    let error_field = response_json.get("error").unwrap_or_else(|| {
        panic!("Error response should have `error` field. Response: {response_text}")
    });
    assert!(
        error_field.is_string(),
        "T0 native error field should be a string, got: {error_field:?}"
    );

    // Should have raw_response array
    let raw_response = response_json.get("raw_response").unwrap_or_else(|| {
        panic!("Error response should have `raw_response` when include_raw_response=true. Response: {response_text}")
    });
    assert!(raw_response.is_array(), "raw_response should be an array");

    let raw_response_array = raw_response.as_array().unwrap();

    assert_eq!(
        raw_response_array.len(),
        1,
        "json_success has a single provider, expected 1 entry, got {}",
        raw_response_array.len()
    );

    let entry = &raw_response_array[0];
    assert_error_raw_response_entry(entry);

    assert_eq!(
        entry["provider_type"].as_str().unwrap(),
        "openai",
        "Provider type should be `openai`"
    );
    assert_eq!(
        entry["api_type"].as_str().unwrap(),
        "chat_completions",
        "API type should be `chat_completions`"
    );
}

// =============================================================================
// T0 native /inference endpoint -- non-streaming -- include_raw_response not set
// =============================================================================

#[tokio::test(flavor = "multi_thread")]
async fn test_json_function_error_inference_not_requested() {
    let episode_id = Uuid::now_v7();

    let payload = json!({
        "function_name": "json_success",
        "variant_name": "openai",
        "episode_id": episode_id,
        "input": {
            "system": {"assistant_name": "JsonBot"},
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "template", "name": "user", "arguments": {"country": "Japan"}}]
                }
            ]
        },
        "extra_body": [{"pointer": "/nonsense_field", "value": "garbage_12345"}]
    });

    let response = Client::new()
        .post(get_gateway_endpoint("/inference"))
        .json(&payload)
        .send()
        .await
        .unwrap();

    assert!(
        !response.status().is_success(),
        "Response should be an error, got status {}",
        response.status()
    );

    let response_json: Value = response.json().await.unwrap();

    // Should have error but no raw_response
    assert!(
        response_json.get("error").is_some(),
        "Error response should have `error` field"
    );
    assert!(
        response_json.get("raw_response").is_none(),
        "raw_response should not be present when include_raw_response is not set"
    );
}

// =============================================================================
// OAI-compatible /openai/v1/chat/completions -- non-streaming -- include_raw_response=true
// =============================================================================

#[tokio::test(flavor = "multi_thread")]
async fn test_json_function_error_openai_non_streaming() {
    let payload = json!({
        "model": "tensorzero::function_name::json_success",
        "messages": [
            {
                "role": "system",
                "content": [{"type": "text", "tensorzero::arguments": {"assistant_name": "JsonBot"}}]
            },
            {
                "role": "user",
                "content": [{"type": "text", "tensorzero::arguments": {"country": "Japan"}}]
            }
        ],
        "tensorzero::variant_name": "openai",
        "tensorzero::include_raw_response": true,
        "tensorzero::extra_body": [{"pointer": "/nonsense_field", "value": "garbage_12345"}]
    });

    let response = Client::new()
        .post(get_gateway_endpoint("/openai/v1/chat/completions"))
        .json(&payload)
        .send()
        .await
        .unwrap();

    assert!(
        !response.status().is_success(),
        "Response should be an error, got status {}",
        response.status()
    );

    let response_text = response.text().await.unwrap();
    let response_json: Value =
        serde_json::from_str(&response_text).expect("Response should be valid JSON");

    // OAI-format error: {"error": {"message": "..."}}
    let error_obj = response_json.get("error").unwrap_or_else(|| {
        panic!("Error response should have `error` field. Response: {response_text}")
    });
    assert!(
        error_obj.is_object(),
        "OAI error should be an object, got: {error_obj:?}"
    );
    assert!(
        error_obj.get("message").is_some(),
        "OAI error object should have `message` field"
    );

    // Should have tensorzero_raw_response array
    let raw_response = response_json.get("tensorzero_raw_response").unwrap_or_else(|| {
        panic!("Error response should have `tensorzero_raw_response` when include_raw_response=true. Response: {response_text}")
    });
    assert!(
        raw_response.is_array(),
        "tensorzero_raw_response should be an array"
    );

    let raw_response_array = raw_response.as_array().unwrap();

    assert_eq!(
        raw_response_array.len(),
        1,
        "json_success has a single provider, expected 1 entry, got {}",
        raw_response_array.len()
    );

    let entry = &raw_response_array[0];
    assert_error_raw_response_entry(entry);

    assert_eq!(
        entry["provider_type"].as_str().unwrap(),
        "openai",
        "Provider type should be `openai`"
    );
    assert_eq!(
        entry["api_type"].as_str().unwrap(),
        "chat_completions",
        "API type should be `chat_completions`"
    );
}

// =============================================================================
// OAI-compatible /openai/v1/chat/completions -- non-streaming -- include_raw_response not set
// =============================================================================

#[tokio::test(flavor = "multi_thread")]
async fn test_json_function_error_openai_not_requested() {
    let payload = json!({
        "model": "tensorzero::function_name::json_success",
        "messages": [
            {
                "role": "system",
                "content": [{"type": "text", "tensorzero::arguments": {"assistant_name": "JsonBot"}}]
            },
            {
                "role": "user",
                "content": [{"type": "text", "tensorzero::arguments": {"country": "Japan"}}]
            }
        ],
        "tensorzero::variant_name": "openai",
        // Note: tensorzero::include_raw_response is NOT set (defaults to false)
        "tensorzero::extra_body": [{"pointer": "/nonsense_field", "value": "garbage_12345"}]
    });

    let response = Client::new()
        .post(get_gateway_endpoint("/openai/v1/chat/completions"))
        .json(&payload)
        .send()
        .await
        .unwrap();

    assert!(
        !response.status().is_success(),
        "Response should be an error, got status {}",
        response.status()
    );

    let response_json: Value = response.json().await.unwrap();

    // Should have error but no tensorzero_raw_response
    assert!(
        response_json.get("error").is_some(),
        "Error response should have `error` field"
    );
    assert!(
        response_json.get("tensorzero_raw_response").is_none(),
        "tensorzero_raw_response should not be present when include_raw_response is not set"
    );
}

// =============================================================================
// T0 native /inference endpoint -- streaming -- include_raw_response=true
// =============================================================================

#[tokio::test(flavor = "multi_thread")]
async fn test_json_function_error_inference_streaming() {
    let episode_id = Uuid::now_v7();

    let payload = json!({
        "function_name": "json_success",
        "variant_name": "openai",
        "episode_id": episode_id,
        "input": {
            "system": {"assistant_name": "JsonBot"},
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "template", "name": "user", "arguments": {"country": "Japan"}}]
                }
            ]
        },
        "stream": true,
        "include_raw_response": true,
        "extra_body": [{"pointer": "/nonsense_field", "value": "garbage_12345"}]
    });

    let response = Client::new()
        .post(get_gateway_endpoint("/inference"))
        .json(&payload)
        .send()
        .await
        .unwrap();

    // Terminal error: should be a non-200 JSON response, not SSE
    assert!(
        !response.status().is_success(),
        "Response should be an error, got status {}",
        response.status()
    );

    let response_text = response.text().await.unwrap();
    let response_json: Value =
        serde_json::from_str(&response_text).expect("Response should be valid JSON");

    // T0 native format: {"error": "..."}
    let error_field = response_json.get("error").unwrap_or_else(|| {
        panic!("Error response should have `error` field. Response: {response_text}")
    });
    assert!(
        error_field.is_string(),
        "T0 native error field should be a string, got: {error_field:?}"
    );

    // Should have raw_response array
    let raw_response = response_json.get("raw_response").unwrap_or_else(|| {
        panic!("Error response should have `raw_response` when include_raw_response=true. Response: {response_text}")
    });
    assert!(raw_response.is_array(), "raw_response should be an array");

    let raw_response_array = raw_response.as_array().unwrap();

    assert_eq!(
        raw_response_array.len(),
        1,
        "json_success has a single provider, expected 1 entry, got {}",
        raw_response_array.len()
    );

    let entry = &raw_response_array[0];
    assert_error_raw_response_entry(entry);

    assert_eq!(
        entry["provider_type"].as_str().unwrap(),
        "openai",
        "Provider type should be `openai`"
    );
    assert_eq!(
        entry["api_type"].as_str().unwrap(),
        "chat_completions",
        "API type should be `chat_completions`"
    );
}

// =============================================================================
// T0 native /inference endpoint -- streaming -- include_raw_response not set
// =============================================================================

#[tokio::test(flavor = "multi_thread")]
async fn test_json_function_error_inference_streaming_not_requested() {
    let episode_id = Uuid::now_v7();

    let payload = json!({
        "function_name": "json_success",
        "variant_name": "openai",
        "episode_id": episode_id,
        "input": {
            "system": {"assistant_name": "JsonBot"},
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "template", "name": "user", "arguments": {"country": "Japan"}}]
                }
            ]
        },
        "stream": true,
        "extra_body": [{"pointer": "/nonsense_field", "value": "garbage_12345"}]
    });

    let response = Client::new()
        .post(get_gateway_endpoint("/inference"))
        .json(&payload)
        .send()
        .await
        .unwrap();

    // Terminal error: should be a non-200 JSON response, not SSE
    assert!(
        !response.status().is_success(),
        "Response should be an error, got status {}",
        response.status()
    );

    let response_json: Value = response.json().await.unwrap();

    // Should have error but no raw_response
    assert!(
        response_json.get("error").is_some(),
        "Error response should have `error` field"
    );
    assert!(
        response_json.get("raw_response").is_none(),
        "raw_response should not be present when include_raw_response is not set"
    );
}

// =============================================================================
// OAI-compatible /openai/v1/chat/completions -- streaming -- include_raw_response=true
// =============================================================================

#[tokio::test(flavor = "multi_thread")]
async fn test_json_function_error_openai_streaming() {
    let payload = json!({
        "model": "tensorzero::function_name::json_success",
        "messages": [
            {
                "role": "system",
                "content": [{"type": "text", "tensorzero::arguments": {"assistant_name": "JsonBot"}}]
            },
            {
                "role": "user",
                "content": [{"type": "text", "tensorzero::arguments": {"country": "Japan"}}]
            }
        ],
        "stream": true,
        "tensorzero::variant_name": "openai",
        "tensorzero::include_raw_response": true,
        "tensorzero::extra_body": [{"pointer": "/nonsense_field", "value": "garbage_12345"}]
    });

    let response = Client::new()
        .post(get_gateway_endpoint("/openai/v1/chat/completions"))
        .json(&payload)
        .send()
        .await
        .unwrap();

    // Terminal error: should be a non-200 JSON response, not SSE
    assert!(
        !response.status().is_success(),
        "Response should be an error, got status {}",
        response.status()
    );

    let response_text = response.text().await.unwrap();
    let response_json: Value =
        serde_json::from_str(&response_text).expect("Response should be valid JSON");

    // OAI-format error: {"error": {"message": "..."}}
    let error_obj = response_json.get("error").unwrap_or_else(|| {
        panic!("Error response should have `error` field. Response: {response_text}")
    });
    assert!(
        error_obj.is_object(),
        "OAI error should be an object, got: {error_obj:?}"
    );
    assert!(
        error_obj.get("message").is_some(),
        "OAI error object should have `message` field"
    );

    // Should have tensorzero_raw_response array
    let raw_response = response_json.get("tensorzero_raw_response").unwrap_or_else(|| {
        panic!("Error response should have `tensorzero_raw_response` when include_raw_response=true. Response: {response_text}")
    });
    assert!(
        raw_response.is_array(),
        "tensorzero_raw_response should be an array"
    );

    let raw_response_array = raw_response.as_array().unwrap();

    assert_eq!(
        raw_response_array.len(),
        1,
        "json_success has a single provider, expected 1 entry, got {}",
        raw_response_array.len()
    );

    let entry = &raw_response_array[0];
    assert_error_raw_response_entry(entry);

    assert_eq!(
        entry["provider_type"].as_str().unwrap(),
        "openai",
        "Provider type should be `openai`"
    );
    assert_eq!(
        entry["api_type"].as_str().unwrap(),
        "chat_completions",
        "API type should be `chat_completions`"
    );
}

// =============================================================================
// OAI-compatible /openai/v1/chat/completions -- streaming -- include_raw_response not set
// =============================================================================

#[tokio::test(flavor = "multi_thread")]
async fn test_json_function_error_openai_streaming_not_requested() {
    let payload = json!({
        "model": "tensorzero::function_name::json_success",
        "messages": [
            {
                "role": "system",
                "content": [{"type": "text", "tensorzero::arguments": {"assistant_name": "JsonBot"}}]
            },
            {
                "role": "user",
                "content": [{"type": "text", "tensorzero::arguments": {"country": "Japan"}}]
            }
        ],
        "stream": true,
        "tensorzero::variant_name": "openai",
        // Note: tensorzero::include_raw_response is NOT set (defaults to false)
        "tensorzero::extra_body": [{"pointer": "/nonsense_field", "value": "garbage_12345"}]
    });

    let response = Client::new()
        .post(get_gateway_endpoint("/openai/v1/chat/completions"))
        .json(&payload)
        .send()
        .await
        .unwrap();

    // Terminal error: should be a non-200 JSON response, not SSE
    assert!(
        !response.status().is_success(),
        "Response should be an error, got status {}",
        response.status()
    );

    let response_json: Value = response.json().await.unwrap();

    // Should have error but no tensorzero_raw_response
    assert!(
        response_json.get("error").is_some(),
        "Error response should have `error` field"
    );
    assert!(
        response_json.get("tensorzero_raw_response").is_none(),
        "tensorzero_raw_response should not be present when include_raw_response is not set"
    );
}
