//! E2E tests for `raw_response` in error responses.
//!
//! Tests that raw provider-specific response data is correctly returned in error responses
//! when `include_raw_response` is true and an inference fails (e.g., due to an invalid model).
//!
//! Uses `openai::gpt-5-mini` as a shorthand model name with nonsense `extra_body` to trigger
//! a provider error. Requires an external gateway running (`cargo run-e2e`).

use reqwest::Client;
use serde_json::{Value, json};

use crate::common::get_gateway_endpoint;

/// Helper to assert a raw_response entry in an error response has valid structure.
///
/// Error entries differ from success entries: `model_inference_id` should be absent
/// (since the inference failed before a `model_inference_id` could be assigned,
/// and `None` values are omitted via `skip_serializing_if`).
fn assert_error_raw_response_entry(entry: &Value) {
    assert!(
        entry.get("model_inference_id").is_none(),
        "model_inference_id should be absent for error entries (no successful model inference)"
    );

    let provider_type = entry
        .get("provider_type")
        .expect("raw_response entry should have provider_type");
    assert!(
        provider_type.is_string(),
        "provider_type should be a string, got: {provider_type:?}"
    );

    let api_type = entry
        .get("api_type")
        .expect("raw_response entry should have api_type");
    let api_type_str = api_type.as_str().expect("api_type should be a string");
    assert!(
        ["chat_completions", "responses", "embeddings", "other"].contains(&api_type_str),
        "api_type should be a valid value, got: {api_type_str}"
    );

    let data = entry
        .get("data")
        .expect("raw_response entry should have data");
    let data_str = data.as_str().expect("data should be a string");
    assert!(!data_str.is_empty(), "data should not be empty");
}

// =============================================================================
// T0 native /inference endpoint -- include_raw_response=true
// =============================================================================

#[tokio::test(flavor = "multi_thread")]
async fn test_raw_response_error_inference_non_streaming() {
    let payload = json!({
        "model_name": "openai::gpt-5-mini",
        "input": {
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": "Error test inference"}]
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

    // Single provider -- expect exactly 1 entry
    assert_eq!(
        raw_response_array.len(),
        1,
        "Shorthand model has a single provider, expected 1 entry, got {}",
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
// T0 native /inference endpoint -- include_raw_response not set (defaults to false)
// =============================================================================

#[tokio::test(flavor = "multi_thread")]
async fn test_raw_response_error_inference_not_requested() {
    let payload = json!({
        "model_name": "openai::gpt-5-mini",
        "input": {
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": "Error test inference not requested"}]
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
// OAI-compatible /openai/v1/chat/completions -- include_raw_response=true
// =============================================================================

#[tokio::test(flavor = "multi_thread")]
async fn test_raw_response_error_openai_non_streaming() {
    let payload = json!({
        "model": "tensorzero::model_name::openai::gpt-5-mini",
        "messages": [
            {
                "role": "user",
                "content": "Error test openai"
            }
        ],
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

    // Single provider -- expect exactly 1 entry
    assert_eq!(
        raw_response_array.len(),
        1,
        "Shorthand model has a single provider, expected 1 entry, got {}",
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
// T0 native /inference endpoint -- OpenAI Responses API -- include_raw_response=true
// =============================================================================

#[tokio::test(flavor = "multi_thread")]
async fn test_raw_response_error_inference_responses_non_streaming() {
    let payload = json!({
        "model_name": "openai::responses::gpt-5-mini",
        "input": {
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": "Error test inference responses"}]
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

    // Single provider -- expect exactly 1 entry
    assert_eq!(
        raw_response_array.len(),
        1,
        "Shorthand model has a single provider, expected 1 entry, got {}",
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
        "responses",
        "API type should be `responses`"
    );
}

// =============================================================================
// OAI-compatible /openai/v1/chat/completions -- OpenAI Responses API -- include_raw_response=true
// =============================================================================

#[tokio::test(flavor = "multi_thread")]
async fn test_raw_response_error_openai_responses_non_streaming() {
    let payload = json!({
        "model": "tensorzero::model_name::openai::responses::gpt-5-mini",
        "messages": [
            {
                "role": "user",
                "content": "Error test openai responses"
            }
        ],
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

    // Single provider -- expect exactly 1 entry
    assert_eq!(
        raw_response_array.len(),
        1,
        "Shorthand model has a single provider, expected 1 entry, got {}",
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
        "responses",
        "API type should be `responses`"
    );
}

// =============================================================================
// OAI-compatible /openai/v1/chat/completions -- include_raw_response not set
// =============================================================================

#[tokio::test(flavor = "multi_thread")]
async fn test_raw_response_error_openai_not_requested() {
    let payload = json!({
        "model": "tensorzero::model_name::openai::gpt-5-mini",
        "messages": [
            {
                "role": "user",
                "content": "Error test openai not requested"
            }
        ],
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
