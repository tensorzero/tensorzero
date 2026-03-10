//! E2E tests for `raw_response` in embeddings endpoint error responses.
//!
//! Tests that raw provider-specific response data from failed embedding providers
//! is correctly included in error responses from the `/openai/v1/embeddings` endpoint.

use reqwest::Client;
use serde_json::{Value, json};

use super::assert_error_raw_response_entry;
use crate::common::get_gateway_endpoint;

// =============================================================================
// Embeddings Endpoint — All Providers Fail
// =============================================================================

/// All embedding providers fail with include_raw_response=true.
/// Expects error response with `tensorzero_raw_response` entries from both failed providers.
#[tokio::test(flavor = "multi_thread")]
async fn test_embeddings_error_with_raw_response() {
    let payload = json!({
        "input": "Hello, world!",
        "model": "tensorzero::embedding_model_name::embedding_both_error_with_raw_response",
        "tensorzero::include_raw_response": true
    });

    let response = Client::new()
        .post(get_gateway_endpoint("/openai/v1/embeddings"))
        .json(&payload)
        .send()
        .await
        .unwrap();

    assert!(
        !response.status().is_success(),
        "Response should be an error (all embedding providers failed), got status {}",
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
        2,
        "Should have 2 entries from 2 failed embedding providers"
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

/// All embedding providers fail without include_raw_response.
/// Expects error response without `tensorzero_raw_response` field.
#[tokio::test(flavor = "multi_thread")]
async fn test_embeddings_error_without_raw_response() {
    let payload = json!({
        "input": "Hello, world!",
        "model": "tensorzero::embedding_model_name::embedding_both_error_with_raw_response"
        // tensorzero::include_raw_response is NOT set (defaults to false)
    });

    let response = Client::new()
        .post(get_gateway_endpoint("/openai/v1/embeddings"))
        .json(&payload)
        .send()
        .await
        .unwrap();

    assert!(
        !response.status().is_success(),
        "Response should be an error (all embedding providers failed)"
    );

    let response_json: Value = response.json().await.unwrap();

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
// Embeddings Endpoint — Partial Failure (Fallback Success)
// =============================================================================

/// Embedding model with fallback: first provider errors with raw_response, second succeeds.
/// Expects success response with `tensorzero::raw_response` containing both error and success entries.
#[tokio::test(flavor = "multi_thread")]
async fn test_embeddings_fallback_with_raw_response() {
    let payload = json!({
        "input": "Hello, world!",
        "model": "tensorzero::embedding_model_name::embedding_fallback_with_raw_response",
        "tensorzero::include_raw_response": true
    });

    let response = Client::new()
        .post(get_gateway_endpoint("/openai/v1/embeddings"))
        .json(&payload)
        .send()
        .await
        .unwrap();

    assert_eq!(
        response.status(),
        reqwest::StatusCode::OK,
        "Response should be successful (fallback to good provider)"
    );

    let response_json: Value = response.json().await.unwrap();

    // Verify standard embedding response fields
    assert_eq!(
        response_json["object"].as_str().unwrap(),
        "list",
        "object should be `list`"
    );

    // Check tensorzero::raw_response exists
    let raw_response = response_json
        .get("tensorzero::raw_response")
        .expect("Response should have tensorzero::raw_response when include_raw_response=true");
    let raw_response_array = raw_response.as_array().unwrap();
    assert_eq!(
        raw_response_array.len(),
        2,
        "raw_response should have 2 entries: 1 error + 1 success"
    );

    // First entry: failed provider (error entry — no model_inference_id)
    let error_entry = &raw_response_array[0];
    assert_error_raw_response_entry(error_entry);
    assert_eq!(
        error_entry["provider_type"].as_str().unwrap(),
        "dummy",
        "Failed provider type should be `dummy`"
    );
    assert_eq!(
        error_entry["api_type"].as_str().unwrap(),
        "embeddings",
        "Failed entry should have api_type `embeddings`"
    );
    assert_eq!(
        error_entry["data"].as_str().unwrap(),
        "dummy error raw response",
        "Failed provider data should be `dummy error raw response`"
    );

    // Second entry: successful provider (has model_inference_id)
    let success_entry = &raw_response_array[1];
    assert!(
        success_entry.get("model_inference_id").is_some(),
        "Successful entry should have model_inference_id"
    );
    assert_eq!(
        success_entry["provider_type"].as_str().unwrap(),
        "dummy",
        "Successful provider type should be `dummy`"
    );
    assert_eq!(
        success_entry["api_type"].as_str().unwrap(),
        "embeddings",
        "Successful entry should have api_type `embeddings`"
    );
}
