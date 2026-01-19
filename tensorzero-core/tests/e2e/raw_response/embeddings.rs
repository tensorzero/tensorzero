//! E2E tests for `tensorzero::include_raw_response` parameter on embeddings endpoint.
//!
//! Tests that raw provider-specific response data is correctly returned when requested
//! for embedding models.

use reqwest::{Client, StatusCode};
use serde_json::{Value, json};

use crate::common::get_gateway_endpoint;

/// Helper to assert raw_response entry structure is valid for embeddings
fn assert_raw_response_entry(entry: &Value) {
    assert!(
        entry.get("model_inference_id").is_some(),
        "raw_response entry should have model_inference_id"
    );
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

    // Verify api_type is "embeddings"
    let api_type = entry.get("api_type").unwrap().as_str().unwrap();
    assert_eq!(
        api_type, "embeddings",
        "Embeddings endpoint should have api_type 'embeddings', got: {api_type}"
    );

    // Verify data is a string (raw response from provider)
    assert!(
        entry.get("data").unwrap().is_string(),
        "data should be a string (raw response from provider)"
    );
}

// =============================================================================
// Basic Embeddings Raw Response Tests
// =============================================================================

#[tokio::test]
async fn e2e_test_embeddings_raw_response_requested() {
    let payload = json!({
        "input": "Hello, world!",
        "model": "tensorzero::embedding_model_name::text_embedding_3_small",
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
        StatusCode::OK,
        "Response should be successful"
    );

    let response_json: Value = response.json().await.unwrap();

    // Verify standard embedding response fields
    assert_eq!(
        response_json["object"].as_str().unwrap(),
        "list",
        "object should be 'list'"
    );
    assert!(
        !response_json["data"].as_array().unwrap().is_empty(),
        "data should have embeddings"
    );

    // Check tensorzero::raw_response exists
    let raw_response = response_json
        .get("tensorzero::raw_response")
        .expect("Response should have tensorzero::raw_response when include_raw_response=true");
    assert!(
        raw_response.is_array(),
        "tensorzero::raw_response should be an array"
    );

    let raw_response_array = raw_response.as_array().unwrap();
    assert!(
        !raw_response_array.is_empty(),
        "tensorzero::raw_response should have at least one entry"
    );

    // Validate entry structure
    let first_entry = &raw_response_array[0];
    assert_raw_response_entry(first_entry);

    // Provider type should be "openai" for text_embedding_3_small
    let provider_type = first_entry.get("provider_type").unwrap().as_str().unwrap();
    assert_eq!(provider_type, "openai", "Provider type should be 'openai'");

    // The data field should be a non-empty string
    let data = first_entry.get("data").unwrap().as_str().unwrap();
    assert!(!data.is_empty(), "data should not be empty");

    // Verify data is valid JSON containing embedding response
    let data_json: Value = serde_json::from_str(data).expect("data should be valid JSON");
    assert!(
        data_json.get("data").is_some(),
        "raw response should contain 'data' field"
    );
}

#[tokio::test]
async fn e2e_test_embeddings_raw_response_not_requested() {
    let payload = json!({
        "input": "Hello, world!",
        "model": "tensorzero::embedding_model_name::text_embedding_3_small"
        // tensorzero::include_raw_response is NOT set (defaults to false)
    });

    let response = Client::new()
        .post(get_gateway_endpoint("/openai/v1/embeddings"))
        .json(&payload)
        .send()
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);

    let response_json: Value = response.json().await.unwrap();

    // tensorzero::raw_response should NOT be present when not requested
    assert!(
        response_json.get("tensorzero::raw_response").is_none(),
        "tensorzero::raw_response should not be present when not requested"
    );

    // Standard response fields should still be present
    assert_eq!(response_json["object"].as_str().unwrap(), "list");
    assert!(!response_json["data"].as_array().unwrap().is_empty());
}

#[tokio::test]
async fn e2e_test_embeddings_raw_response_explicitly_false() {
    let payload = json!({
        "input": "Hello, world!",
        "model": "tensorzero::embedding_model_name::text_embedding_3_small",
        "tensorzero::include_raw_response": false
    });

    let response = Client::new()
        .post(get_gateway_endpoint("/openai/v1/embeddings"))
        .json(&payload)
        .send()
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);

    let response_json: Value = response.json().await.unwrap();

    // tensorzero::raw_response should NOT be present when explicitly false
    assert!(
        response_json.get("tensorzero::raw_response").is_none(),
        "tensorzero::raw_response should not be present when explicitly set to false"
    );
}

// =============================================================================
// Bulk Embeddings Tests
// =============================================================================

#[tokio::test]
async fn e2e_test_embeddings_raw_response_batch() {
    let inputs = vec![
        "Hello, world!",
        "How are you today?",
        "This is a test of batch embeddings.",
    ];
    let payload = json!({
        "input": inputs,
        "model": "tensorzero::embedding_model_name::text_embedding_3_small",
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
        StatusCode::OK,
        "Response should be successful"
    );

    let response_json: Value = response.json().await.unwrap();

    // Verify standard embedding response fields
    assert_eq!(response_json["object"].as_str().unwrap(), "list");
    assert_eq!(
        response_json["data"].as_array().unwrap().len(),
        inputs.len(),
        "Should have embedding for each input"
    );

    // Check tensorzero::raw_response exists
    let raw_response = response_json
        .get("tensorzero::raw_response")
        .expect("Response should have tensorzero::raw_response when include_raw_response=true");

    let raw_response_array = raw_response.as_array().unwrap();

    // Should have exactly one entry (one API call for batch)
    assert_eq!(
        raw_response_array.len(),
        1,
        "Batch embedding should have exactly 1 raw_response entry (single API call)"
    );

    // Validate entry structure
    assert_raw_response_entry(&raw_response_array[0]);
}

// =============================================================================
// Cache Interaction Tests
// =============================================================================

#[tokio::test]
async fn e2e_test_embeddings_raw_response_with_cache() {
    let input_text = format!(
        "This is a cache test for embeddings raw_response - {}",
        rand::random::<u32>()
    );

    // First request: populate cache with raw_response enabled
    let payload = json!({
        "input": input_text,
        "model": "tensorzero::embedding_model_name::text_embedding_3_small",
        "tensorzero::include_raw_response": true,
        "tensorzero::cache_options": {
            "enabled": "on",
            "max_age_s": 60
        }
    });

    let response = Client::new()
        .post(get_gateway_endpoint("/openai/v1/embeddings"))
        .json(&payload)
        .send()
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    let response_json: Value = response.json().await.unwrap();

    // First request should have raw_response with data
    let raw_response = response_json
        .get("tensorzero::raw_response")
        .expect("First request should have tensorzero::raw_response");
    let raw_response_array = raw_response.as_array().unwrap();
    assert!(
        !raw_response_array.is_empty(),
        "First (non-cached) request should have raw_response entries"
    );
    assert_raw_response_entry(&raw_response_array[0]);

    // Usage should be non-zero for first request
    assert!(
        response_json["usage"]["prompt_tokens"].as_u64().unwrap() > 0,
        "First request should have non-zero prompt_tokens"
    );

    // Wait for cache write to complete
    tokio::time::sleep(std::time::Duration::from_secs(2)).await;

    // Second request: should hit cache
    let response_cached = Client::new()
        .post(get_gateway_endpoint("/openai/v1/embeddings"))
        .json(&payload)
        .send()
        .await
        .unwrap();

    assert_eq!(response_cached.status(), StatusCode::OK);
    let response_cached_json: Value = response_cached.json().await.unwrap();

    // Cached response should have tensorzero::raw_response but with empty array
    let raw_response_cached = response_cached_json
        .get("tensorzero::raw_response")
        .expect("Cached request should still have tensorzero::raw_response when requested");
    let raw_response_cached_array = raw_response_cached.as_array().unwrap();
    assert!(
        raw_response_cached_array.is_empty(),
        "Cached response should have empty tensorzero::raw_response array"
    );

    // Usage should be zero for cached request
    assert_eq!(
        response_cached_json["usage"]["prompt_tokens"]
            .as_u64()
            .unwrap(),
        0,
        "Cached request should have zero prompt_tokens"
    );
}

// =============================================================================
// Entry Structure Validation Tests
// =============================================================================

#[tokio::test]
async fn e2e_test_embeddings_raw_response_entry_structure() {
    let payload = json!({
        "input": "Test entry structure",
        "model": "tensorzero::embedding_model_name::text_embedding_3_small",
        "tensorzero::include_raw_response": true
    });

    let response = Client::new()
        .post(get_gateway_endpoint("/openai/v1/embeddings"))
        .json(&payload)
        .send()
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    let response_json: Value = response.json().await.unwrap();

    let raw_response = response_json
        .get("tensorzero::raw_response")
        .expect("Response should have tensorzero::raw_response");
    let raw_response_array = raw_response.as_array().unwrap();
    assert!(!raw_response_array.is_empty());

    let entry = &raw_response_array[0];

    // Check model_inference_id is a valid UUID string
    let model_inference_id = entry.get("model_inference_id").unwrap().as_str().unwrap();
    assert!(
        uuid::Uuid::parse_str(model_inference_id).is_ok(),
        "model_inference_id should be a valid UUID"
    );

    // Check provider_type is a non-empty string
    let provider_type = entry.get("provider_type").unwrap().as_str().unwrap();
    assert!(
        !provider_type.is_empty(),
        "provider_type should be non-empty"
    );

    // Check api_type is "embeddings"
    let api_type = entry.get("api_type").unwrap().as_str().unwrap();
    assert_eq!(
        api_type, "embeddings",
        "api_type should be 'embeddings' for embeddings endpoint"
    );

    // Check data is a non-empty string containing valid JSON
    let data = entry.get("data").unwrap().as_str().unwrap();
    assert!(!data.is_empty(), "data should be a non-empty string");
    let _data_json: Value = serde_json::from_str(data).expect("data should be valid JSON");
}

// =============================================================================
// Dimensions Parameter Tests
// =============================================================================

#[tokio::test]
async fn e2e_test_embeddings_raw_response_with_dimensions() {
    let payload = json!({
        "input": "Test with specific dimensions",
        "model": "tensorzero::embedding_model_name::text_embedding_3_small",
        "dimensions": 512,
        "tensorzero::include_raw_response": true
    });

    let response = Client::new()
        .post(get_gateway_endpoint("/openai/v1/embeddings"))
        .json(&payload)
        .send()
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    let response_json: Value = response.json().await.unwrap();

    // Verify dimensions are respected in the embedding
    assert_eq!(
        response_json["data"][0]["embedding"]
            .as_array()
            .unwrap()
            .len(),
        512,
        "Embedding should have 512 dimensions"
    );

    // Check tensorzero::raw_response exists
    let raw_response = response_json.get("tensorzero::raw_response").unwrap();
    let raw_response_array = raw_response.as_array().unwrap();
    assert!(!raw_response_array.is_empty());
    assert_raw_response_entry(&raw_response_array[0]);
}
