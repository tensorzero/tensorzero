//! E2E tests for `tensorzero::include_raw_response` parameter on embeddings endpoint.
//!
//! Tests that raw provider-specific response data is correctly returned when requested
//! for embedding models.

use axum::body::Body;
use axum::extract::State;
use axum::response::Response;
use http_body_util::BodyExt;
use serde_json::Value;
use tensorzero::ClientExt;
use tensorzero::test_helpers::make_embedded_gateway_e2e_with_unique_db;
use tensorzero_core::embeddings::{EmbeddingEncodingFormat, EmbeddingInput};
use tensorzero_core::endpoints::inference::InferenceCredentials;
use tensorzero_core::endpoints::openai_compatible::OpenAIStructuredJson;
use tensorzero_core::endpoints::openai_compatible::embeddings::embeddings_handler;
use tensorzero_core::endpoints::openai_compatible::types::embeddings::OpenAICompatibleEmbeddingParams;

/// Helper to extract JSON body from a Response<Body>
async fn response_to_json(response: Response<Body>) -> Value {
    let body = response.into_body();
    let bytes = body.collect().await.unwrap().to_bytes();
    serde_json::from_slice(&bytes).unwrap()
}

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

#[tokio::test(flavor = "multi_thread")]
async fn test_embeddings_raw_response_requested() {
    let client = make_embedded_gateway_e2e_with_unique_db("emb_raw_response_requested").await;
    let state = client.get_app_state_data().unwrap().clone();

    let params = OpenAICompatibleEmbeddingParams {
        input: EmbeddingInput::Single("Hello, world!".to_string()),
        model: "tensorzero::embedding_model_name::text-embedding-3-small".to_string(),
        dimensions: None,
        encoding_format: EmbeddingEncodingFormat::Float,
        tensorzero_credentials: InferenceCredentials::default(),
        tensorzero_dryrun: None,
        tensorzero_cache_options: None,
        tensorzero_include_raw_response: true,
    };

    let response = embeddings_handler(State(state), None, OpenAIStructuredJson(params))
        .await
        .expect("Response should be successful");

    let response_json: Value = response_to_json(response).await;

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

    // Provider type should be "openai" for text-embedding-3-small
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

#[tokio::test(flavor = "multi_thread")]
async fn test_embeddings_raw_response_not_requested() {
    let client = make_embedded_gateway_e2e_with_unique_db("emb_raw_response_not_requested").await;
    let state = client.get_app_state_data().unwrap().clone();

    let params = OpenAICompatibleEmbeddingParams {
        input: EmbeddingInput::Single("Hello, world!".to_string()),
        model: "tensorzero::embedding_model_name::text-embedding-3-small".to_string(),
        dimensions: None,
        encoding_format: EmbeddingEncodingFormat::Float,
        tensorzero_credentials: InferenceCredentials::default(),
        tensorzero_dryrun: None,
        tensorzero_cache_options: None,
        tensorzero_include_raw_response: false, // not requested
    };

    let response = embeddings_handler(State(state), None, OpenAIStructuredJson(params))
        .await
        .expect("Response should be successful");

    let response_json: Value = response_to_json(response).await;

    // tensorzero::raw_response should NOT be present when not requested
    assert!(
        response_json.get("tensorzero::raw_response").is_none(),
        "tensorzero::raw_response should not be present when not requested"
    );

    // Standard response fields should still be present
    assert_eq!(response_json["object"].as_str().unwrap(), "list");
    assert!(!response_json["data"].as_array().unwrap().is_empty());
}

#[tokio::test(flavor = "multi_thread")]
async fn test_embeddings_raw_response_explicitly_false() {
    let client =
        make_embedded_gateway_e2e_with_unique_db("emb_raw_response_explicitly_false").await;
    let state = client.get_app_state_data().unwrap().clone();

    let params = OpenAICompatibleEmbeddingParams {
        input: EmbeddingInput::Single("Hello, world!".to_string()),
        model: "tensorzero::embedding_model_name::text-embedding-3-small".to_string(),
        dimensions: None,
        encoding_format: EmbeddingEncodingFormat::Float,
        tensorzero_credentials: InferenceCredentials::default(),
        tensorzero_dryrun: None,
        tensorzero_cache_options: None,
        tensorzero_include_raw_response: false,
    };

    let response = embeddings_handler(State(state), None, OpenAIStructuredJson(params))
        .await
        .expect("Response should be successful");

    let response_json: Value = response_to_json(response).await;

    // tensorzero::raw_response should NOT be present when explicitly false
    assert!(
        response_json.get("tensorzero::raw_response").is_none(),
        "tensorzero::raw_response should not be present when explicitly set to false"
    );
}

// =============================================================================
// Bulk Embeddings Tests
// =============================================================================

#[tokio::test(flavor = "multi_thread")]
async fn test_embeddings_raw_response_batch() {
    let client = make_embedded_gateway_e2e_with_unique_db("emb_raw_response_batch").await;
    let state = client.get_app_state_data().unwrap().clone();

    let inputs = [
        "Hello, world!",
        "How are you today?",
        "This is a test of batch embeddings.",
    ];
    let params = OpenAICompatibleEmbeddingParams {
        input: EmbeddingInput::Batch(inputs.iter().map(|s| s.to_string()).collect()),
        model: "tensorzero::embedding_model_name::text-embedding-3-small".to_string(),
        dimensions: None,
        encoding_format: EmbeddingEncodingFormat::Float,
        tensorzero_credentials: InferenceCredentials::default(),
        tensorzero_dryrun: None,
        tensorzero_cache_options: None,
        tensorzero_include_raw_response: true,
    };

    let response = embeddings_handler(State(state), None, OpenAIStructuredJson(params))
        .await
        .expect("Response should be successful");

    let response_json: Value = response_to_json(response).await;

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

#[tokio::test(flavor = "multi_thread")]
async fn test_embeddings_raw_response_with_cache() {
    let client = make_embedded_gateway_e2e_with_unique_db("emb_raw_response_with_cache").await;
    let state = client.get_app_state_data().unwrap().clone();

    let input_text = format!(
        "This is a cache test for embeddings raw_response - {}",
        rand::random::<u32>()
    );

    // First request: populate cache with raw_response enabled
    let params = OpenAICompatibleEmbeddingParams {
        input: EmbeddingInput::Single(input_text.clone()),
        model: "tensorzero::embedding_model_name::text-embedding-3-small".to_string(),
        dimensions: None,
        encoding_format: EmbeddingEncodingFormat::Float,
        tensorzero_credentials: InferenceCredentials::default(),
        tensorzero_dryrun: None,
        tensorzero_cache_options: Some(tensorzero_core::cache::CacheParamsOptions {
            enabled: tensorzero_core::cache::CacheEnabledMode::On,
            max_age_s: Some(60),
        }),
        tensorzero_include_raw_response: true,
    };

    let response = embeddings_handler(State(state.clone()), None, OpenAIStructuredJson(params))
        .await
        .expect("Response should be successful");

    let response_json: Value = response_to_json(response).await;

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
    let params_cached = OpenAICompatibleEmbeddingParams {
        input: EmbeddingInput::Single(input_text),
        model: "tensorzero::embedding_model_name::text-embedding-3-small".to_string(),
        dimensions: None,
        encoding_format: EmbeddingEncodingFormat::Float,
        tensorzero_credentials: InferenceCredentials::default(),
        tensorzero_dryrun: None,
        tensorzero_cache_options: Some(tensorzero_core::cache::CacheParamsOptions {
            enabled: tensorzero_core::cache::CacheEnabledMode::On,
            max_age_s: Some(60),
        }),
        tensorzero_include_raw_response: true,
    };

    let response_cached =
        embeddings_handler(State(state), None, OpenAIStructuredJson(params_cached))
            .await
            .expect("Cached response should be successful");

    let response_cached_json: Value = response_to_json(response_cached).await;

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

#[tokio::test(flavor = "multi_thread")]
async fn test_embeddings_raw_response_entry_structure() {
    let client = make_embedded_gateway_e2e_with_unique_db("emb_raw_response_entry_structure").await;
    let state = client.get_app_state_data().unwrap().clone();

    let params = OpenAICompatibleEmbeddingParams {
        input: EmbeddingInput::Single("Test entry structure".to_string()),
        model: "tensorzero::embedding_model_name::text-embedding-3-small".to_string(),
        dimensions: None,
        encoding_format: EmbeddingEncodingFormat::Float,
        tensorzero_credentials: InferenceCredentials::default(),
        tensorzero_dryrun: None,
        tensorzero_cache_options: None,
        tensorzero_include_raw_response: true,
    };

    let response = embeddings_handler(State(state), None, OpenAIStructuredJson(params))
        .await
        .expect("Response should be successful");

    let response_json: Value = response_to_json(response).await;

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

#[tokio::test(flavor = "multi_thread")]
async fn test_embeddings_raw_response_with_dimensions() {
    let client = make_embedded_gateway_e2e_with_unique_db("emb_raw_response_with_dimensions").await;
    let state = client.get_app_state_data().unwrap().clone();

    let params = OpenAICompatibleEmbeddingParams {
        input: EmbeddingInput::Single("Test with specific dimensions".to_string()),
        model: "tensorzero::embedding_model_name::text-embedding-3-small".to_string(),
        dimensions: Some(512),
        encoding_format: EmbeddingEncodingFormat::Float,
        tensorzero_credentials: InferenceCredentials::default(),
        tensorzero_dryrun: None,
        tensorzero_cache_options: None,
        tensorzero_include_raw_response: true,
    };

    let response = embeddings_handler(State(state), None, OpenAIStructuredJson(params))
        .await
        .expect("Response should be successful");

    let response_json: Value = response_to_json(response).await;

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

// =============================================================================
// Error Tests
// =============================================================================

#[tokio::test(flavor = "multi_thread")]
async fn test_embeddings_raw_response_error() {
    let client = make_embedded_gateway_e2e_with_unique_db("emb_raw_response_error").await;
    let state = client.get_app_state_data().unwrap().clone();

    let params = OpenAICompatibleEmbeddingParams {
        input: EmbeddingInput::Single("Hello, world!".to_string()),
        model: "tensorzero::embedding_model_name::error_with_raw_response".to_string(),
        dimensions: None,
        encoding_format: EmbeddingEncodingFormat::Float,
        tensorzero_credentials: InferenceCredentials::default(),
        tensorzero_dryrun: None,
        tensorzero_cache_options: None,
        tensorzero_include_raw_response: true,
    };

    let result = embeddings_handler(State(state), None, OpenAIStructuredJson(params)).await;

    // When include_raw_response is true, errors are returned as Ok(response) with error status
    let response = result.expect("Handler should return Ok when include_raw_response is true");

    // Should be an error status
    let status = response.status();
    assert!(
        !status.is_success(),
        "Response should be an error, got status: {status}"
    );

    // Parse the response body
    let body = response_to_json(response).await;

    // Should have raw_response in error response
    let raw_response = body
        .get("raw_response")
        .expect("Error should include raw_response when include_raw_response=true");
    assert!(raw_response.is_array(), "raw_response should be an array");

    let entries = raw_response.as_array().unwrap();
    assert!(
        !entries.is_empty(),
        "Should have at least one raw_response entry"
    );

    // Verify entry contains error data
    let entry = &entries[0];
    assert_eq!(
        entry.get("provider_type").and_then(|v| v.as_str()),
        Some("dummy"),
        "Provider type should be 'dummy'"
    );
    assert_eq!(
        entry.get("api_type").and_then(|v| v.as_str()),
        Some("embeddings"),
        "API type should be 'embeddings'"
    );

    let data = entry.get("data").and_then(|d| d.as_str()).unwrap();
    assert!(
        data.contains("embedding_test_error"),
        "raw_response data should contain error info"
    );
}
