//! E2E tests for the model inferences endpoint.

use reqwest::Client;
use tensorzero_core::endpoints::internal::model_inferences::GetModelInferencesResponse;
use uuid::Uuid;

use crate::common::get_gateway_endpoint;

#[tokio::test(flavor = "multi_thread")]
async fn test_get_model_inferences_for_chat_inference() {
    // Use a hardcoded inference ID known to have model inferences
    let inference_id = Uuid::parse_str("0196c682-72e0-7c83-a92b-9d1a3c7630f2").expect("Valid UUID");

    // Now get model inferences for this inference_id
    let http_client = Client::new();
    let url = get_gateway_endpoint(&format!("/internal/model_inferences/{inference_id}"));

    let resp = http_client.get(url).send().await.unwrap();
    assert!(
        resp.status().is_success(),
        "get_model_inferences request failed: status={:?}",
        resp.status()
    );

    let response: GetModelInferencesResponse = resp.json().await.unwrap();

    // Each inference should have at least one model inference
    assert!(
        !response.model_inferences.is_empty(),
        "Expected at least one model inference for inference_id {inference_id}"
    );

    // Verify the model inferences have the correct inference_id
    for model_inference in &response.model_inferences {
        assert_eq!(
            model_inference.inference_id, inference_id,
            "Model inference should have the queried inference_id"
        );
        // Verify required fields are present
        assert!(!model_inference.model_name.is_empty());
        assert!(!model_inference.model_provider_name.is_empty());
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn test_get_model_inferences_for_json_inference() {
    // Use a hardcoded extract_entities inference ID known to have model inferences
    let inference_id = Uuid::parse_str("0196374c-2c6d-7ce0-b508-e3b24ee4579c").expect("Valid UUID");

    // Get model inferences
    let http_client = Client::new();
    let url = get_gateway_endpoint(&format!("/internal/model_inferences/{inference_id}"));

    let resp = http_client.get(url).send().await.unwrap();
    assert!(
        resp.status().is_success(),
        "get_model_inferences request failed: status={:?}",
        resp.status()
    );

    let response: GetModelInferencesResponse = resp.json().await.unwrap();

    assert!(
        !response.model_inferences.is_empty(),
        "Expected at least one model inference for inference_id {inference_id}"
    );

    for model_inference in &response.model_inferences {
        assert_eq!(model_inference.inference_id, inference_id);
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn test_get_model_inferences_nonexistent_id() {
    // Use a random UUID that doesn't exist
    let nonexistent_id = Uuid::now_v7();

    let http_client = Client::new();
    let url = get_gateway_endpoint(&format!("/internal/model_inferences/{nonexistent_id}"));

    let resp = http_client.get(url).send().await.unwrap();
    assert!(resp.status().is_success());

    let response: GetModelInferencesResponse = resp.json().await.unwrap();

    // Should return empty array for non-existent inference_id
    assert!(
        response.model_inferences.is_empty(),
        "Expected empty model_inferences for non-existent inference_id"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn test_get_model_inferences_invalid_uuid() {
    let http_client = Client::new();
    let url = get_gateway_endpoint("/internal/model_inferences/not-a-valid-uuid");

    let resp = http_client.get(url).send().await.unwrap();

    // Should return an error for invalid UUID
    assert!(
        !resp.status().is_success(),
        "Expected error for invalid UUID"
    );
}
