//! E2E tests for the model inferences endpoint.

use reqwest::Client;
use tensorzero_core::endpoints::internal::model_inferences::GetModelInferencesResponse;
use tensorzero_core::endpoints::stored_inferences::v1::types::{
    GetInferencesResponse, ListInferencesRequest,
};
use uuid::Uuid;

use crate::common::get_gateway_endpoint;

/// Helper function to list inferences to get an inference_id
async fn get_available_inference_id(request: ListInferencesRequest) -> Uuid {
    let http_client = Client::new();
    let resp = http_client
        .post(get_gateway_endpoint("/v1/inferences/list_inferences"))
        .json(&request)
        .send()
        .await
        .unwrap();

    assert!(
        resp.status().is_success(),
        "get_available_inference_id request failed: status={:?}",
        resp.status(),
    );

    let response: GetInferencesResponse = resp.json().await.unwrap();
    response
        .inferences
        .first()
        .expect("Expected at least one inference")
        .id()
}

#[tokio::test(flavor = "multi_thread")]
async fn test_get_model_inferences_for_chat_inference() {
    // First, get an inference_id from the list_inferences endpoint
    let inference_id = get_available_inference_id(ListInferencesRequest {
        function_name: Some("write_haiku".to_string()),
        limit: Some(1),
        ..Default::default()
    })
    .await;

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
    // Get an inference_id for a JSON function
    let inference_id = get_available_inference_id(ListInferencesRequest {
        function_name: Some("extract_entities".to_string()),
        limit: Some(1),
        ..Default::default()
    })
    .await;

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
