//! E2E tests for the count matching inferences endpoint.

use reqwest::Client;
use serde_json::json;
use tensorzero::{
    ClientInferenceParams, InferenceResponse, Input, InputMessage, InputMessageContent, Role,
    System,
};
use tokio::time::{Duration, sleep};
use uuid::Uuid;

use crate::common::get_gateway_endpoint;

use tensorzero_core::{
    db::datasets::DatasetOutputSource,
    endpoints::datasets::{
        DatapointKind,
        internal::{CountMatchingInferencesResponse, FilterInferencesForDatasetBuilderRequest},
    },
    inference::types::{Arguments, Text},
};

/// Helper function to make an inference
async fn make_inference(client: &Client, function_name: &str, variant_name: &str) -> Uuid {
    let payload = ClientInferenceParams {
        function_name: Some(function_name.to_string()),
        variant_name: Some(variant_name.to_string()),
        input: Input {
            system: Some(System::Template(Arguments(
                json!({"assistant_name": "TestBot"})
                    .as_object()
                    .unwrap()
                    .clone(),
            ))),
            messages: vec![InputMessage {
                role: Role::User,
                content: vec![InputMessageContent::Text(Text {
                    text: format!("Test message {}", Uuid::now_v7()),
                })],
            }],
        },
        ..Default::default()
    };

    let response = client
        .post(get_gateway_endpoint("/inference"))
        .json(&payload)
        .send()
        .await
        .unwrap();

    let status = response.status();
    let body = response.text().await.unwrap();
    assert!(
        status.is_success(),
        "Failed to make inference: {status} {body}"
    );

    let inference_response: InferenceResponse =
        serde_json::from_str(&body).expect("Failed to deserialize inference response");

    match inference_response {
        InferenceResponse::Chat(chat) => chat.inference_id,
        InferenceResponse::Json(json) => json.inference_id,
    }
}

/// Helper function to count matching inferences
async fn count_matching_inferences(
    client: &Client,
    inference_type: DatapointKind,
    function_name: Option<&str>,
    variant_name: Option<&str>,
    output_source: DatasetOutputSource,
) -> CountMatchingInferencesResponse {
    let payload = FilterInferencesForDatasetBuilderRequest {
        inference_type,
        function_name: function_name.map(String::from),
        variant_name: variant_name.map(String::from),
        output_source,
        metric_filter: None,
    };

    let response = client
        .post(get_gateway_endpoint("/internal/inferences/count"))
        .json(&payload)
        .send()
        .await
        .unwrap();

    let status = response.status();
    let body = response.text().await.unwrap();
    assert!(
        status.is_success(),
        "Expected success status, got {status}: {body}"
    );

    serde_json::from_str(&body).unwrap()
}

#[tokio::test(flavor = "multi_thread")]
async fn test_count_matching_inferences_filters_by_variant() {
    let client = Client::new();

    // Get initial counts
    let initial_test_count = count_matching_inferences(
        &client,
        DatapointKind::Chat,
        Some("basic_test"),
        Some("test"),
        DatasetOutputSource::None,
    )
    .await;
    let initial_prometheus_count = count_matching_inferences(
        &client,
        DatapointKind::Chat,
        Some("basic_test"),
        Some("prometheus"),
        DatasetOutputSource::None,
    )
    .await;

    // Make an inference with the "test" variant
    make_inference(&client, "basic_test", "test").await;

    // Wait for ClickHouse to process the insertion
    sleep(Duration::from_millis(1000)).await;

    // Count for "test" variant should have increased
    let new_test_count = count_matching_inferences(
        &client,
        DatapointKind::Chat,
        Some("basic_test"),
        Some("test"),
        DatasetOutputSource::None,
    )
    .await;
    assert_eq!(
        new_test_count.count,
        initial_test_count.count + 1,
        "Expected test variant count to increase by 1"
    );

    // Count for "prometheus" variant should remain the same
    let new_prometheus_count = count_matching_inferences(
        &client,
        DatapointKind::Chat,
        Some("basic_test"),
        Some("prometheus"),
        DatasetOutputSource::None,
    )
    .await;
    assert_eq!(
        new_prometheus_count.count, initial_prometheus_count.count,
        "Expected prometheus variant count to remain the same"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn test_count_matching_inferences_nonexistent_function_returns_zero() {
    let client = Client::new();

    let count = count_matching_inferences(
        &client,
        DatapointKind::Chat,
        Some("nonexistent_function_12345"),
        None,
        DatasetOutputSource::None,
    )
    .await;

    assert_eq!(count.count, 0, "Expected 0 for nonexistent function");
}

#[tokio::test(flavor = "multi_thread")]
async fn test_count_matching_inferences_with_output_source_inference() {
    let client = Client::new();

    // Count with output_source = "inference" should work
    let count = count_matching_inferences(
        &client,
        DatapointKind::Chat,
        Some("basic_test"),
        None,
        DatasetOutputSource::Inference,
    )
    .await;

    // Just verify it returns a valid response (the count depends on test state)
    assert!(count.count > 0, "Expect at least 1 inference");
}

#[tokio::test(flavor = "multi_thread")]
async fn test_count_matching_inferences_requires_function_for_variant() {
    let client = Client::new();

    // Trying to filter by variant without function should fail
    let payload = FilterInferencesForDatasetBuilderRequest {
        inference_type: DatapointKind::Chat,
        function_name: None,
        variant_name: Some("test".to_string()),
        output_source: DatasetOutputSource::None,
        metric_filter: None,
    };

    let response = client
        .post(get_gateway_endpoint("/internal/inferences/count"))
        .json(&payload)
        .send()
        .await
        .unwrap();

    // This should return an error because variant_name requires function_name
    assert!(
        !response.status().is_success(),
        "Expected error when variant_name is provided without function_name"
    );
}
