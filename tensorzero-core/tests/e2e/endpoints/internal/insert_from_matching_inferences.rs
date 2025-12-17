//! E2E tests for the insert from matching inferences endpoint.

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
        internal::{
            FilterInferencesForDatasetBuilderRequest, GetDatapointCountResponse,
            InsertFromMatchingInferencesResponse,
        },
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

/// Helper function to insert matching inferences into a dataset
async fn insert_from_matching_inferences(
    client: &Client,
    dataset_name: &str,
    inference_type: DatapointKind,
    function_name: Option<&str>,
    variant_name: Option<&str>,
    output_source: DatasetOutputSource,
) -> InsertFromMatchingInferencesResponse {
    let payload = FilterInferencesForDatasetBuilderRequest {
        inference_type,
        function_name: function_name.map(String::from),
        variant_name: variant_name.map(String::from),
        output_source,
        metric_filter: None,
    };

    let response = client
        .post(get_gateway_endpoint(&format!(
            "/internal/datasets/{dataset_name}/datapoints/insert-inferences"
        )))
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

/// Helper function to get the datapoint count for a dataset
async fn get_datapoint_count(client: &Client, dataset_name: &str) -> GetDatapointCountResponse {
    let response = client
        .get(get_gateway_endpoint(&format!(
            "/internal/datasets/{dataset_name}/datapoints/count"
        )))
        .send()
        .await
        .unwrap();

    let status = response.status();
    let body = response.text().await.unwrap();
    assert!(
        status.is_success(),
        "Failed to get datapoint count: {status} {body}"
    );

    serde_json::from_str(&body).expect("Failed to deserialize datapoint count response")
}

#[tokio::test(flavor = "multi_thread")]
async fn test_insert_from_matching_inferences_basic() {
    let client = Client::new();
    let dataset_name = format!("test_insert_dataset_{}", Uuid::now_v7());

    // Make an inference first
    make_inference(&client, "basic_test", "test").await;

    // Wait for ClickHouse to process the insertion
    sleep(Duration::from_millis(1000)).await;

    // Insert inferences matching basic_test function into the dataset
    let result = insert_from_matching_inferences(
        &client,
        &dataset_name,
        DatapointKind::Chat,
        Some("basic_test"),
        None,
        DatasetOutputSource::Inference,
    )
    .await;

    // Should have inserted at least one row
    assert!(
        result.rows_inserted > 0,
        "Expected at least one row to be inserted"
    );

    // Wait for the insert to be visible
    sleep(Duration::from_millis(1000)).await;

    // Verify the dataset has datapoints
    let count_response = get_datapoint_count(&client, &dataset_name).await;
    assert!(
        count_response.datapoint_count > 0,
        "Expected dataset to have datapoints"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn test_insert_from_matching_inferences_filters_by_variant() {
    let client = Client::new();
    let dataset_name = format!("test_insert_variant_{}", Uuid::now_v7());

    // Make an inference with the "test" variant
    make_inference(&client, "basic_test", "test").await;

    // Wait for ClickHouse to process the insertion
    sleep(Duration::from_millis(1000)).await;

    // Insert inferences matching only the "test" variant
    let result = insert_from_matching_inferences(
        &client,
        &dataset_name,
        DatapointKind::Chat,
        Some("basic_test"),
        Some("test"),
        DatasetOutputSource::Inference,
    )
    .await;

    // Should have inserted at least one row
    assert!(
        result.rows_inserted > 0,
        "Expected at least one row to be inserted for 'test' variant"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn test_insert_from_matching_inferences_nonexistent_function_inserts_zero() {
    let client = Client::new();
    let dataset_name = format!("test_insert_nonexistent_{}", Uuid::now_v7());

    let result = insert_from_matching_inferences(
        &client,
        &dataset_name,
        DatapointKind::Chat,
        Some("nonexistent_function_12345"),
        None,
        DatasetOutputSource::None,
    )
    .await;

    assert_eq!(
        result.rows_inserted, 0,
        "Expected 0 rows inserted for nonexistent function"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn test_insert_from_matching_inferences_requires_function_for_variant() {
    let client = Client::new();
    let dataset_name = format!("test_insert_invalid_{}", Uuid::now_v7());

    // Trying to filter by variant without function should fail
    let payload = FilterInferencesForDatasetBuilderRequest {
        inference_type: DatapointKind::Chat,
        function_name: None,
        variant_name: Some("test".to_string()),
        output_source: DatasetOutputSource::None,
        metric_filter: None,
    };

    let response = client
        .post(get_gateway_endpoint(&format!(
            "/internal/datasets/{dataset_name}/datapoints/insert-inferences"
        )))
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
