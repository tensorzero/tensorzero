use reqwest::Client;
use uuid::Uuid;

use tensorzero_core::db::clickhouse::query_builder::{
    InferenceFilter, TagComparisonOperator, TagFilter,
};
use tensorzero_core::db::inferences::InferenceOutputSource;
use tensorzero_core::endpoints::datasets::v1::types::{
    CreateDatapointsFromInferenceRequest, CreateDatapointsFromInferenceRequestParams,
    CreateDatapointsResponse,
};
use tensorzero_core::endpoints::stored_inferences::v1::types::ListInferencesRequest;

use crate::common::get_gateway_endpoint;

#[tokio::test(flavor = "multi_thread")]
async fn test_create_from_inference_ids_success() {
    let client = Client::new();

    // Use hardcoded inference IDs known to have input and output data
    let inference_id1 =
        Uuid::parse_str("0196c682-72e0-7c83-a92b-9d1a3c7630f2").expect("Valid UUID");
    let inference_id2 =
        Uuid::parse_str("01963691-b040-7441-8069-44c9b2814f57").expect("Valid UUID");

    // Create datapoints from these inferences
    let request = CreateDatapointsFromInferenceRequest {
        params: CreateDatapointsFromInferenceRequestParams::InferenceIds {
            inference_ids: vec![inference_id1, inference_id2],
            output_source: Some(InferenceOutputSource::Inference),
        },
    };

    let response = client
        .post(get_gateway_endpoint(
            "/v1/datasets/test_dataset/from_inferences",
        ))
        .json(&request)
        .send()
        .await
        .unwrap();

    assert_eq!(response.status(), 200);
    let result: CreateDatapointsResponse = response.json().await.unwrap();

    assert_eq!(result.ids.len(), 2);
}

#[tokio::test]
async fn test_create_from_inference_query_success() {
    let client = Client::new();

    // Create datapoints using a query
    let request = CreateDatapointsFromInferenceRequest {
        params: CreateDatapointsFromInferenceRequestParams::InferenceQuery {
            query: Box::new(ListInferencesRequest {
                function_name: Some("write_haiku".to_string()),
                variant_name: Some("better_prompt_haiku_4_5".to_string()),
                output_source: InferenceOutputSource::Inference,
                ..Default::default()
            }),
        },
    };

    let response = client
        .post(get_gateway_endpoint(
            "/v1/datasets/test_dataset_query/from_inferences",
        ))
        .json(&request)
        .send()
        .await
        .unwrap();

    assert_eq!(response.status(), 200);
    let result: CreateDatapointsResponse = response.json().await.unwrap();

    // Should have created datapoints from existing inferences
    assert!(!result.ids.is_empty(), "Expected at least one datapoint");
}

#[tokio::test(flavor = "multi_thread")]
async fn test_create_from_same_inference_multiple_times_succeeds() {
    let client = Client::new();

    // Use a hardcoded inference ID known to have input and output data
    let inference_id = Uuid::parse_str("0196c682-72e0-7c83-a92b-9d1a3c7630f2").expect("Valid UUID");

    // Create datapoint from this inference
    let request = CreateDatapointsFromInferenceRequest {
        params: CreateDatapointsFromInferenceRequestParams::InferenceIds {
            inference_ids: vec![inference_id],
            output_source: Some(InferenceOutputSource::Inference),
        },
    };

    let response1 = client
        .post(get_gateway_endpoint(
            "/v1/datasets/test_dataset_dup/from_inferences",
        ))
        .json(&request)
        .send()
        .await
        .unwrap();
    assert_eq!(response1.status(), 200);
    let result1: CreateDatapointsResponse = response1.json().await.unwrap();
    assert_eq!(result1.ids.len(), 1);

    // Try to create datapoint from the same inference again - should succeed and create another datapoint
    let response2 = client
        .post(get_gateway_endpoint(
            "/v1/datasets/test_dataset_dup/from_inferences",
        ))
        .json(&request)
        .send()
        .await
        .unwrap();
    assert_eq!(response2.status(), 200);
    let result2: CreateDatapointsResponse = response2.json().await.unwrap();
    assert_eq!(result2.ids.len(), 1);

    // The datapoint IDs should be different
    assert_ne!(result1.ids[0], result2.ids[0]);
}

#[tokio::test(flavor = "multi_thread")]
async fn test_create_from_inference_missing_ids_error() {
    let client = Client::new();

    // Use a hardcoded inference ID known to have input and output data
    let real_inference_id =
        Uuid::parse_str("0196c682-72e0-7c83-a92b-9d1a3c7630f2").expect("Valid UUID");

    // Generate a fake inference ID that doesn't exist
    let fake_inference_id = Uuid::now_v7();

    // Try to create datapoints from both IDs
    let request = CreateDatapointsFromInferenceRequest {
        params: CreateDatapointsFromInferenceRequestParams::InferenceIds {
            inference_ids: vec![real_inference_id, fake_inference_id],
            output_source: Some(InferenceOutputSource::Inference),
        },
    };

    let response = client
        .post(get_gateway_endpoint(
            "/v1/datasets/test_dataset_missing/from_inferences",
        ))
        .json(&request)
        .send()
        .await
        .unwrap();

    // Should get an error because one of the inferences doesn't exist
    assert_eq!(response.status(), 400);
    let error: serde_json::Value = response.json().await.unwrap();
    let error_message = error["error"].as_str().unwrap();
    assert!(
        error_message.contains("not found"),
        "Expected 'not found' error, got: {error_message}"
    );
}

#[tokio::test]
async fn test_create_from_inference_with_filters() {
    let client = Client::new();

    // Create datapoints using a tag filter that exists in the test data
    let filter = InferenceFilter::Tag(TagFilter {
        key: "tensorzero::evaluation_name".to_string(),
        value: "haiku".to_string(),
        comparison_operator: TagComparisonOperator::Equal,
    });

    let request = CreateDatapointsFromInferenceRequest {
        params: CreateDatapointsFromInferenceRequestParams::InferenceQuery {
            query: Box::new(ListInferencesRequest {
                function_name: Some("write_haiku".to_string()),
                variant_name: Some("better_prompt_haiku_4_5".to_string()),
                filters: Some(filter),
                output_source: InferenceOutputSource::Inference,
                ..Default::default()
            }),
        },
    };

    let response = client
        .post(get_gateway_endpoint(
            "/v1/datasets/test_dataset_filtered/from_inferences",
        ))
        .json(&request)
        .send()
        .await
        .unwrap();

    assert_eq!(response.status(), 200);
    let result: CreateDatapointsResponse = response.json().await.unwrap();

    // Should have created datapoints from filtered inferences
    assert!(!result.ids.is_empty(), "Expected at least one datapoint");
}
