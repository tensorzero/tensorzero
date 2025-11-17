use reqwest::Client;
use std::sync::Arc;
use uuid::Uuid;

use tensorzero::ClientExt;
use tensorzero_core::config::Config;
use tensorzero_core::db::clickhouse::query_builder::{
    InferenceFilter, TagComparisonOperator, TagFilter,
};
use tensorzero_core::db::clickhouse::test_helpers::get_clickhouse;
use tensorzero_core::db::clickhouse::ClickHouseConnectionInfo;
use tensorzero_core::db::inferences::{InferenceQueries, ListInferencesParams};
use tensorzero_core::endpoints::datasets::v1::types::{
    CreateDatapointsFromInferenceOutputSource, CreateDatapointsFromInferenceRequest,
    CreateDatapointsFromInferenceRequestParams, CreateDatapointsResponse,
};

use crate::common::get_gateway_endpoint;

lazy_static::lazy_static! {
    static ref TEST_SETUP: tokio::sync::OnceCell<(ClickHouseConnectionInfo, Arc<Config>)> = tokio::sync::OnceCell::new();
}

async fn get_test_setup() -> &'static (ClickHouseConnectionInfo, Arc<Config>) {
    TEST_SETUP
        .get_or_init(|| async {
            let clickhouse: ClickHouseConnectionInfo = get_clickhouse().await;

            let client = tensorzero::test_helpers::make_embedded_gateway().await;
            let config = client.get_config().unwrap();
            (clickhouse, config)
        })
        .await
}

#[tokio::test(flavor = "multi_thread")]
async fn test_create_from_inference_ids_success() {
    let client = Client::new();
    let (clickhouse, config) = get_test_setup().await;

    // Get some existing inferences from the database
    let params = ListInferencesParams {
        function_name: Some("write_haiku"),
        limit: 2,
        ..Default::default()
    };
    let inferences = clickhouse.list_inferences(config, &params).await.unwrap();
    assert!(inferences.len() >= 2, "Need at least 2 inferences for test");

    let inference_id1 = inferences[0].id();
    let inference_id2 = inferences[1].id();

    // Create datapoints from these inferences
    let request = CreateDatapointsFromInferenceRequest {
        params: CreateDatapointsFromInferenceRequestParams::InferenceIds {
            inference_ids: vec![inference_id1, inference_id2],
        },
        output_source: Some(CreateDatapointsFromInferenceOutputSource::Inference),
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

    // Create datapoints using a query (no filters, just function name)
    let request = CreateDatapointsFromInferenceRequest {
        params: CreateDatapointsFromInferenceRequestParams::InferenceQuery {
            function_name: "write_haiku".to_string(),
            variant_name: None,
            filters: None,
        },
        output_source: Some(CreateDatapointsFromInferenceOutputSource::Inference),
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
    let (clickhouse, config) = get_test_setup().await;

    // Get an existing inference from the database
    let params = ListInferencesParams {
        function_name: Some("write_haiku"),
        limit: 1,
        ..Default::default()
    };
    let inferences = clickhouse.list_inferences(config, &params).await.unwrap();
    assert!(!inferences.is_empty(), "Need at least 1 inference for test");

    let inference_id = inferences[0].id();

    // Create datapoint from this inference
    let request = CreateDatapointsFromInferenceRequest {
        params: CreateDatapointsFromInferenceRequestParams::InferenceIds {
            inference_ids: vec![inference_id],
        },
        output_source: Some(CreateDatapointsFromInferenceOutputSource::Inference),
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
    let (clickhouse, config) = get_test_setup().await;

    // Get one real inference
    let params = ListInferencesParams {
        function_name: Some("write_haiku"),
        limit: 1,
        ..Default::default()
    };
    let inferences = clickhouse.list_inferences(config, &params).await.unwrap();
    assert!(!inferences.is_empty(), "Need at least 1 inference for test");

    let real_inference_id = inferences[0].id();

    // Generate a fake inference ID that doesn't exist
    let fake_inference_id = Uuid::now_v7();

    // Try to create datapoints from both IDs
    let request = CreateDatapointsFromInferenceRequest {
        params: CreateDatapointsFromInferenceRequestParams::InferenceIds {
            inference_ids: vec![real_inference_id, fake_inference_id],
        },
        output_source: Some(CreateDatapointsFromInferenceOutputSource::Inference),
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
            function_name: "write_haiku".to_string(),
            variant_name: None,
            filters: Some(filter),
        },
        output_source: Some(CreateDatapointsFromInferenceOutputSource::Inference),
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
