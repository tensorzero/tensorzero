use reqwest::Client;
use uuid::Uuid;

use tensorzero_core::db::clickhouse::query_builder::{
    InferenceFilter, TagComparisonOperator, TagFilter,
};
use tensorzero_core::endpoints::datasets::v1::types::{
    CreateDatapointsFromInferenceOutputSource, CreateDatapointsFromInferenceRequest,
    CreateDatapointsFromInferenceRequestParams, CreateDatapointsFromInferenceResponse,
};
use tensorzero_core::inference::types::Input;

use crate::common::get_gateway_endpoint;

#[tokio::test]
async fn test_create_from_inference_ids_success() {
    let client = Client::new();

    // First, create some inferences
    let input = Input {
        system: None,
        messages: vec![],
    };
    let inference_params = serde_json::json!({
        "function_name": "write_haiku",
        "input": input,
        "stream": false,
    });

    // Create first inference
    let response1 = client
        .post(get_gateway_endpoint("/inference"))
        .json(&inference_params)
        .send()
        .await
        .unwrap();
    assert_eq!(response1.status(), 200);
    let inference1: serde_json::Value = response1.json().await.unwrap();
    let inference_id1 = Uuid::parse_str(inference1["inference_id"].as_str().unwrap()).unwrap();

    // Create second inference
    let response2 = client
        .post(get_gateway_endpoint("/inference"))
        .json(&inference_params)
        .send()
        .await
        .unwrap();
    assert_eq!(response2.status(), 200);
    let inference2: serde_json::Value = response2.json().await.unwrap();
    let inference_id2 = Uuid::parse_str(inference2["inference_id"].as_str().unwrap()).unwrap();

    // Now create datapoints from these inferences
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
    let result: CreateDatapointsFromInferenceResponse = response.json().await.unwrap();

    assert_eq!(result.ids.len(), 2);
    for id in &result.ids {
        assert_ne!(id, &Uuid::nil());
    }
}

#[tokio::test]
async fn test_create_from_inference_query_success() {
    let client = Client::new();

    // Create some inferences with a unique tag for test isolation
    let test_tag = format!("test_query_{}", uuid::Uuid::now_v7());
    let input = Input {
        system: None,
        messages: vec![],
    };
    let inference_params = serde_json::json!({
        "function_name": "write_haiku",
        "input": input,
        "stream": false,
        "tags": {
            "test_id": test_tag
        }
    });

    // Create multiple inferences
    for _ in 0..3 {
        let response = client
            .post(get_gateway_endpoint("/inference"))
            .json(&inference_params)
            .send()
            .await
            .unwrap();
        assert_eq!(response.status(), 200);
    }

    // Wait a bit for data to be written
    tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;

    // Now create datapoints using a query with tag filter
    let filter = InferenceFilter::Tag(TagFilter {
        key: "test_id".to_string(),
        value: test_tag,
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
            "/v1/datasets/test_dataset_query/from_inferences",
        ))
        .json(&request)
        .send()
        .await
        .unwrap();

    assert_eq!(response.status(), 200);
    let result: CreateDatapointsFromInferenceResponse = response.json().await.unwrap();

    // Should have created exactly 3 datapoints (one for each inference we created)
    assert_eq!(result.ids.len(), 3, "Expected exactly 3 datapoints");
}

#[tokio::test]
async fn test_create_from_inference_duplicate_error() {
    let client = Client::new();

    // Create an inference
    let input = Input {
        system: None,
        messages: vec![],
    };
    let inference_params = serde_json::json!({
        "function_name": "write_haiku",
        "input": input,
        "stream": false,
    });

    let response = client
        .post(get_gateway_endpoint("/inference"))
        .json(&inference_params)
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), 200);
    let inference: serde_json::Value = response.json().await.unwrap();
    let inference_id = Uuid::parse_str(inference["inference_id"].as_str().unwrap()).unwrap();

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
    let result1: CreateDatapointsFromInferenceResponse = response1.json().await.unwrap();
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
    let result2: CreateDatapointsFromInferenceResponse = response2.json().await.unwrap();
    assert_eq!(result2.ids.len(), 1);

    // The datapoint IDs should be different
    assert_ne!(result1.ids[0], result2.ids[0]);
}

#[tokio::test]
async fn test_create_from_inference_missing_ids() {
    let client = Client::new();

    // Create one real inference
    let input = Input {
        system: None,
        messages: vec![],
    };
    let inference_params = serde_json::json!({
        "function_name": "write_haiku",
        "input": input,
        "stream": false,
    });

    let response = client
        .post(get_gateway_endpoint("/inference"))
        .json(&inference_params)
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), 200);
    let inference: serde_json::Value = response.json().await.unwrap();
    let real_inference_id = Uuid::parse_str(inference["inference_id"].as_str().unwrap()).unwrap();

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

    // Create some inferences with different tags - use unique values for test isolation
    let test_value1 = format!("filter_test_1_{}", uuid::Uuid::now_v7());
    let test_value2 = format!("filter_test_2_{}", uuid::Uuid::now_v7());

    let input = Input {
        system: None,
        messages: vec![],
    };

    // Create inference with first unique tag
    let inference_params1 = serde_json::json!({
        "function_name": "write_haiku",
        "input": input,
        "stream": false,
        "tags": {
            "test_tag": test_value1
        }
    });

    let response1 = client
        .post(get_gateway_endpoint("/inference"))
        .json(&inference_params1)
        .send()
        .await
        .unwrap();
    assert_eq!(response1.status(), 200);

    // Create inference with second unique tag
    let inference_params2 = serde_json::json!({
        "function_name": "write_haiku",
        "input": input,
        "stream": false,
        "tags": {
            "test_tag": test_value2
        }
    });

    let response2 = client
        .post(get_gateway_endpoint("/inference"))
        .json(&inference_params2)
        .send()
        .await
        .unwrap();
    assert_eq!(response2.status(), 200);

    // Wait a bit for data to be written
    tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;

    // Create datapoints using a tag filter for the first value
    let filter = InferenceFilter::Tag(TagFilter {
        key: "test_tag".to_string(),
        value: test_value1.clone(),
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
    let result: CreateDatapointsFromInferenceResponse = response.json().await.unwrap();

    // Should have created exactly 1 datapoint (only the inference with matching tag)
    assert_eq!(result.ids.len(), 1, "Expected exactly 1 datapoint");
}

#[tokio::test]
async fn test_create_from_inference_dataset_name_with_spaces() {
    let client = Client::new();

    // Create an inference
    let input = Input {
        system: None,
        messages: vec![],
    };
    let inference_params = serde_json::json!({
        "function_name": "write_haiku",
        "input": input,
        "stream": false,
    });

    let response = client
        .post(get_gateway_endpoint("/inference"))
        .json(&inference_params)
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), 200);
    let inference: serde_json::Value = response.json().await.unwrap();
    let inference_id = Uuid::parse_str(inference["inference_id"].as_str().unwrap()).unwrap();

    // Create datapoint with dataset name containing spaces
    let request = CreateDatapointsFromInferenceRequest {
        params: CreateDatapointsFromInferenceRequestParams::InferenceIds {
            inference_ids: vec![inference_id],
        },
        output_source: Some(CreateDatapointsFromInferenceOutputSource::Inference),
    };

    let response = client
        .post(get_gateway_endpoint(
            "/v1/datasets/invalid dataset name/from_inferences",
        ))
        .json(&request)
        .send()
        .await
        .unwrap();

    // Dataset names with spaces are allowed - should succeed
    assert_eq!(response.status(), 200);
    let result: CreateDatapointsFromInferenceResponse = response.json().await.unwrap();
    assert_eq!(result.ids.len(), 1);
}
