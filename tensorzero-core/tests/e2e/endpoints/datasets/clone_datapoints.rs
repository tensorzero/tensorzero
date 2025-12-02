use reqwest::Client;
use serde::Deserialize;
use serde_json::json;
use std::sync::Arc;
use uuid::Uuid;

use tensorzero::{ClientExt, GetDatapointParams, StoredDatapoint};
use tensorzero_core::config::Config;
use tensorzero_core::db::clickhouse::test_helpers::get_clickhouse;
use tensorzero_core::db::clickhouse::ClickHouseConnectionInfo;
use tensorzero_core::db::datasets::DatasetQueries;
use tensorzero_core::db::inferences::{InferenceQueries, ListInferencesParams};
use tensorzero_core::endpoints::datasets::v1::types::{
    CreateDatapointsFromInferenceOutputSource, CreateDatapointsFromInferenceRequest,
    CreateDatapointsFromInferenceRequestParams, CreateDatapointsResponse,
};
use tensorzero_core::stored_inference::StoredSample;

/// Response from the clone datapoints endpoint
#[derive(Debug, Deserialize)]
struct CloneDatapointsResponse {
    datapoint_ids: Vec<Option<Uuid>>,
}

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
async fn test_clone_datapoint_preserves_source_inference_id() {
    let client = Client::new();
    let (clickhouse, config) = get_test_setup().await;

    // Step 1: Get an existing inference from the database
    let params = ListInferencesParams {
        function_name: Some("write_haiku"),
        limit: 1,
        ..Default::default()
    };
    let inferences = clickhouse.list_inferences(config, &params).await.unwrap();
    assert!(!inferences.is_empty(), "Need at least 1 inference for test");
    let inference_id = inferences[0].id();

    // Step 2: Create a datapoint from this inference
    let source_dataset = format!("test_clone_source_{}", Uuid::now_v7());
    let request = CreateDatapointsFromInferenceRequest {
        params: CreateDatapointsFromInferenceRequestParams::InferenceIds {
            inference_ids: vec![inference_id],
        },
        output_source: Some(CreateDatapointsFromInferenceOutputSource::Inference),
    };

    let response = client
        .post(get_gateway_endpoint(&format!(
            "/v1/datasets/{source_dataset}/from_inferences"
        )))
        .json(&request)
        .send()
        .await
        .unwrap();

    assert_eq!(response.status(), 200);
    let create_result: CreateDatapointsResponse = response.json().await.unwrap();
    assert_eq!(create_result.ids.len(), 1);
    let source_datapoint_id = create_result.ids[0];

    // Step 3: Verify the source datapoint has source_inference_id set
    let source_datapoint = clickhouse
        .get_datapoint(&GetDatapointParams {
            dataset_name: source_dataset.clone(),
            datapoint_id: source_datapoint_id,
            allow_stale: None,
        })
        .await
        .unwrap();

    let source_inference_id_from_datapoint = match &source_datapoint {
        StoredDatapoint::Chat(dp) => dp.source_inference_id,
        StoredDatapoint::Json(dp) => dp.source_inference_id,
    };
    assert!(
        source_inference_id_from_datapoint.is_some(),
        "Source datapoint should have source_inference_id set"
    );
    assert_eq!(source_inference_id_from_datapoint.unwrap(), inference_id);

    // Step 4: Clone the datapoint to a target dataset
    let target_dataset = format!("test_clone_target_{}", Uuid::now_v7());
    let clone_request = json!({
        "datapoint_ids": [source_datapoint_id]
    });

    let clone_response = client
        .post(get_gateway_endpoint(&format!(
            "/internal/datasets/{target_dataset}/datapoints/clone"
        )))
        .json(&clone_request)
        .send()
        .await
        .unwrap();

    assert_eq!(clone_response.status(), 200);
    let clone_result: CloneDatapointsResponse = clone_response.json().await.unwrap();
    assert_eq!(clone_result.datapoint_ids.len(), 1);
    let cloned_datapoint_id = clone_result.datapoint_ids[0].expect("Clone should succeed");

    // Step 5: Verify the cloned datapoint preserves source_inference_id
    let cloned_datapoint = clickhouse
        .get_datapoint(&GetDatapointParams {
            dataset_name: target_dataset.clone(),
            datapoint_id: cloned_datapoint_id,
            allow_stale: None,
        })
        .await
        .unwrap();

    // Verify IDs are different
    assert_ne!(source_datapoint_id, cloned_datapoint_id);

    // Verify source_inference_id is preserved
    let cloned_source_inference_id = match &cloned_datapoint {
        StoredDatapoint::Chat(dp) => dp.source_inference_id,
        StoredDatapoint::Json(dp) => dp.source_inference_id,
    };
    assert_eq!(
        cloned_source_inference_id, source_inference_id_from_datapoint,
        "Cloned datapoint should have the same source_inference_id as the original"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn test_clone_chat_datapoint_success() {
    let client = Client::new();
    let (clickhouse, _config) = get_test_setup().await;

    // Step 1: Create a chat datapoint
    let source_dataset = format!("test_clone_chat_source_{}", Uuid::now_v7());
    let create_request = json!({
        "datapoints": [{
            "type": "chat",
            "function_name": "write_haiku",
            "input": {
                "messages": [{
                    "role": "user",
                    "content": [{"type": "template", "name": "user", "arguments": {"topic": "cloning"}}]
                }]
            },
            "output": [{
                "type": "text",
                "text": "Bits copy and flow\nIdentical yet unique\nClone finds its own path"
            }],
            "tags": {
                "test_tag": "clone_test"
            }
        }]
    });

    let response = client
        .post(get_gateway_endpoint(&format!(
            "/v1/datasets/{source_dataset}/datapoints"
        )))
        .json(&create_request)
        .send()
        .await
        .unwrap();

    assert_eq!(response.status(), 200);
    let create_result: CreateDatapointsResponse = response.json().await.unwrap();
    let source_datapoint_id = create_result.ids[0];

    // Step 2: Clone to target dataset
    let target_dataset = format!("test_clone_chat_target_{}", Uuid::now_v7());
    let clone_request = json!({
        "datapoint_ids": [source_datapoint_id]
    });

    let clone_response = client
        .post(get_gateway_endpoint(&format!(
            "/internal/datasets/{target_dataset}/datapoints/clone"
        )))
        .json(&clone_request)
        .send()
        .await
        .unwrap();

    assert_eq!(clone_response.status(), 200);
    let clone_result: CloneDatapointsResponse = clone_response.json().await.unwrap();
    assert_eq!(clone_result.datapoint_ids.len(), 1);
    let cloned_datapoint_id = clone_result.datapoint_ids[0].expect("Clone should succeed");

    // Step 3: Verify the cloned datapoint
    let cloned_datapoint = clickhouse
        .get_datapoint(&GetDatapointParams {
            dataset_name: target_dataset.clone(),
            datapoint_id: cloned_datapoint_id,
            allow_stale: None,
        })
        .await
        .unwrap();

    // Verify ID is different
    assert_ne!(source_datapoint_id, cloned_datapoint_id);

    // Verify it's a chat datapoint with correct function name
    assert_eq!(cloned_datapoint.function_name(), "write_haiku");

    // Verify it's a chat type
    assert!(matches!(cloned_datapoint, StoredDatapoint::Chat(_)));
}

#[tokio::test(flavor = "multi_thread")]
async fn test_clone_to_same_dataset() {
    let client = Client::new();
    let (clickhouse, _config) = get_test_setup().await;

    // Step 1: Create a datapoint
    let dataset_name = format!("test_clone_same_{}", Uuid::now_v7());
    let create_request = json!({
        "datapoints": [{
            "type": "chat",
            "function_name": "write_haiku",
            "input": {
                "messages": [{
                    "role": "user",
                    "content": [{"type": "template", "name": "user", "arguments": {"topic": "duplicates"}}]
                }]
            },
            "output": [{
                "type": "text",
                "text": "Two become one more\nMirrors reflecting mirrors\nEndless copies grow"
            }]
        }]
    });

    let response = client
        .post(get_gateway_endpoint(&format!(
            "/v1/datasets/{dataset_name}/datapoints"
        )))
        .json(&create_request)
        .send()
        .await
        .unwrap();

    assert_eq!(response.status(), 200);
    let create_result: CreateDatapointsResponse = response.json().await.unwrap();
    let source_datapoint_id = create_result.ids[0];

    // Step 2: Clone to the same dataset
    let clone_request = json!({
        "datapoint_ids": [source_datapoint_id]
    });

    let clone_response = client
        .post(get_gateway_endpoint(&format!(
            "/internal/datasets/{dataset_name}/datapoints/clone"
        )))
        .json(&clone_request)
        .send()
        .await
        .unwrap();

    assert_eq!(clone_response.status(), 200);
    let clone_result: CloneDatapointsResponse = clone_response.json().await.unwrap();
    assert_eq!(clone_result.datapoint_ids.len(), 1);
    let cloned_datapoint_id = clone_result.datapoint_ids[0].expect("Clone should succeed");

    // Step 3: Verify both datapoints exist
    let source_datapoint = clickhouse
        .get_datapoint(&GetDatapointParams {
            dataset_name: dataset_name.clone(),
            datapoint_id: source_datapoint_id,
            allow_stale: None,
        })
        .await
        .unwrap();

    let cloned_datapoint = clickhouse
        .get_datapoint(&GetDatapointParams {
            dataset_name: dataset_name.clone(),
            datapoint_id: cloned_datapoint_id,
            allow_stale: None,
        })
        .await
        .unwrap();

    // Both should exist and have different IDs
    assert_eq!(source_datapoint.id(), source_datapoint_id);
    assert_eq!(cloned_datapoint.id(), cloned_datapoint_id);
    assert_ne!(source_datapoint_id, cloned_datapoint_id);
}

#[tokio::test(flavor = "multi_thread")]
async fn test_clone_multiple_datapoints() {
    let client = Client::new();
    let (clickhouse, _config) = get_test_setup().await;

    // Step 1: Create multiple datapoints
    let source_dataset = format!("test_clone_multi_source_{}", Uuid::now_v7());
    let create_request = json!({
        "datapoints": [
            {
                "type": "chat",
                "function_name": "write_haiku",
                "input": {
                    "messages": [{
                        "role": "user",
                        "content": [{"type": "template", "name": "user", "arguments": {"topic": "first"}}]
                    }]
                },
                "output": [{
                    "type": "text",
                    "text": "First haiku here"
                }]
            },
            {
                "type": "chat",
                "function_name": "write_haiku",
                "input": {
                    "messages": [{
                        "role": "user",
                        "content": [{"type": "template", "name": "user", "arguments": {"topic": "second"}}]
                    }]
                },
                "output": [{
                    "type": "text",
                    "text": "Second haiku here"
                }]
            }
        ]
    });

    let response = client
        .post(get_gateway_endpoint(&format!(
            "/v1/datasets/{source_dataset}/datapoints"
        )))
        .json(&create_request)
        .send()
        .await
        .unwrap();

    assert_eq!(response.status(), 200);
    let create_result: CreateDatapointsResponse = response.json().await.unwrap();
    assert_eq!(create_result.ids.len(), 2);

    // Step 2: Clone both datapoints
    let target_dataset = format!("test_clone_multi_target_{}", Uuid::now_v7());
    let clone_request = json!({
        "datapoint_ids": create_result.ids
    });

    let clone_response = client
        .post(get_gateway_endpoint(&format!(
            "/internal/datasets/{target_dataset}/datapoints/clone"
        )))
        .json(&clone_request)
        .send()
        .await
        .unwrap();

    assert_eq!(clone_response.status(), 200);
    let clone_result: CloneDatapointsResponse = clone_response.json().await.unwrap();
    assert_eq!(clone_result.datapoint_ids.len(), 2);

    // Step 3: Verify the cloned datapoints exist
    let cloned_ids: Vec<Uuid> = clone_result
        .datapoint_ids
        .iter()
        .map(|id| id.expect("Clone should succeed"))
        .collect();

    for cloned_id in &cloned_ids {
        let cloned_datapoint = clickhouse
            .get_datapoint(&GetDatapointParams {
                dataset_name: target_dataset.clone(),
                datapoint_id: *cloned_id,
                allow_stale: None,
            })
            .await
            .unwrap();

        assert_eq!(cloned_datapoint.id(), *cloned_id);
    }

    // Verify all IDs are different from original
    for original_id in &create_result.ids {
        assert!(!cloned_ids.contains(original_id));
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn test_clone_nonexistent_datapoint_returns_null() {
    let client = Client::new();

    // Try to clone a nonexistent datapoint
    let nonexistent_id = Uuid::now_v7();
    let target_dataset = format!("test_clone_nonexistent_{}", Uuid::now_v7());
    let clone_request = json!({
        "datapoint_ids": [nonexistent_id]
    });

    let clone_response = client
        .post(get_gateway_endpoint(&format!(
            "/internal/datasets/{target_dataset}/datapoints/clone"
        )))
        .json(&clone_request)
        .send()
        .await
        .unwrap();

    assert_eq!(clone_response.status(), 200);
    let clone_result: CloneDatapointsResponse = clone_response.json().await.unwrap();

    // Should return a list with one null entry
    assert_eq!(clone_result.datapoint_ids.len(), 1);
    assert!(
        clone_result.datapoint_ids[0].is_none(),
        "Should return null for nonexistent datapoint"
    );
}
