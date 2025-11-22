/// E2E tests for the delete dataset API endpoint.
/// Tests the DELETE /v1/datasets/{dataset_name} endpoint.
use reqwest::{Client, StatusCode};
use serde_json::json;
use std::time::Duration;
use uuid::Uuid;

use tensorzero_core::db::clickhouse::test_helpers::get_clickhouse;
use tensorzero_core::db::datasets::{
    ChatInferenceDatapointInsert, DatapointInsert, DatasetQueries, GetDatapointsParams,
    JsonInferenceDatapointInsert,
};
use tensorzero_core::endpoints::datasets::v1::types::DeleteDatapointsResponse;
use tensorzero_core::endpoints::datasets::StoredDatapoint;
use tensorzero_core::inference::types::{
    ContentBlockChatOutput, JsonInferenceOutput, Role, StoredInput, StoredInputMessage,
    StoredInputMessageContent, Text,
};

use crate::common::get_gateway_endpoint;

#[tokio::test]
async fn test_delete_dataset_with_single_datapoint() {
    let http_client = Client::new();
    let clickhouse = get_clickhouse().await;
    let dataset_name = format!("test-delete-dataset-single-{}", Uuid::now_v7());

    // Create a single datapoint
    let datapoint_id = Uuid::now_v7();
    let datapoint_insert = DatapointInsert::Chat(ChatInferenceDatapointInsert {
        dataset_name: dataset_name.clone(),
        function_name: "basic_test".to_string(),
        name: Some("Test Datapoint".to_string()),
        id: datapoint_id,
        episode_id: None,
        input: StoredInput {
            system: None,
            messages: vec![StoredInputMessage {
                role: Role::User,
                content: vec![StoredInputMessageContent::Text(Text {
                    text: "Hello!".to_string(),
                })],
            }],
        },
        output: Some(vec![ContentBlockChatOutput::Text(Text {
            text: "Hi!".to_string(),
        })]),
        tool_params: None,
        tags: None,
        auxiliary: String::new(),
        staled_at: None,
        source_inference_id: None,
        is_custom: true,
    });

    clickhouse
        .insert_datapoints(std::slice::from_ref(&datapoint_insert))
        .await
        .unwrap();

    tokio::time::sleep(Duration::from_millis(500)).await;

    // Verify the datapoint exists
    let datapoints = clickhouse
        .get_datapoints(&GetDatapointsParams {
            dataset_name: Some(dataset_name.clone()),
            function_name: None,
            ids: None,
            limit: 10,
            offset: 0,
            allow_stale: false,
            filter: None,
            order_by: None,
            search_query_experimental: None,
        })
        .await
        .unwrap();
    assert_eq!(datapoints.len(), 1);

    // Delete the entire dataset via the endpoint
    let resp = http_client
        .delete(get_gateway_endpoint(&format!(
            "/v1/datasets/{dataset_name}"
        )))
        .send()
        .await
        .unwrap();

    assert_eq!(resp.status(), StatusCode::OK);
    let delete_response: DeleteDatapointsResponse = resp.json().await.unwrap();
    assert_eq!(delete_response.num_deleted_datapoints, 1);

    tokio::time::sleep(Duration::from_millis(500)).await;

    // Verify all datapoints are now stale
    let datapoints = clickhouse
        .get_datapoints(&GetDatapointsParams {
            dataset_name: Some(dataset_name.clone()),
            function_name: None,
            ids: None,
            limit: 10,
            offset: 0,
            allow_stale: false,
            filter: None,
            order_by: None,
            search_query_experimental: None,
        })
        .await
        .unwrap();
    assert_eq!(
        datapoints.len(),
        0,
        "Deleted dataset should have no datapoints"
    );

    // Verify we can still fetch stale datapoints
    let stale_datapoints = clickhouse
        .get_datapoints(&GetDatapointsParams {
            dataset_name: Some(dataset_name.clone()),
            function_name: None,
            ids: None,
            limit: 10,
            offset: 0,
            allow_stale: true,
            filter: None,
            order_by: None,
            search_query_experimental: None,
        })
        .await
        .unwrap();
    assert_eq!(stale_datapoints.len(), 1);
    // Verify staled_at is set
    match &stale_datapoints[0] {
        StoredDatapoint::Chat(dp) => {
            assert!(dp.staled_at.is_some());
        }
        StoredDatapoint::Json(_) => {
            panic!("Expected chat datapoint")
        }
    }
}

#[tokio::test]
async fn test_delete_dataset_with_multiple_mixed_datapoints() {
    let http_client = Client::new();
    let clickhouse = get_clickhouse().await;
    let dataset_name = format!("test-delete-dataset-multiple-{}", Uuid::now_v7());

    // Create multiple datapoints: 3 chat and 2 JSON
    let mut inserts = vec![];

    for i in 0..3 {
        inserts.push(DatapointInsert::Chat(ChatInferenceDatapointInsert {
            dataset_name: dataset_name.clone(),
            function_name: "basic_test".to_string(),
            name: Some(format!("Chat Datapoint {i}")),
            id: Uuid::now_v7(),
            episode_id: None,
            input: StoredInput {
                system: None,
                messages: vec![StoredInputMessage {
                    role: Role::User,
                    content: vec![StoredInputMessageContent::Text(Text {
                        text: format!("Message {i}"),
                    })],
                }],
            },
            output: Some(vec![ContentBlockChatOutput::Text(Text {
                text: format!("Response {i}"),
            })]),
            tool_params: None,
            tags: None,
            auxiliary: String::new(),
            staled_at: None,
            source_inference_id: None,
            is_custom: true,
        }));
    }

    for i in 0..2 {
        inserts.push(DatapointInsert::Json(JsonInferenceDatapointInsert {
            dataset_name: dataset_name.clone(),
            function_name: "json_success".to_string(),
            name: Some(format!("JSON Datapoint {i}")),
            id: Uuid::now_v7(),
            episode_id: None,
            input: StoredInput {
                system: None,
                messages: vec![StoredInputMessage {
                    role: Role::User,
                    content: vec![StoredInputMessageContent::Text(Text {
                        text: format!("Query {i}"),
                    })],
                }],
            },
            output: Some(JsonInferenceOutput {
                raw: Some(format!(r#"{{"result":"result_{i}"}}"#)),
                parsed: Some(json!({"result": format!("result_{i}")})),
            }),
            output_schema: json!({"type": "object"}),
            tags: None,
            auxiliary: String::new(),
            staled_at: None,
            source_inference_id: None,
            is_custom: true,
        }));
    }

    clickhouse.insert_datapoints(&inserts).await.unwrap();

    tokio::time::sleep(Duration::from_millis(500)).await;

    // Verify we have 5 datapoints
    let datapoints = clickhouse
        .get_datapoints(&GetDatapointsParams {
            dataset_name: Some(dataset_name.clone()),
            function_name: None,
            ids: None,
            limit: 100,
            offset: 0,
            allow_stale: false,
            filter: None,
            order_by: None,
            search_query_experimental: None,
        })
        .await
        .unwrap();
    assert_eq!(datapoints.len(), 5);

    // Delete the entire dataset
    let resp = http_client
        .delete(get_gateway_endpoint(&format!(
            "/v1/datasets/{dataset_name}"
        )))
        .send()
        .await
        .unwrap();

    assert_eq!(resp.status(), StatusCode::OK);
    let delete_response: DeleteDatapointsResponse = resp.json().await.unwrap();
    assert_eq!(delete_response.num_deleted_datapoints, 5);

    tokio::time::sleep(Duration::from_millis(500)).await;

    // Verify all datapoints are now stale
    let datapoints = clickhouse
        .get_datapoints(&GetDatapointsParams {
            dataset_name: Some(dataset_name.clone()),
            function_name: None,
            ids: None,
            limit: 100,
            offset: 0,
            allow_stale: false,
            filter: None,
            order_by: None,
            search_query_experimental: None,
        })
        .await
        .unwrap();
    assert_eq!(datapoints.len(), 0, "All datapoints should be stale");

    // Verify we can still fetch all 5 as stale
    let stale_datapoints = clickhouse
        .get_datapoints(&GetDatapointsParams {
            dataset_name: Some(dataset_name.clone()),
            function_name: None,
            ids: None,
            limit: 100,
            offset: 0,
            allow_stale: true,
            filter: None,
            order_by: None,
            search_query_experimental: None,
        })
        .await
        .unwrap();
    assert_eq!(stale_datapoints.len(), 5);
    for dp in &stale_datapoints {
        match dp {
            StoredDatapoint::Chat(chat_dp) => {
                assert!(chat_dp.staled_at.is_some());
            }
            StoredDatapoint::Json(json_dp) => {
                assert!(json_dp.staled_at.is_some());
            }
        }
    }
}

#[tokio::test]
async fn test_delete_empty_dataset() {
    let http_client = Client::new();
    let dataset_name = format!("test-delete-dataset-empty-{}", Uuid::now_v7());

    // Try to delete a dataset that doesn't exist or is empty
    let resp = http_client
        .delete(get_gateway_endpoint(&format!(
            "/v1/datasets/{dataset_name}"
        )))
        .send()
        .await
        .unwrap();

    // Should succeed even if the dataset is empty
    assert_eq!(resp.status(), StatusCode::OK);
    let delete_response: DeleteDatapointsResponse = resp.json().await.unwrap();
    assert_eq!(delete_response.num_deleted_datapoints, 0);
}

#[tokio::test]
async fn test_delete_dataset_invalid_name() {
    let http_client = Client::new();

    // Try to delete a dataset with invalid characters
    let resp = http_client
        .delete(get_gateway_endpoint("/v1/datasets/tensorzero::dataset"))
        .send()
        .await
        .unwrap();

    // Should return 400 because the dataset name is invalid
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn test_delete_dataset_twice() {
    let http_client = Client::new();
    let clickhouse = get_clickhouse().await;
    let dataset_name = format!("test-delete-dataset-twice-{}", Uuid::now_v7());

    // Create a datapoint
    let datapoint_id = Uuid::now_v7();
    let datapoint_insert = DatapointInsert::Chat(ChatInferenceDatapointInsert {
        dataset_name: dataset_name.clone(),
        function_name: "basic_test".to_string(),
        name: None,
        id: datapoint_id,
        episode_id: None,
        input: StoredInput {
            system: None,
            messages: vec![StoredInputMessage {
                role: Role::User,
                content: vec![StoredInputMessageContent::Text(Text {
                    text: "Test".to_string(),
                })],
            }],
        },
        output: Some(vec![ContentBlockChatOutput::Text(Text {
            text: "Response".to_string(),
        })]),
        tool_params: None,
        tags: None,
        auxiliary: String::new(),
        staled_at: None,
        source_inference_id: None,
        is_custom: true,
    });

    clickhouse
        .insert_datapoints(std::slice::from_ref(&datapoint_insert))
        .await
        .unwrap();

    tokio::time::sleep(Duration::from_millis(500)).await;

    // Delete the dataset
    let resp = http_client
        .delete(get_gateway_endpoint(&format!(
            "/v1/datasets/{dataset_name}"
        )))
        .send()
        .await
        .unwrap();

    assert_eq!(resp.status(), StatusCode::OK);
    let delete_response: DeleteDatapointsResponse = resp.json().await.unwrap();
    assert_eq!(delete_response.num_deleted_datapoints, 1);

    tokio::time::sleep(Duration::from_millis(500)).await;

    // Delete again (should succeed but do nothing since all datapoints are already stale)
    let resp = http_client
        .delete(get_gateway_endpoint(&format!(
            "/v1/datasets/{dataset_name}"
        )))
        .send()
        .await
        .unwrap();

    assert_eq!(resp.status(), StatusCode::OK);
    let delete_response: DeleteDatapointsResponse = resp.json().await.unwrap();
    assert_eq!(delete_response.num_deleted_datapoints, 0);
}

#[tokio::test]
async fn test_delete_dataset_with_different_function_names() {
    let http_client = Client::new();
    let clickhouse = get_clickhouse().await;
    let dataset_name = format!("test-delete-dataset-functions-{}", Uuid::now_v7());

    // Create datapoints with different function names in the same dataset
    let function1_insert = DatapointInsert::Chat(ChatInferenceDatapointInsert {
        dataset_name: dataset_name.clone(),
        function_name: "function_one".to_string(),
        name: None,
        id: Uuid::now_v7(),
        episode_id: None,
        input: StoredInput {
            system: None,
            messages: vec![StoredInputMessage {
                role: Role::User,
                content: vec![StoredInputMessageContent::Text(Text {
                    text: "Test 1".to_string(),
                })],
            }],
        },
        output: Some(vec![ContentBlockChatOutput::Text(Text {
            text: "Response 1".to_string(),
        })]),
        tool_params: None,
        tags: None,
        auxiliary: String::new(),
        staled_at: None,
        source_inference_id: None,
        is_custom: true,
    });

    let function2_insert = DatapointInsert::Chat(ChatInferenceDatapointInsert {
        dataset_name: dataset_name.clone(),
        function_name: "function_two".to_string(),
        name: None,
        id: Uuid::now_v7(),
        episode_id: None,
        input: StoredInput {
            system: None,
            messages: vec![StoredInputMessage {
                role: Role::User,
                content: vec![StoredInputMessageContent::Text(Text {
                    text: "Test 2".to_string(),
                })],
            }],
        },
        output: Some(vec![ContentBlockChatOutput::Text(Text {
            text: "Response 2".to_string(),
        })]),
        tool_params: None,
        tags: None,
        auxiliary: String::new(),
        staled_at: None,
        source_inference_id: None,
        is_custom: true,
    });

    clickhouse
        .insert_datapoints(&[function1_insert, function2_insert])
        .await
        .unwrap();

    tokio::time::sleep(Duration::from_millis(500)).await;

    // Delete the entire dataset
    let resp = http_client
        .delete(get_gateway_endpoint(&format!(
            "/v1/datasets/{dataset_name}"
        )))
        .send()
        .await
        .unwrap();

    assert_eq!(resp.status(), StatusCode::OK);
    let delete_response: DeleteDatapointsResponse = resp.json().await.unwrap();
    assert_eq!(delete_response.num_deleted_datapoints, 2);

    tokio::time::sleep(Duration::from_millis(500)).await;

    // Verify both function's datapoints are stale
    let datapoints_func1 = clickhouse
        .get_datapoints(&GetDatapointsParams {
            dataset_name: Some(dataset_name.clone()),
            function_name: Some("function_one".to_string()),
            ids: None,
            limit: 10,
            offset: 0,
            allow_stale: false,
            filter: None,
            order_by: None,
            search_query_experimental: None,
        })
        .await
        .unwrap();
    assert_eq!(datapoints_func1.len(), 0);

    let datapoints_func2 = clickhouse
        .get_datapoints(&GetDatapointsParams {
            dataset_name: Some(dataset_name.clone()),
            function_name: Some("function_two".to_string()),
            ids: None,
            limit: 10,
            offset: 0,
            allow_stale: false,
            filter: None,
            order_by: None,
            search_query_experimental: None,
        })
        .await
        .unwrap();
    assert_eq!(datapoints_func2.len(), 0);
}
