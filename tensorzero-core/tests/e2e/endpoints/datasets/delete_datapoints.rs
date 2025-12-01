/// E2E tests for the delete datapoints API endpoint.
/// Tests the DELETE /v1/datasets/{dataset_name}/datapoints endpoint.
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
    JsonInferenceOutput, Role, StoredInput, StoredInputMessage, StoredInputMessageContent, Text,
};

use crate::common::get_gateway_endpoint;

#[tokio::test]
async fn test_delete_datapoints_single_chat() {
    let http_client = Client::new();
    let clickhouse = get_clickhouse().await;
    let dataset_name = format!("test-delete-dp-single-chat-{}", Uuid::now_v7());

    // Create a chat datapoint
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
                    text: "Hello, world!".to_string(),
                })],
            }],
        },
        output: Some(vec![
            tensorzero_core::inference::types::ContentBlockChatOutput::Text(Text {
                text: "Hi there!".to_string(),
            }),
        ]),
        tool_params: None,
        tags: None,
        auxiliary: String::new(),
        staled_at: None,
        source_inference_id: None,
        is_custom: true,
    });

    clickhouse
        .insert_datapoints(&[datapoint_insert])
        .await
        .unwrap();

    tokio::time::sleep(Duration::from_millis(500)).await;

    // Verify the datapoint exists
    let datapoints = clickhouse
        .get_datapoints(&GetDatapointsParams {
            dataset_name: Some(dataset_name.clone()),
            function_name: None,
            ids: Some(vec![datapoint_id]),
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

    // Delete the datapoint via the endpoint
    let resp = http_client
        .delete(get_gateway_endpoint(&format!(
            "/v1/datasets/{dataset_name}/datapoints"
        )))
        .json(&json!({
            "ids": [datapoint_id.to_string()]
        }))
        .send()
        .await
        .unwrap();

    assert_eq!(resp.status(), StatusCode::OK);
    let delete_response: DeleteDatapointsResponse = resp.json().await.unwrap();
    assert_eq!(delete_response.num_deleted_datapoints, 1);

    tokio::time::sleep(Duration::from_millis(500)).await;

    // Verify the datapoint is now stale (doesn't show up in normal queries)
    let datapoints = clickhouse
        .get_datapoints(&GetDatapointsParams {
            dataset_name: Some(dataset_name.clone()),
            function_name: None,
            ids: Some(vec![datapoint_id]),
            limit: 10,
            offset: 0,
            allow_stale: false,
            filter: None,
            order_by: None,
            search_query_experimental: None,
        })
        .await
        .unwrap();
    assert_eq!(datapoints.len(), 0, "Deleted datapoint should not appear");

    // Verify we can still fetch it with allow_stale=true
    let stale_datapoints = clickhouse
        .get_datapoints(&GetDatapointsParams {
            dataset_name: Some(dataset_name.clone()),
            function_name: None,
            ids: Some(vec![datapoint_id]),
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
    let StoredDatapoint::Chat(dp) = &stale_datapoints[0] else {
        panic!("Expected chat datapoint");
    };
    assert!(dp.staled_at.is_some());
}

#[tokio::test]
async fn test_delete_datapoints_multiple_mixed() {
    let http_client = Client::new();
    let clickhouse = get_clickhouse().await;
    let dataset_name = format!("test-delete-dp-multiple-{}", Uuid::now_v7());

    // Create multiple datapoints: 2 chat and 1 JSON
    let chat_id1 = Uuid::now_v7();
    let chat_id2 = Uuid::now_v7();
    let json_id = Uuid::now_v7();

    let chat_insert1 = DatapointInsert::Chat(ChatInferenceDatapointInsert {
        dataset_name: dataset_name.clone(),
        function_name: "basic_test".to_string(),
        name: None,
        id: chat_id1,
        episode_id: None,
        input: StoredInput {
            system: None,
            messages: vec![StoredInputMessage {
                role: Role::User,
                content: vec![StoredInputMessageContent::Text(Text {
                    text: "Message 1".to_string(),
                })],
            }],
        },
        output: Some(vec![
            tensorzero_core::inference::types::ContentBlockChatOutput::Text(Text {
                text: "Response 1".to_string(),
            }),
        ]),
        tool_params: None,
        tags: None,
        auxiliary: String::new(),
        staled_at: None,
        source_inference_id: None,
        is_custom: true,
    });

    let chat_insert2 = DatapointInsert::Chat(ChatInferenceDatapointInsert {
        dataset_name: dataset_name.clone(),
        function_name: "basic_test".to_string(),
        name: None,
        id: chat_id2,
        episode_id: None,
        input: StoredInput {
            system: None,
            messages: vec![StoredInputMessage {
                role: Role::User,
                content: vec![StoredInputMessageContent::Text(Text {
                    text: "Message 2".to_string(),
                })],
            }],
        },
        output: Some(vec![
            tensorzero_core::inference::types::ContentBlockChatOutput::Text(Text {
                text: "Response 2".to_string(),
            }),
        ]),
        tool_params: None,
        tags: None,
        auxiliary: String::new(),
        staled_at: None,
        source_inference_id: None,
        is_custom: true,
    });

    let json_insert = DatapointInsert::Json(JsonInferenceDatapointInsert {
        dataset_name: dataset_name.clone(),
        function_name: "json_success".to_string(),
        name: None,
        id: json_id,
        episode_id: None,
        input: StoredInput {
            system: None,
            messages: vec![StoredInputMessage {
                role: Role::User,
                content: vec![StoredInputMessageContent::Text(Text {
                    text: "Query".to_string(),
                })],
            }],
        },
        output: Some(JsonInferenceOutput {
            raw: Some(r#"{"answer":"json_answer"}"#.to_string()),
            parsed: Some(json!({"answer": "json_answer"})),
        }),
        output_schema: json!({"type": "object"}),
        tags: None,
        auxiliary: String::new(),
        staled_at: None,
        source_inference_id: None,
        is_custom: true,
    });

    clickhouse
        .insert_datapoints(&[chat_insert1, chat_insert2, json_insert])
        .await
        .unwrap();

    tokio::time::sleep(Duration::from_millis(500)).await;

    // Delete all three datapoints
    let resp = http_client
        .delete(get_gateway_endpoint(&format!(
            "/v1/datasets/{dataset_name}/datapoints"
        )))
        .json(&json!({
            "ids": [chat_id1.to_string(), chat_id2.to_string(), json_id.to_string()]
        }))
        .send()
        .await
        .unwrap();

    assert_eq!(resp.status(), StatusCode::OK);
    let delete_response: DeleteDatapointsResponse = resp.json().await.unwrap();
    assert_eq!(delete_response.num_deleted_datapoints, 3);

    tokio::time::sleep(Duration::from_millis(500)).await;

    // Verify all are now stale
    let datapoints = clickhouse
        .get_datapoints(&GetDatapointsParams {
            dataset_name: Some(dataset_name.clone()),
            function_name: None,
            ids: Some(vec![chat_id1, chat_id2, json_id]),
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
        "All deleted datapoints should not appear"
    );
}

#[tokio::test]
async fn test_delete_datapoints_empty_ids_list() {
    let http_client = Client::new();
    let dataset_name = format!("test-delete-dp-empty-{}", Uuid::now_v7());

    // Try to delete with empty IDs list
    let resp = http_client
        .delete(get_gateway_endpoint(&format!(
            "/v1/datasets/{dataset_name}/datapoints"
        )))
        .json(&json!({
            "ids": []
        }))
        .send()
        .await
        .unwrap();

    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn test_delete_datapoints_non_existent_id() {
    let http_client = Client::new();
    let clickhouse = get_clickhouse().await;
    let dataset_name = format!("test-delete-dp-non-existent-{}", Uuid::now_v7());

    // Create one datapoint
    let existing_id = Uuid::now_v7();
    let datapoint_insert = DatapointInsert::Chat(ChatInferenceDatapointInsert {
        dataset_name: dataset_name.clone(),
        function_name: "basic_test".to_string(),
        name: None,
        id: existing_id,
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
        output: Some(vec![
            tensorzero_core::inference::types::ContentBlockChatOutput::Text(Text {
                text: "Response".to_string(),
            }),
        ]),
        tool_params: None,
        tags: None,
        auxiliary: String::new(),
        staled_at: None,
        source_inference_id: None,
        is_custom: true,
    });

    clickhouse
        .insert_datapoints(&[datapoint_insert])
        .await
        .unwrap();

    tokio::time::sleep(Duration::from_millis(500)).await;

    // Try to delete with a mix of existing and non-existent IDs
    let non_existent_id = Uuid::now_v7();
    let resp = http_client
        .delete(get_gateway_endpoint(&format!(
            "/v1/datasets/{dataset_name}/datapoints"
        )))
        .json(&json!({
            "ids": [existing_id.to_string(), non_existent_id.to_string()]
        }))
        .send()
        .await
        .unwrap();

    // This should return Success - we ignore unknown IDs.
    assert_eq!(resp.status(), StatusCode::OK);
    let delete_response: DeleteDatapointsResponse = resp.json().await.unwrap();
    assert_eq!(delete_response.num_deleted_datapoints, 1);

    // Verify the existing datapoint was deleted.
    let datapoints = clickhouse
        .get_datapoints(&GetDatapointsParams {
            dataset_name: Some(dataset_name.clone()),
            function_name: None,
            ids: Some(vec![existing_id]),
            limit: 10,
            offset: 0,
            allow_stale: false,
            filter: None,
            order_by: None,
            search_query_experimental: None,
        })
        .await
        .unwrap();
    assert_eq!(datapoints.len(), 0, "Existing datapoint should be deleted.");
}

#[tokio::test]
async fn test_delete_datapoints_invalid_dataset_name() {
    let http_client = Client::new();
    let datapoint_id = Uuid::now_v7();

    // Try to delete from a dataset with a reserved name
    let resp = http_client
        .delete(get_gateway_endpoint(
            "/v1/datasets/tensorzero::dataset/datapoints",
        ))
        .json(&json!({
            "ids": [datapoint_id.to_string()]
        }))
        .send()
        .await
        .unwrap();

    // Should return 400 because the dataset name is invalid
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn test_delete_datapoints_from_empty_dataset() {
    let http_client = Client::new();
    let dataset_name = format!("test-delete-dp-empty-dataset-{}", Uuid::now_v7());
    let non_existent_id = Uuid::now_v7();

    // Try to delete from an empty/non-existent dataset
    let resp = http_client
        .delete(get_gateway_endpoint(&format!(
            "/v1/datasets/{dataset_name}/datapoints"
        )))
        .json(&json!({
            "ids": [non_existent_id.to_string()]
        }))
        .send()
        .await
        .unwrap();

    // Even if we delete nothing, this still returns success.
    assert_eq!(resp.status(), StatusCode::OK);
    let delete_response: DeleteDatapointsResponse = resp.json().await.unwrap();
    assert_eq!(delete_response.num_deleted_datapoints, 0);
}

#[tokio::test]
async fn test_delete_datapoints_already_stale() {
    let http_client = Client::new();
    let clickhouse = get_clickhouse().await;
    let dataset_name = format!("test-delete-dp-already-stale-{}", Uuid::now_v7());

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
        output: Some(vec![
            tensorzero_core::inference::types::ContentBlockChatOutput::Text(Text {
                text: "Response".to_string(),
            }),
        ]),
        tool_params: None,
        tags: None,
        auxiliary: String::new(),
        staled_at: None,
        source_inference_id: None,
        is_custom: true,
    });

    clickhouse
        .insert_datapoints(&[datapoint_insert])
        .await
        .unwrap();

    tokio::time::sleep(Duration::from_millis(500)).await;

    // Delete it once
    let resp = http_client
        .delete(get_gateway_endpoint(&format!(
            "/v1/datasets/{dataset_name}/datapoints"
        )))
        .json(&json!({
            "ids": [datapoint_id.to_string()]
        }))
        .send()
        .await
        .unwrap();

    assert_eq!(resp.status(), StatusCode::OK);
    let delete_response: DeleteDatapointsResponse = resp.json().await.unwrap();
    assert_eq!(delete_response.num_deleted_datapoints, 1);

    tokio::time::sleep(Duration::from_millis(500)).await;

    // Try to delete it again
    let resp = http_client
        .delete(get_gateway_endpoint(&format!(
            "/v1/datasets/{dataset_name}/datapoints"
        )))
        .json(&json!({
            "ids": [datapoint_id.to_string()]
        }))
        .send()
        .await
        .unwrap();

    // Even if we delete nothing, this still returns success.
    assert_eq!(
        resp.status(),
        StatusCode::OK,
        "Should return success the second time we try to delete the same datapoint."
    );
    let delete_response: DeleteDatapointsResponse = resp.json().await.unwrap();
    assert_eq!(delete_response.num_deleted_datapoints, 0);
}
