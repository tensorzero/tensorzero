#![cfg(feature = "e2e_tests")]
#![expect(clippy::unwrap_used, clippy::missing_panics_doc)]

use serde_json::json;
use tensorzero::{
    make_gateway_test_functions, Client, CreateDatapointsFromInferenceRequest,
    DeleteDatapointsRequest, GetDatapointsRequest, InsertDatapointParams, ListDatapointsRequest,
    UpdateDatapointsMetadataRequest, UpdateDatapointsRequest,
};
use tensorzero_core::endpoints::datasets::v1::types::{
    DatapointMetadataUpdate, UpdateChatDatapointRequest, UpdateDatapointRequest,
    UpdateJsonDatapointRequest,
};
use tensorzero_core::inference::types::{
    ContentBlockChatOutput, Input, InputMessage, InputMessageContent, JsonInferenceOutput, Role,
    System, Text,
};
use uuid::Uuid;

/// Helper function to create a unique dataset name for testing
fn test_dataset_name(prefix: &str) -> String {
    format!("{}_{}", prefix, Uuid::now_v7())
}

/// Test get_datapoints: Retrieve multiple datapoints by IDs
async fn test_get_datapoints_by_ids(client: Client) {
    let dataset_name = test_dataset_name("test_get_v1");

    // Insert test datapoints
    let datapoints = vec![
        serde_json::from_value(json!({
            "type": "chat",
            "function_name": "basic_test",
            "input": {
                "system": {"assistant_name": "TestBot"},
                "messages": [{
                    "role": "user",
                    "content": [{"type": "text", "value": "First message"}]
                }]
            },
            "output": [{"type": "text", "value": "First response"}]
        }))
        .unwrap(),
        serde_json::from_value(json!({
            "type": "chat",
            "function_name": "basic_test",
            "input": {
                "system": {"assistant_name": "TestBot"},
                "messages": [{
                    "role": "user",
                    "content": [{"type": "text", "value": "Second message"}]
                }]
            },
            "output": [{"type": "text", "value": "Second response"}]
        }))
        .unwrap(),
        serde_json::from_value(json!({
            "type": "json",
            "function_name": "json_success",
            "input": {
                "system": {"assistant_name": "JsonBot"},
                "messages": [{
                    "role": "user",
                    "content": [{"type": "text", "value": "What is the capital of Canada?"}]
                }]
            },
            "output": {"answer": "Ottawa"}
        }))
        .unwrap(),
    ];

    let datapoint_ids = client
        .create_datapoints(dataset_name.clone(), InsertDatapointParams { datapoints })
        .await
        .unwrap();

    assert_eq!(datapoint_ids.len(), 3);

    // Get all datapoints by IDs using v1 endpoint
    let request = GetDatapointsRequest {
        ids: datapoint_ids.clone(),
    };
    let response = client.get_datapoints(request).await.unwrap();

    assert_eq!(response.datapoints.len(), 3);

    // Verify we got the correct datapoints
    let retrieved_ids: Vec<Uuid> = response.datapoints.iter().map(|dp| dp.id()).collect();
    assert_eq!(retrieved_ids.len(), 3);
    for id in &datapoint_ids {
        assert!(retrieved_ids.contains(id));
    }

    // Clean up
    for dp_id in datapoint_ids {
        client
            .delete_datapoint(dataset_name.clone(), dp_id)
            .await
            .unwrap();
    }
}

make_gateway_test_functions!(test_get_datapoints_by_ids);

/// Test list_datapoints: List datapoints with pagination
async fn test_list_datapoints_with_pagination(client: Client) {
    let dataset_name = test_dataset_name("test_list_v1");

    // Insert multiple datapoints
    let datapoints = vec![
        serde_json::from_value(json!({
            "type": "chat",
            "function_name": "basic_test",
            "input": {
                "system": {"assistant_name": "Bot1"},
                "messages": [{
                    "role": "user",
                    "content": [{"type": "text", "value": "msg1"}]
                }]
            }
        }))
        .unwrap(),
        serde_json::from_value(json!({
            "type": "chat",
            "function_name": "basic_test",
            "input": {
                "system": {"assistant_name": "Bot2"},
                "messages": [{
                    "role": "user",
                    "content": [{"type": "text", "value": "msg2"}]
                }]
            }
        }))
        .unwrap(),
        serde_json::from_value(json!({
            "type": "json",
            "function_name": "json_success",
            "input": {
                "system": {"assistant_name": "JsonBot"},
                "messages": [{
                    "role": "user",
                    "content": [{"type": "text", "value": "msg3"}]
                }]
            },
            "output": {"answer": "test"}
        }))
        .unwrap(),
    ];

    let datapoint_ids = client
        .create_datapoints(dataset_name.clone(), InsertDatapointParams { datapoints })
        .await
        .unwrap();

    assert_eq!(datapoint_ids.len(), 3);

    // List all datapoints
    let request = ListDatapointsRequest {
        filters: None,
        page_size: Some(10),
        page_token: None,
    };
    let response = client
        .list_datapoints(dataset_name.clone(), request)
        .await
        .unwrap();

    assert_eq!(response.datapoints.len(), 3);

    // List with page_size limit
    let request = ListDatapointsRequest {
        filters: None,
        page_size: Some(2),
        page_token: None,
    };
    let response = client
        .list_datapoints(dataset_name.clone(), request)
        .await
        .unwrap();

    assert_eq!(response.datapoints.len(), 2);

    // List with page token (next page)
    if let Some(next_token) = response.next_page_token {
        let request = ListDatapointsRequest {
            filters: None,
            page_size: Some(2),
            page_token: Some(next_token),
        };
        let response = client
            .list_datapoints(dataset_name.clone(), request)
            .await
            .unwrap();

        assert_eq!(response.datapoints.len(), 1);
    }

    // Clean up
    for dp_id in datapoint_ids {
        client
            .delete_datapoint(dataset_name.clone(), dp_id)
            .await
            .unwrap();
    }
}

make_gateway_test_functions!(test_list_datapoints_with_pagination);

/// Test update_datapoints: Update chat and JSON datapoints
async fn test_update_datapoints(client: Client) {
    let dataset_name = test_dataset_name("test_update_v1");

    // Insert initial datapoints
    let datapoints = vec![
        serde_json::from_value(json!({
            "type": "chat",
            "function_name": "basic_test",
            "input": {
                "system": {"assistant_name": "OriginalBot"},
                "messages": [{
                    "role": "user",
                    "content": [{"type": "text", "value": "Original message"}]
                }]
            },
            "output": [{"type": "text", "value": "Original response"}]
        }))
        .unwrap(),
        serde_json::from_value(json!({
            "type": "json",
            "function_name": "json_success",
            "input": {
                "system": {"assistant_name": "OriginalJson"},
                "messages": [{
                    "role": "user",
                    "content": [{"type": "text", "value": "Original query"}]
                }]
            },
            "output": {"answer": "original"}
        }))
        .unwrap(),
    ];

    let datapoint_ids = client
        .create_datapoints(dataset_name.clone(), InsertDatapointParams { datapoints })
        .await
        .unwrap();

    assert_eq!(datapoint_ids.len(), 2);

    // Update the chat datapoint
    let updated_output = vec![ContentBlockChatOutput::Text(Text {
        text: "Updated response".to_string(),
    })];

    let chat_update = UpdateDatapointRequest::Chat(UpdateChatDatapointRequest {
        id: datapoint_ids[0],
        input: None,
        output: Some(updated_output),
        tool_params: None,
        tags: None,
        metadata: None,
    });

    // Update the JSON datapoint
    let json_update = UpdateDatapointRequest::Json(UpdateJsonDatapointRequest {
        id: datapoint_ids[1],
        input: None,
        output: Some(Some(json!({"answer": "updated"}))),
        output_schema: None,
        tags: None,
        metadata: None,
    });

    let request = UpdateDatapointsRequest {
        datapoints: vec![chat_update, json_update],
    };

    let response = client
        .update_datapoints(dataset_name.clone(), request)
        .await
        .unwrap();

    // Update creates new IDs
    assert_eq!(response.ids.len(), 2);
    assert_ne!(response.ids[0], datapoint_ids[0]);
    assert_ne!(response.ids[1], datapoint_ids[1]);

    // Clean up - delete old and new datapoints
    for dp_id in datapoint_ids {
        let _ = client.delete_datapoint(dataset_name.clone(), dp_id).await;
    }
    for dp_id in response.ids {
        let _ = client.delete_datapoint(dataset_name.clone(), dp_id).await;
    }
}

make_gateway_test_functions!(test_update_datapoints);

/// Test update_datapoints_metadata: Update metadata without creating new IDs
async fn test_update_datapoints_metadata(client: Client) {
    let dataset_name = test_dataset_name("test_update_meta");

    // Insert datapoint with initial name
    let datapoints = vec![serde_json::from_value(json!({
        "type": "chat",
        "function_name": "basic_test",
        "input": {
            "system": {"assistant_name": "MetaBot"},
            "messages": [{
                "role": "user",
                "content": [{"type": "text", "value": "original"}]
            }]
        },
        "name": "original_name"
    }))
    .unwrap()];

    let datapoint_ids = client
        .create_datapoints(dataset_name.clone(), InsertDatapointParams { datapoints })
        .await
        .unwrap();

    let original_id = datapoint_ids[0];

    // Update metadata
    let request = UpdateDatapointsMetadataRequest {
        datapoints: vec![serde_json::from_value(json!({
            "id": original_id,
            "name": "updated_name"
        }))
        .unwrap()],
    };

    let response = client
        .update_datapoints_metadata(dataset_name.clone(), request)
        .await
        .unwrap();

    assert_eq!(response.ids.len(), 1);
    // The ID should remain the same (not a new ID like update_datapoints would create)
    assert_eq!(response.ids[0], original_id);

    // Clean up
    client
        .delete_datapoint(dataset_name, original_id)
        .await
        .unwrap();
}

make_gateway_test_functions!(test_update_datapoints_metadata);

/// Test delete_datapoints: Delete multiple datapoints at once
async fn test_delete_multiple_datapoints(client: Client) {
    let dataset_name = test_dataset_name("test_delete_multi");

    // Insert multiple datapoints
    let datapoints: Vec<_> = (0..5)
        .map(|i| {
            serde_json::from_value(json!({
                "type": "chat",
                "function_name": "basic_test",
                "input": {
                    "system": {"assistant_name": "DeleteBot"},
                    "messages": [{
                        "role": "user",
                        "content": [{"type": "text", "value": format!("message {}", i)}]
                    }]
                }
            }))
            .unwrap()
        })
        .collect();

    let datapoint_ids = client
        .create_datapoints(dataset_name.clone(), InsertDatapointParams { datapoints })
        .await
        .unwrap();

    assert_eq!(datapoint_ids.len(), 5);

    // Delete first 3 datapoints using v1 bulk delete
    let ids_to_delete = datapoint_ids[0..3].to_vec();
    let request = DeleteDatapointsRequest {
        ids: ids_to_delete.clone(),
    };

    let response = client
        .delete_datapoints(dataset_name.clone(), request)
        .await
        .unwrap();

    assert_eq!(response.num_deleted_datapoints, 3);

    // Verify remaining datapoints
    let request = ListDatapointsRequest {
        filters: None,
        page_size: Some(100),
        page_token: None,
    };
    let remaining = client
        .list_datapoints(dataset_name.clone(), request)
        .await
        .unwrap();

    assert_eq!(remaining.datapoints.len(), 2);

    let remaining_ids: Vec<Uuid> = remaining.datapoints.iter().map(|dp| dp.id()).collect();
    for id in &datapoint_ids[3..] {
        assert!(remaining_ids.contains(id));
    }

    // Clean up remaining
    for dp_id in &datapoint_ids[3..] {
        client
            .delete_datapoint(dataset_name.clone(), *dp_id)
            .await
            .unwrap();
    }
}

make_gateway_test_functions!(test_delete_multiple_datapoints);

/// Test delete_dataset: Delete an entire dataset
async fn test_delete_entire_dataset(client: Client) {
    let dataset_name = test_dataset_name("test_delete_dataset");

    // Insert multiple datapoints
    let datapoints: Vec<_> = (0..3)
        .map(|i| {
            serde_json::from_value(json!({
                "type": "chat",
                "function_name": "basic_test",
                "input": {
                    "system": {"assistant_name": "DatasetBot"},
                    "messages": [{
                        "role": "user",
                        "content": [{"type": "text", "value": format!("message {}", i)}]
                    }]
                }
            }))
            .unwrap()
        })
        .collect();

    let datapoint_ids = client
        .create_datapoints(dataset_name.clone(), InsertDatapointParams { datapoints })
        .await
        .unwrap();

    assert_eq!(datapoint_ids.len(), 3);

    // Delete the entire dataset
    let response = client.delete_dataset(dataset_name.clone()).await.unwrap();

    assert_eq!(response.num_deleted_datapoints, 3);

    // Verify no datapoints remain
    let request = ListDatapointsRequest {
        filters: None,
        page_size: Some(100),
        page_token: None,
    };
    let remaining = client
        .list_datapoints(dataset_name.clone(), request)
        .await
        .unwrap();

    assert_eq!(remaining.datapoints.len(), 0);
}

make_gateway_test_functions!(test_delete_entire_dataset);

/// Test create_from_inferences: Create datapoints from inference results
async fn test_create_from_inferences(client: Client) {
    let dataset_name = test_dataset_name("test_from_inferences");

    // First, create some inferences by running an inference
    let inference_params = tensorzero::ClientInferenceParams {
        function_name: Some("basic_test".to_string()),
        episode_id: None,
        input: tensorzero::ClientInput {
            system: Some(tensorzero::System::Template(
                tensorzero_core::inference::types::Arguments(serde_json::Map::from_iter([(
                    "assistant_name".to_string(),
                    json!("TestBot"),
                )])),
            )),
            messages: vec![tensorzero::ClientInputMessage {
                role: Role::User,
                content: vec![tensorzero::ClientInputMessageContent::Text(
                    "Test message".to_string(),
                )],
            }],
        },
        ..Default::default()
    };

    let inference_result = client.inference(inference_params).await.unwrap();
    let inference_id = inference_result.inference_id;

    // Create datapoints from the inference
    let request = CreateDatapointsFromInferenceRequest {
        inference_ids: vec![inference_id],
        output_source: None, // Use inference output
    };

    let response = client
        .create_from_inferences(dataset_name.clone(), request)
        .await
        .unwrap();

    assert_eq!(response.ids.len(), 1);

    // Verify the datapoint was created
    let get_request = GetDatapointsRequest {
        ids: response.ids.clone(),
    };
    let datapoints = client.get_datapoints(get_request).await.unwrap();

    assert_eq!(datapoints.datapoints.len(), 1);

    // Clean up
    client
        .delete_datapoint(dataset_name, response.ids[0])
        .await
        .unwrap();
}

make_gateway_test_functions!(test_create_from_inferences);
