#![cfg(feature = "e2e_tests")]
#![expect(clippy::unwrap_used)]
use std::collections::HashMap;
use tensorzero::{
    Client, ClientExt, CreateChatDatapointRequest, CreateDatapointRequest,
    CreateDatapointsFromInferenceRequestParams, Datapoint, ListDatapointsRequest,
    UpdateChatDatapointRequest, UpdateDatapointMetadataRequest, UpdateDatapointRequest,
};
use tensorzero_core::db::inferences::InferenceOutputSource;
use tensorzero_core::endpoints::datasets::v1::types::DatapointMetadataUpdate;
use tensorzero_core::endpoints::stored_inferences::v1::types::ListInferencesRequest;
use tensorzero_core::inference::types::{
    ContentBlockChatOutput, Input, InputMessage, InputMessageContent, Text,
};
use tensorzero_core::tool::DynamicToolParams;
use uuid::Uuid;

/// Helper function to create a unique dataset name for testing
fn test_dataset_name(prefix: &str) -> String {
    format!("{}_{}", prefix, Uuid::now_v7())
}

/// Helper to create a simple chat input with system message
fn create_chat_input_with_system(
    message: &str,
    system: Option<HashMap<String, serde_json::Value>>,
) -> Input {
    use tensorzero_core::inference::types::{Arguments, System};

    Input {
        system: system.map(|s| {
            let map: serde_json::Map<String, serde_json::Value> = s.into_iter().collect();
            System::Template(Arguments(map))
        }),
        messages: vec![InputMessage {
            role: tensorzero_core::inference::types::Role::User,
            content: vec![InputMessageContent::Text(Text {
                text: message.to_string(),
            })],
        }],
    }
}

/// Helper to create chat output
fn create_chat_output(text: &str) -> Vec<ContentBlockChatOutput> {
    vec![ContentBlockChatOutput::Text(Text {
        text: text.to_string(),
    })]
}

// ============================================================================
// Create Datapoints Tests
// ============================================================================

/// Test creating multiple datapoints in a single request
async fn test_create_datapoints(client: Client) {
    let dataset_name = test_dataset_name("test_create_multiple");

    // Insert test datapoints
    let mut system = HashMap::new();
    system.insert(
        "assistant_name".to_string(),
        serde_json::Value::String("TestBot".to_string()),
    );

    let datapoints = vec![
        CreateDatapointRequest::Chat(CreateChatDatapointRequest {
            function_name: "basic_test".to_string(),
            episode_id: None,
            input: create_chat_input_with_system("First message", Some(system.clone())),
            output: Some(vec![ContentBlockChatOutput::Text(Text {
                text: "First response".to_string(),
            })]),
            dynamic_tool_params: DynamicToolParams::default(),
            tags: None,
            name: Some("first_datapoint".to_string()),
        }),
        CreateDatapointRequest::Chat(CreateChatDatapointRequest {
            function_name: "basic_test".to_string(),
            episode_id: None,
            input: create_chat_input_with_system("Second message", Some(system.clone())),
            output: Some(vec![ContentBlockChatOutput::Text(Text {
                text: "Second response".to_string(),
            })]),
            dynamic_tool_params: DynamicToolParams::default(),
            tags: None,
            name: Some("second_datapoint".to_string()),
        }),
    ];

    let response = client
        .create_datapoints(dataset_name.clone(), datapoints)
        .await
        .unwrap();

    assert_eq!(response.ids.len(), 2);

    // Verify all datapoints were created
    let get_response = client
        .get_datapoints(Some(dataset_name.clone()), response.ids.clone())
        .await
        .unwrap();

    assert_eq!(get_response.datapoints.len(), 2);

    // Clean up
    let delete_response = client
        .delete_datapoints(dataset_name, response.ids.clone())
        .await
        .unwrap();

    assert_eq!(delete_response.num_deleted_datapoints, 2);
}

tensorzero::make_gateway_test_functions!(test_create_datapoints);

// ============================================================================
// Get Datapoints Tests
// ============================================================================

/// Test retrieving datapoints by IDs using v1 endpoint
async fn test_get_datapoints_by_ids(client: Client) {
    let dataset_name = test_dataset_name("test_get_v1");

    // Insert test datapoints
    let mut system = HashMap::new();
    system.insert(
        "assistant_name".to_string(),
        serde_json::Value::String("TestBot".to_string()),
    );

    let datapoints = vec![
        CreateDatapointRequest::Chat(CreateChatDatapointRequest {
            function_name: "basic_test".to_string(),
            episode_id: None,
            input: create_chat_input_with_system("First message", Some(system.clone())),
            output: Some(create_chat_output("First response")),
            dynamic_tool_params: DynamicToolParams::default(),
            tags: None,
            name: None,
        }),
        CreateDatapointRequest::Chat(CreateChatDatapointRequest {
            function_name: "basic_test".to_string(),
            episode_id: None,
            input: create_chat_input_with_system("Second message", Some(system.clone())),
            output: Some(create_chat_output("Second response")),
            dynamic_tool_params: DynamicToolParams::default(),
            tags: None,
            name: None,
        }),
    ];

    let response = client
        .create_datapoints(dataset_name.clone(), datapoints)
        .await
        .unwrap();

    assert_eq!(response.ids.len(), 2);
    let datapoint_ids = response.ids;

    // Get all datapoints by IDs using v1 endpoint
    let response = client
        .get_datapoints(Some(dataset_name.clone()), datapoint_ids.clone())
        .await
        .unwrap();

    assert_eq!(response.datapoints.len(), 2);

    // Verify we got the correct datapoints
    let retrieved_ids: Vec<Uuid> = response.datapoints.iter().map(Datapoint::id).collect();
    assert_eq!(retrieved_ids.len(), 2);
    for id in &datapoint_ids {
        assert!(retrieved_ids.contains(id));
    }

    // Clean up
    client
        .delete_datapoints(dataset_name.clone(), datapoint_ids)
        .await
        .unwrap();
}

tensorzero::make_gateway_test_functions!(test_get_datapoints_by_ids);

// ============================================================================
// List Datapoints Tests
// ============================================================================

/// Test listing datapoints with pagination
async fn test_list_datapoints_with_pagination(client: Client) {
    let dataset_name = test_dataset_name("test_list_v1");

    // Insert multiple datapoints
    let mut system1 = HashMap::new();
    system1.insert(
        "assistant_name".to_string(),
        serde_json::Value::String("Bot1".to_string()),
    );
    let mut system2 = HashMap::new();
    system2.insert(
        "assistant_name".to_string(),
        serde_json::Value::String("Bot2".to_string()),
    );

    let datapoints = vec![
        CreateDatapointRequest::Chat(CreateChatDatapointRequest {
            function_name: "basic_test".to_string(),
            episode_id: None,
            input: create_chat_input_with_system("msg1", Some(system1)),
            output: None,
            dynamic_tool_params: DynamicToolParams::default(),
            tags: None,
            name: None,
        }),
        CreateDatapointRequest::Chat(CreateChatDatapointRequest {
            function_name: "basic_test".to_string(),
            episode_id: None,
            input: create_chat_input_with_system("msg2", Some(system2)),
            output: None,
            dynamic_tool_params: DynamicToolParams::default(),
            tags: None,
            name: None,
        }),
    ];

    let response = client
        .create_datapoints(dataset_name.clone(), datapoints)
        .await
        .unwrap();

    assert_eq!(response.ids.len(), 2);
    let datapoint_ids = response.ids;

    // List all datapoints
    let request = ListDatapointsRequest {
        limit: Some(10),
        offset: Some(0),
        ..Default::default()
    };
    let response = client
        .list_datapoints(dataset_name.clone(), request)
        .await
        .unwrap();

    assert_eq!(response.datapoints.len(), 2);

    // List with limit
    let request = ListDatapointsRequest {
        limit: Some(1),
        offset: Some(0),
        ..Default::default()
    };
    let response = client
        .list_datapoints(dataset_name.clone(), request)
        .await
        .unwrap();

    assert_eq!(response.datapoints.len(), 1);

    // List with offset
    let request = ListDatapointsRequest {
        limit: Some(10),
        offset: Some(1),
        ..Default::default()
    };
    let response = client
        .list_datapoints(dataset_name.clone(), request)
        .await
        .unwrap();

    assert_eq!(response.datapoints.len(), 1);

    // Clean up
    client
        .delete_datapoints(dataset_name.clone(), datapoint_ids)
        .await
        .unwrap();
}

tensorzero::make_gateway_test_functions!(test_list_datapoints_with_pagination);

// ============================================================================
// Update Datapoints Tests
// ============================================================================

/// Test updating datapoints (creates new IDs)
async fn test_update_datapoints(client: Client) {
    let dataset_name = test_dataset_name("test_update_v1");

    // Insert initial datapoints
    let mut system = HashMap::new();
    system.insert(
        "assistant_name".to_string(),
        serde_json::Value::String("OriginalBot".to_string()),
    );

    let datapoints = vec![CreateDatapointRequest::Chat(CreateChatDatapointRequest {
        function_name: "basic_test".to_string(),
        episode_id: None,
        input: create_chat_input_with_system("Original message", Some(system)),
        output: Some(create_chat_output("Original response")),
        dynamic_tool_params: DynamicToolParams::default(),
        tags: None,
        name: None,
    })];

    let response = client
        .create_datapoints(dataset_name.clone(), datapoints)
        .await
        .unwrap();

    assert_eq!(response.ids.len(), 1);
    let datapoint_ids = response.ids;

    // Update the chat datapoint
    let updated_output = vec![ContentBlockChatOutput::Text(Text {
        text: "Updated response".to_string(),
    })];

    let chat_update = UpdateDatapointRequest::Chat(UpdateChatDatapointRequest {
        id: datapoint_ids[0],
        input: None,
        output: Some(Some(updated_output)),
        #[expect(deprecated)]
        deprecated_do_not_use_tool_params: Default::default(),
        tags: None,
        #[expect(deprecated)]
        deprecated_do_not_use_metadata: Default::default(),
        metadata: Default::default(),
        tool_params: Default::default(),
    });

    let response = client
        .update_datapoints(dataset_name.clone(), vec![chat_update])
        .await
        .unwrap();

    // Update creates new IDs
    assert_eq!(response.ids.len(), 1);
    assert_ne!(response.ids[0], datapoint_ids[0]);

    // Clean up - delete the new datapoint
    client
        .delete_datapoints(dataset_name.clone(), response.ids)
        .await
        .unwrap();
}

tensorzero::make_gateway_test_functions!(test_update_datapoints);

/// Test update_datapoints_metadata: Update metadata without creating new IDs
async fn test_update_datapoints_metadata(client: Client) {
    let dataset_name = test_dataset_name("test_update_meta");

    // Insert datapoint with initial name
    let mut system = HashMap::new();
    system.insert(
        "assistant_name".to_string(),
        serde_json::Value::String("MetaBot".to_string()),
    );

    let datapoints = vec![CreateDatapointRequest::Chat(CreateChatDatapointRequest {
        function_name: "basic_test".to_string(),
        episode_id: None,
        input: create_chat_input_with_system("original", Some(system)),
        output: None,
        dynamic_tool_params: DynamicToolParams::default(),
        tags: None,
        name: Some("original_name".to_string()),
    })];

    let response = client
        .create_datapoints(dataset_name.clone(), datapoints)
        .await
        .unwrap();

    let original_id = response.ids[0];

    // Update metadata
    let metadata_updates = vec![UpdateDatapointMetadataRequest {
        id: original_id,
        metadata: DatapointMetadataUpdate {
            name: Some(Some("updated_name".to_string())),
        },
    }];

    let response = client
        .update_datapoints_metadata(dataset_name.clone(), metadata_updates)
        .await
        .unwrap();

    assert_eq!(response.ids.len(), 1);
    // The ID should remain the same (not a new ID like update_datapoints would create)
    assert_eq!(response.ids[0], original_id);

    // Clean up
    client
        .delete_datapoints(dataset_name, vec![original_id])
        .await
        .unwrap();
}

tensorzero::make_gateway_test_functions!(test_update_datapoints_metadata);

// ============================================================================
// Delete Datapoints Tests
// ============================================================================

/// Test deleting multiple datapoints
async fn test_delete_multiple_datapoints(client: Client) {
    let dataset_name = test_dataset_name("test_delete_multi");

    // Insert multiple datapoints
    let mut system = HashMap::new();
    system.insert(
        "assistant_name".to_string(),
        serde_json::Value::String("DeleteBot".to_string()),
    );

    let datapoints: Vec<_> = (0..5)
        .map(|i| {
            CreateDatapointRequest::Chat(CreateChatDatapointRequest {
                function_name: "basic_test".to_string(),
                episode_id: None,
                input: create_chat_input_with_system(&format!("message {i}"), Some(system.clone())),
                output: None,
                dynamic_tool_params: DynamicToolParams::default(),
                tags: None,
                name: None,
            })
        })
        .collect();

    let response = client
        .create_datapoints(dataset_name.clone(), datapoints)
        .await
        .unwrap();

    assert_eq!(response.ids.len(), 5);
    let datapoint_ids = response.ids;

    // Delete first 3 datapoints using v1 bulk delete
    let ids_to_delete = datapoint_ids[0..3].to_vec();

    let response = client
        .delete_datapoints(dataset_name.clone(), ids_to_delete.clone())
        .await
        .unwrap();

    assert_eq!(response.num_deleted_datapoints, 3);

    // Verify remaining datapoints
    let request = ListDatapointsRequest {
        limit: Some(100),
        offset: Some(0),
        ..Default::default()
    };
    let remaining = client
        .list_datapoints(dataset_name.clone(), request)
        .await
        .unwrap();

    assert_eq!(remaining.datapoints.len(), 2);

    let remaining_ids: Vec<Uuid> = remaining.datapoints.iter().map(Datapoint::id).collect();
    for id in &datapoint_ids[3..] {
        assert!(remaining_ids.contains(id));
    }

    // Clean up remaining
    client
        .delete_datapoints(dataset_name.clone(), datapoint_ids[3..].to_vec())
        .await
        .unwrap();
}

tensorzero::make_gateway_test_functions!(test_delete_multiple_datapoints);

/// Test delete_dataset: Delete an entire dataset
async fn test_delete_entire_dataset(client: Client) {
    let dataset_name = test_dataset_name("test_delete_dataset");

    // Insert multiple datapoints
    let mut system = HashMap::new();
    system.insert(
        "assistant_name".to_string(),
        serde_json::Value::String("DatasetBot".to_string()),
    );

    let datapoints: Vec<_> = (0..3)
        .map(|i| {
            CreateDatapointRequest::Chat(CreateChatDatapointRequest {
                function_name: "basic_test".to_string(),
                episode_id: None,
                input: create_chat_input_with_system(&format!("message {i}"), Some(system.clone())),
                output: None,
                dynamic_tool_params: DynamicToolParams::default(),
                tags: None,
                name: None,
            })
        })
        .collect();

    let response = client
        .create_datapoints(dataset_name.clone(), datapoints)
        .await
        .unwrap();

    assert_eq!(response.ids.len(), 3);

    // Delete the entire dataset
    let response = client.delete_dataset(dataset_name.clone()).await.unwrap();

    assert_eq!(response.num_deleted_datapoints, 3);

    // Verify no datapoints remain
    let request = ListDatapointsRequest {
        limit: Some(100),
        offset: Some(0),
        ..Default::default()
    };
    let remaining = client
        .list_datapoints(dataset_name.clone(), request)
        .await
        .unwrap();

    assert_eq!(remaining.datapoints.len(), 0);
}

tensorzero::make_gateway_test_functions!(test_delete_entire_dataset);

// ============================================================================
// Create from Inferences Tests
// ============================================================================

/// Test creating datapoints from inferences
async fn test_create_datapoints_from_inferences(client: Client) {
    let dataset_name = test_dataset_name("test_from_inferences");

    // Create datapoints from an inference query
    let params = CreateDatapointsFromInferenceRequestParams::InferenceQuery {
        query: Box::new(ListInferencesRequest {
            function_name: Some("write_haiku".to_string()),
            variant_name: Some("better_prompt_haiku_4_5".to_string()),
            output_source: InferenceOutputSource::Inference,
            ..Default::default()
        }),
    };

    let response = client
        .create_datapoints_from_inferences(dataset_name.clone(), params)
        .await
        .unwrap();

    assert!(!response.ids.is_empty(), "Expected at least one datapoint");

    // Verify the datapoint was created

    let datapoints = client
        .get_datapoints(Some(dataset_name.clone()), response.ids.clone())
        .await
        .unwrap();

    assert_eq!(
        datapoints.datapoints.len(),
        response.ids.len(),
        "Each inference should create a datapoint"
    );

    // Clean up
    client.delete_dataset(dataset_name).await.unwrap();
}

tensorzero::make_gateway_test_functions!(test_create_datapoints_from_inferences);
