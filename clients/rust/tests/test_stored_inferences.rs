#![cfg(feature = "e2e_tests")]
#![expect(clippy::unwrap_used)]
#![expect(clippy::unreachable)]
use serde_json::json;
use tensorzero::{
    Client, ClientExt, ClientInferenceParams, InferenceOutput, InferenceOutputSource, Input,
    InputMessage, InputMessageContent, ListInferencesRequest, Role, System,
};
use tensorzero_core::inference::types::{Arguments, Text};
use uuid::Uuid;

// Helper function to create test inferences using the client
// This ensures embedded gateway tests write and read through the same ClickHouse connection
async fn create_test_inference(client: &Client) -> Uuid {
    let response = client
        .inference(ClientInferenceParams {
            function_name: Some("basic_test".to_string()),
            input: Input {
                system: Some(System::Template(Arguments(
                    json!({"assistant_name": "Assistant"})
                        .as_object()
                        .unwrap()
                        .clone(),
                ))),
                messages: vec![InputMessage {
                    role: Role::User,
                    content: vec![InputMessageContent::Text(Text {
                        text: "Hello".to_string(),
                    })],
                }],
            },
            stream: Some(false),
            ..Default::default()
        })
        .await
        .unwrap();

    match response {
        InferenceOutput::NonStreaming(r) => r.inference_id(),
        InferenceOutput::Streaming(_) => unreachable!("Expected non-streaming response"),
    }
}

// ============================================================================
// Get Inferences Tests
// ============================================================================

/// Test retrieving inferences by IDs
async fn test_get_inferences_by_ids(client: Client) {
    // Create some test inferences first
    let _id1 = create_test_inference(&client).await;
    let _id2 = create_test_inference(&client).await;
    let _id3 = create_test_inference(&client).await;

    // Wait a bit for the inferences to be written to the database
    tokio::time::sleep(tokio::time::Duration::from_millis(500)).await;

    // First list some existing inferences
    let list_request = ListInferencesRequest {
        function_name: Some("basic_test".to_string()),
        limit: Some(3),
        ..Default::default()
    };
    let list_response = client.list_inferences(list_request).await.unwrap();

    assert!(
        !list_response.inferences.is_empty(),
        "Expected at least some inferences to exist"
    );

    // Get the IDs of some existing inferences
    let inference_ids: Vec<Uuid> = list_response
        .inferences
        .iter()
        .map(|inf| match inf {
            tensorzero::StoredInference::Chat(chat_inf) => chat_inf.inference_id,
            tensorzero::StoredInference::Json(json_inf) => json_inf.inference_id,
        })
        .collect();

    // Get inferences by IDs
    let response = client
        .get_inferences(
            inference_ids.clone(),
            None,
            InferenceOutputSource::Inference,
        )
        .await
        .unwrap();

    assert_eq!(response.inferences.len(), inference_ids.len());

    // Verify we got the correct inferences
    let retrieved_ids: Vec<Uuid> = response
        .inferences
        .iter()
        .map(|inf| match inf {
            tensorzero::StoredInference::Chat(chat_inf) => chat_inf.inference_id,
            tensorzero::StoredInference::Json(json_inf) => json_inf.inference_id,
        })
        .collect();

    for id in &inference_ids {
        assert!(retrieved_ids.contains(id));
    }
}

tensorzero::make_gateway_test_functions!(test_get_inferences_by_ids);

/// Test getting inferences with empty ID list
async fn test_get_inferences_empty_ids(client: Client) {
    let response = client
        .get_inferences(vec![], None, InferenceOutputSource::Inference)
        .await
        .unwrap();

    assert_eq!(response.inferences.len(), 0);
}

tensorzero::make_gateway_test_functions!(test_get_inferences_empty_ids);

/// Test getting inferences with unknown IDs
async fn test_get_inferences_unknown_ids(client: Client) {
    let unknown_id = Uuid::now_v7();

    let response = client
        .get_inferences(vec![unknown_id], None, InferenceOutputSource::Inference)
        .await
        .unwrap();

    assert_eq!(response.inferences.len(), 0);
}

tensorzero::make_gateway_test_functions!(test_get_inferences_unknown_ids);

/// Test getting inferences with function_name parameter for better performance
async fn test_get_inferences_with_function_name(client: Client) {
    // Create some test inferences first
    let _id1 = create_test_inference(&client).await;
    let _id2 = create_test_inference(&client).await;
    let _id3 = create_test_inference(&client).await;

    // Wait a bit for the inferences to be written to the database
    tokio::time::sleep(tokio::time::Duration::from_millis(500)).await;

    // First, list some inferences to get their IDs
    let list_request = ListInferencesRequest {
        function_name: Some("basic_test".to_string()),
        limit: Some(3),
        ..Default::default()
    };
    let list_response = client.list_inferences(list_request).await.unwrap();
    assert!(
        !list_response.inferences.is_empty(),
        "Should have at least one inference"
    );

    // Get the inference IDs
    let inference_ids: Vec<_> = list_response
        .inferences
        .iter()
        .map(|inf| match inf {
            tensorzero::StoredInference::Chat(chat_inf) => chat_inf.inference_id,
            tensorzero::StoredInference::Json(json_inf) => json_inf.inference_id,
        })
        .collect();

    // Test get_inferences WITH function_name (should use optimized query)
    let response_with_function = client
        .get_inferences(
            inference_ids.clone(),
            Some("basic_test".to_string()),
            InferenceOutputSource::Inference,
        )
        .await
        .unwrap();

    assert_eq!(response_with_function.inferences.len(), inference_ids.len());

    // Verify we got the correct inferences
    for inference in &response_with_function.inferences {
        let inference_id = match inference {
            tensorzero::StoredInference::Chat(chat_inf) => chat_inf.inference_id,
            tensorzero::StoredInference::Json(json_inf) => json_inf.inference_id,
        };
        assert!(inference_ids.contains(&inference_id));
    }

    // Test get_inferences WITHOUT function_name (should still work, but slower)
    let response_without_function = client
        .get_inferences(
            inference_ids.clone(),
            None,
            InferenceOutputSource::Inference,
        )
        .await
        .unwrap();

    assert_eq!(
        response_without_function.inferences.len(),
        inference_ids.len()
    );

    // Both should return the same results
    assert_eq!(
        response_with_function.inferences.len(),
        response_without_function.inferences.len()
    );
}

tensorzero::make_gateway_test_functions!(test_get_inferences_with_function_name);

// ============================================================================
// List Inferences Tests
// ============================================================================

/// Test listing inferences with pagination
async fn test_list_inferences_with_pagination(client: Client) {
    // Create some test inferences first
    for _ in 0..5 {
        let _ = create_test_inference(&client).await;
    }

    // Wait a bit for the inferences to be written to the database
    tokio::time::sleep(tokio::time::Duration::from_millis(500)).await;

    // List all inferences with default pagination
    let request = ListInferencesRequest {
        function_name: Some("basic_test".to_string()),
        limit: Some(100),
        ..Default::default()
    };
    let response = client.list_inferences(request).await.unwrap();

    assert!(
        !response.inferences.is_empty(),
        "Expected at least some inferences"
    );
    let total_count = response.inferences.len();

    // List with limit
    let request = ListInferencesRequest {
        function_name: Some("basic_test".to_string()),
        limit: Some(2),
        ..Default::default()
    };
    let response = client.list_inferences(request).await.unwrap();

    assert!(
        response.inferences.len() <= 2,
        "Limit should cap the results at 2"
    );

    // List with offset (only if we have enough inferences)
    if total_count > 2 {
        let request = ListInferencesRequest {
            function_name: Some("basic_test".to_string()),
            limit: Some(100),
            offset: Some(2),
            ..Default::default()
        };
        let response = client.list_inferences(request).await.unwrap();

        assert!(
            !response.inferences.is_empty(),
            "Expected at least some inferences with offset"
        );
    }
}

tensorzero::make_gateway_test_functions!(test_list_inferences_with_pagination);

/// Test listing inferences by function name
async fn test_list_inferences_by_function(client: Client) {
    // Create some test inferences first
    for _ in 0..3 {
        let _ = create_test_inference(&client).await;
    }

    // Wait a bit for the inferences to be written to the database
    tokio::time::sleep(tokio::time::Duration::from_millis(500)).await;

    // List inferences for basic_test with filtering
    let request = ListInferencesRequest {
        function_name: Some("basic_test".to_string()),
        limit: Some(100),
        ..Default::default()
    };
    let response = client.list_inferences(request).await.unwrap();

    // Verify all returned inferences are from basic_test
    assert!(
        !response.inferences.is_empty(),
        "Expected at least some inferences"
    );
    for inference in &response.inferences {
        let function_name = match inference {
            tensorzero::StoredInference::Chat(chat_inf) => &chat_inf.function_name,
            tensorzero::StoredInference::Json(json_inf) => &json_inf.function_name,
        };
        assert_eq!(function_name, "basic_test");
    }
}

tensorzero::make_gateway_test_functions!(test_list_inferences_by_function);

/// Test listing inferences by variant name
async fn test_list_inferences_by_variant(client: Client) {
    // Create some test inferences first
    for _ in 0..3 {
        let _ = create_test_inference(&client).await;
    }

    // Wait a bit for the inferences to be written to the database
    tokio::time::sleep(tokio::time::Duration::from_millis(500)).await;

    // First get existing inferences to find a variant name
    let list_request = ListInferencesRequest {
        function_name: Some("basic_test".to_string()),
        limit: Some(1),
        ..Default::default()
    };
    let list_response = client.list_inferences(list_request).await.unwrap();

    assert!(
        !list_response.inferences.is_empty(),
        "Expected at least some inferences to exist"
    );

    // Get the variant name from the first inference
    let variant_name = match &list_response.inferences[0] {
        tensorzero::StoredInference::Chat(chat_inf) => &chat_inf.variant_name,
        tensorzero::StoredInference::Json(json_inf) => &json_inf.variant_name,
    };

    // List inferences for that specific variant
    let request = ListInferencesRequest {
        function_name: Some("basic_test".to_string()),
        variant_name: Some(variant_name.clone()),
        limit: Some(100),
        ..Default::default()
    };
    let response = client.list_inferences(request).await.unwrap();

    // Verify all returned inferences are from the specified variant
    assert!(
        !response.inferences.is_empty(),
        "Expected at least some inferences with this variant"
    );
    for inference in &response.inferences {
        let inf_variant_name = match inference {
            tensorzero::StoredInference::Chat(chat_inf) => &chat_inf.variant_name,
            tensorzero::StoredInference::Json(json_inf) => &json_inf.variant_name,
        };
        assert_eq!(inf_variant_name, variant_name);
    }
}

tensorzero::make_gateway_test_functions!(test_list_inferences_by_variant);

// Test listing inferences by episode ID
async fn test_list_inferences_by_episode(client: Client) {
    // Create some test inferences first
    for _ in 0..3 {
        let _ = create_test_inference(&client).await;
    }

    // Wait a bit for the inferences to be written to the database
    tokio::time::sleep(tokio::time::Duration::from_millis(500)).await;

    // First get an existing inference to extract an episode_id
    let list_request = ListInferencesRequest {
        function_name: Some("basic_test".to_string()),
        limit: Some(100),
        ..Default::default()
    };
    let list_response = client.list_inferences(list_request).await.unwrap();

    assert!(
        !list_response.inferences.is_empty(),
        "Expected at least some inferences to exist"
    );

    // Get an episode_id from one of the existing inferences
    let episode_id = match &list_response.inferences[0] {
        tensorzero::StoredInference::Chat(chat_inf) => chat_inf.episode_id,
        tensorzero::StoredInference::Json(json_inf) => json_inf.episode_id,
    };

    // List inferences by episode ID
    let request = ListInferencesRequest {
        episode_id: Some(episode_id),
        limit: Some(100),
        ..Default::default()
    };
    let response = client.list_inferences(request).await.unwrap();

    assert!(
        !response.inferences.is_empty(),
        "Expected at least one inference with this episode_id"
    );

    // Verify all inferences have the correct episode ID
    for inference in &response.inferences {
        let inf_episode_id = match inference {
            tensorzero::StoredInference::Chat(chat_inf) => chat_inf.episode_id,
            tensorzero::StoredInference::Json(json_inf) => json_inf.episode_id,
        };
        assert_eq!(inf_episode_id, episode_id);
    }
}

tensorzero::make_gateway_test_functions!(test_list_inferences_by_episode);

/// Test listing inferences with ordering
async fn test_list_inferences_with_ordering(client: Client) {
    // Create some test inferences first
    for _ in 0..2 {
        let _ = create_test_inference(&client).await;
        // Add a small delay to ensure different timestamps
        tokio::time::sleep(tokio::time::Duration::from_millis(50)).await;
    }

    // Wait a bit for the inferences to be written to the database
    tokio::time::sleep(tokio::time::Duration::from_millis(500)).await;

    // List inferences ordered by timestamp descending
    let request = ListInferencesRequest {
        function_name: Some("basic_test".to_string()),
        limit: Some(10),
        order_by: Some(vec![tensorzero::OrderBy {
            term: tensorzero::OrderByTerm::Timestamp,
            direction: tensorzero::OrderDirection::Desc,
        }]),
        ..Default::default()
    };
    let response = client.list_inferences(request).await.unwrap();

    assert!(!response.inferences.is_empty());

    // Verify timestamps are in descending order
    let timestamps: Vec<_> = response
        .inferences
        .iter()
        .map(|inf| match inf {
            tensorzero::StoredInference::Chat(chat_inf) => chat_inf.timestamp,
            tensorzero::StoredInference::Json(json_inf) => json_inf.timestamp,
        })
        .collect();

    for i in 0..timestamps.len().saturating_sub(1) {
        assert!(
            timestamps[i] >= timestamps[i + 1],
            "Timestamps should be in descending order"
        );
    }
}

tensorzero::make_gateway_test_functions!(test_list_inferences_with_ordering);

/// Test listing inferences with tags filter
async fn test_list_inferences_with_tag_filter(client: Client) {
    // Create an inference with a specific tag using the client
    let mut tags = std::collections::HashMap::new();
    tags.insert("test_key".to_string(), "test_value".to_string());

    let _response = client
        .inference(ClientInferenceParams {
            function_name: Some("basic_test".to_string()),
            input: Input {
                system: Some(System::Template(Arguments(
                    json!({"assistant_name": "Assistant"})
                        .as_object()
                        .unwrap()
                        .clone(),
                ))),
                messages: vec![InputMessage {
                    role: Role::User,
                    content: vec![InputMessageContent::Text(Text {
                        text: "Hello with tags".to_string(),
                    })],
                }],
            },
            stream: Some(false),
            tags,
            ..Default::default()
        })
        .await
        .unwrap();

    // Wait for async writes to be visible
    tokio::time::sleep(tokio::time::Duration::from_millis(500)).await;

    // First get existing inferences to find one with tags
    let list_request = ListInferencesRequest {
        function_name: Some("basic_test".to_string()),
        limit: Some(100),
        ..Default::default()
    };
    let list_response = client.list_inferences(list_request).await.unwrap();

    assert!(
        !list_response.inferences.is_empty(),
        "Expected at least some inferences to exist"
    );

    // Find an inference with tags
    let inference_with_tags = list_response.inferences.iter().find(|inf| match inf {
        tensorzero::StoredInference::Chat(chat_inf) => !chat_inf.tags.is_empty(),
        tensorzero::StoredInference::Json(json_inf) => !json_inf.tags.is_empty(),
    });

    // If we found an inference with tags, test filtering by one of its tags
    if let Some(inference) = inference_with_tags {
        let (key, value) = match inference {
            tensorzero::StoredInference::Chat(chat_inf) => chat_inf.tags.iter().next().unwrap(),
            tensorzero::StoredInference::Json(json_inf) => json_inf.tags.iter().next().unwrap(),
        };

        // List inferences filtered by tag
        let request = ListInferencesRequest {
            function_name: Some("basic_test".to_string()),
            limit: Some(100),
            filters: Some(tensorzero::InferenceFilter::Tag(tensorzero::TagFilter {
                key: key.clone(),
                value: value.clone(),
                comparison_operator: tensorzero::TagComparisonOperator::Equal,
            })),
            ..Default::default()
        };
        let response = client.list_inferences(request).await.unwrap();

        // Verify all returned inferences have the tag
        assert!(
            !response.inferences.is_empty(),
            "Expected at least some inferences with this tag"
        );
        for inference in &response.inferences {
            let inf_tags = match inference {
                tensorzero::StoredInference::Chat(chat_inf) => &chat_inf.tags,
                tensorzero::StoredInference::Json(json_inf) => &json_inf.tags,
            };
            assert_eq!(
                inf_tags.get(key),
                Some(value),
                "All returned inferences should have the {key}={value} tag",
            );
        }
    }
}

tensorzero::make_gateway_test_functions!(test_list_inferences_with_tag_filter);
