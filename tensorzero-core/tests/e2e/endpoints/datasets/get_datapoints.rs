/// Comprehensive tests for the get_datapoints and list_datapoints API endpoints.
/// Tests both the dataset-scoped POST /v1/datasets/{dataset_name}/get_datapoints and the deprecated
/// POST /v1/datasets/get_datapoints endpoints alongside POST /v1/datasets/{dataset_name}/list_datapoints.
use reqwest::{Client, StatusCode};
use serde_json::{json, Value};
use std::collections::HashMap;
use std::time::Duration;
use uuid::Uuid;

use tensorzero_core::db::clickhouse::test_helpers::get_clickhouse;
use tensorzero_core::db::datasets::{
    ChatInferenceDatapointInsert, DatapointInsert, DatasetQueries, JsonInferenceDatapointInsert,
};
use tensorzero_core::inference::types::{
    Arguments, JsonInferenceOutput, Role, StoredInput, StoredInputMessage,
    StoredInputMessageContent, System, Text,
};

use crate::common::get_gateway_endpoint;

/// Tests for the /v1/datasets/{dataset_name}/get_datapoints endpoint.
mod get_datapoints_tests {
    use super::*;

    #[tokio::test]
    async fn test_get_datapoints_single_chat_datapoint_without_dataset_name() {
        let http_client = Client::new();
        let clickhouse = get_clickhouse().await;
        let dataset_name = format!("test-get-dp-single-chat-{}", Uuid::now_v7());

        // Create a chat datapoint
        let datapoint_id = Uuid::now_v7();
        let mut tags = HashMap::new();
        tags.insert("env".to_string(), "test".to_string());

        let datapoint_insert = DatapointInsert::Chat(ChatInferenceDatapointInsert {
            dataset_name: dataset_name.clone(),
            function_name: "basic_test".to_string(),
            name: Some("Test Datapoint".to_string()),
            id: datapoint_id,
            episode_id: None,
            input: StoredInput {
                system: Some(System::Template(Arguments(
                    json!({"assistant_name": "TestBot"})
                        .as_object()
                        .unwrap()
                        .clone(),
                ))),
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
            tags: Some(tags.clone()),
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

        // Get the datapoint via the endpoint
        let resp = http_client
            .post(get_gateway_endpoint("/v1/datasets/get_datapoints"))
            .json(&json!({
                "ids": [datapoint_id.to_string()]
            }))
            .send()
            .await
            .unwrap();

        assert!(
            resp.status().is_success(),
            "Request failed: {:?}",
            resp.status()
        );

        let resp_json: Value = resp.json().await.unwrap();
        let datapoints = resp_json["datapoints"].as_array().unwrap();
        assert_eq!(datapoints.len(), 1);

        let dp = &datapoints[0];
        assert_eq!(dp["id"], datapoint_id.to_string());
        assert_eq!(dp["type"], "chat");
        assert_eq!(dp["dataset_name"], dataset_name);
        assert_eq!(dp["function_name"], "basic_test");
        assert_eq!(dp["name"], "Test Datapoint");
        assert_eq!(dp["tags"]["env"], "test");
        assert_eq!(dp["output"][0]["type"], "text");
        assert_eq!(dp["output"][0]["text"], "Hi there!");
    }

    #[tokio::test]
    async fn test_get_datapoints_single_chat_datapoint() {
        let http_client = Client::new();
        let clickhouse = get_clickhouse().await;
        let dataset_name = format!("test-get-dp-single-chat-{}", Uuid::now_v7());

        // Create a chat datapoint
        let datapoint_id = Uuid::now_v7();
        let mut tags = HashMap::new();
        tags.insert("env".to_string(), "test".to_string());

        let datapoint_insert = DatapointInsert::Chat(ChatInferenceDatapointInsert {
            dataset_name: dataset_name.clone(),
            function_name: "basic_test".to_string(),
            name: Some("Test Datapoint".to_string()),
            id: datapoint_id,
            episode_id: None,
            input: StoredInput {
                system: Some(System::Template(Arguments(
                    json!({"assistant_name": "TestBot"})
                        .as_object()
                        .unwrap()
                        .clone(),
                ))),
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
            tags: Some(tags.clone()),
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

        // Get the datapoint via the endpoint
        let resp = http_client
            .post(get_gateway_endpoint(&format!(
                "/v1/datasets/{dataset_name}/get_datapoints"
            )))
            .json(&json!({
                "ids": [datapoint_id.to_string()]
            }))
            .send()
            .await
            .unwrap();

        assert!(
            resp.status().is_success(),
            "Request failed: {:?}",
            resp.status()
        );

        let resp_json: Value = resp.json().await.unwrap();
        let datapoints = resp_json["datapoints"].as_array().unwrap();
        assert_eq!(datapoints.len(), 1);

        let dp = &datapoints[0];
        assert_eq!(dp["id"], datapoint_id.to_string());
        assert_eq!(dp["type"], "chat");
        assert_eq!(dp["dataset_name"], dataset_name);
        assert_eq!(dp["function_name"], "basic_test");
        assert_eq!(dp["name"], "Test Datapoint");
        assert_eq!(dp["tags"]["env"], "test");
        assert_eq!(dp["output"][0]["type"], "text");
        assert_eq!(dp["output"][0]["text"], "Hi there!");
    }

    #[tokio::test]
    async fn test_get_datapoints_single_json_datapoint() {
        let http_client = Client::new();
        let clickhouse = get_clickhouse().await;
        let dataset_name = format!("test-get-dp-single-json-{}", Uuid::now_v7());

        // Create a JSON datapoint
        let datapoint_id = Uuid::now_v7();
        let output_schema = json!({
            "type": "object",
            "properties": {"answer": {"type": "string"}},
            "required": ["answer"],
            "additionalProperties": false
        });

        let datapoint_insert = DatapointInsert::Json(JsonInferenceDatapointInsert {
            dataset_name: dataset_name.clone(),
            function_name: "json_success".to_string(),
            name: None,
            id: datapoint_id,
            episode_id: None,
            input: StoredInput {
                system: Some(System::Template(Arguments(
                    json!({"assistant_name": "JsonBot"})
                        .as_object()
                        .unwrap()
                        .clone(),
                ))),
                messages: vec![StoredInputMessage {
                    role: Role::User,
                    content: vec![StoredInputMessageContent::Text(Text {
                        text: json!({"query": "test"}).to_string(),
                    })],
                }],
            },
            output: Some(JsonInferenceOutput {
                raw: Some(r#"{"answer":"test_answer"}"#.to_string()),
                parsed: Some(json!({"answer": "test_answer"})),
            }),
            output_schema,
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

        // Get the datapoint via the endpoint
        let resp = http_client
            .post(get_gateway_endpoint(&format!(
                "/v1/datasets/{dataset_name}/get_datapoints"
            )))
            .json(&json!({
                "ids": [datapoint_id.to_string()]
            }))
            .send()
            .await
            .unwrap();

        assert!(resp.status().is_success());

        let resp_json: Value = resp.json().await.unwrap();
        let datapoints = resp_json["datapoints"].as_array().unwrap();
        assert_eq!(datapoints.len(), 1);

        let dp = &datapoints[0];
        assert_eq!(dp["id"], datapoint_id.to_string());
        assert_eq!(dp["type"], "json");
        assert_eq!(dp["dataset_name"], dataset_name);
        assert_eq!(dp["function_name"], "json_success");
        assert_eq!(dp["output"]["parsed"]["answer"], "test_answer");
    }

    #[tokio::test]
    async fn test_get_datapoints_multiple_mixed_datapoints() {
        let http_client = Client::new();
        let clickhouse = get_clickhouse().await;
        let dataset_name = format!("test-get-dp-multiple-{}", Uuid::now_v7());

        // Create multiple datapoints
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

        // Get all three datapoints
        let resp = http_client
            .post(get_gateway_endpoint(&format!(
                "/v1/datasets/{dataset_name}/get_datapoints"
            )))
            .json(&json!({
                "ids": [chat_id1.to_string(), chat_id2.to_string(), json_id.to_string()]
            }))
            .send()
            .await
            .unwrap();

        assert!(resp.status().is_success());

        let resp_json: Value = resp.json().await.unwrap();
        let datapoints = resp_json["datapoints"].as_array().unwrap();
        assert_eq!(datapoints.len(), 3);

        // Verify we got all three IDs
        let returned_ids: Vec<String> = datapoints
            .iter()
            .map(|dp| dp["id"].as_str().unwrap().to_string())
            .collect();
        assert!(returned_ids.contains(&chat_id1.to_string()));
        assert!(returned_ids.contains(&chat_id2.to_string()));
        assert!(returned_ids.contains(&json_id.to_string()));

        // Count types
        let chat_count = datapoints.iter().filter(|dp| dp["type"] == "chat").count();
        let json_count = datapoints.iter().filter(|dp| dp["type"] == "json").count();
        assert_eq!(chat_count, 2);
        assert_eq!(json_count, 1);
    }

    #[tokio::test]
    async fn test_get_datapoints_with_non_existent_ids() {
        let http_client = Client::new();
        let clickhouse = get_clickhouse().await;
        let dataset_name = format!("test-get-dp-non-existent-{}", Uuid::now_v7());

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
                    text: "Test".to_string(),
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

        // Query with both existing and non-existent IDs
        let non_existent_id1 = Uuid::now_v7();
        let non_existent_id2 = Uuid::now_v7();

        let resp = http_client
        .post(get_gateway_endpoint(&format!(
            "/v1/datasets/{dataset_name}/get_datapoints"
        )))
        .json(&json!({
            "ids": [existing_id.to_string(), non_existent_id1.to_string(), non_existent_id2.to_string()]
        }))
        .send()
        .await
        .unwrap();

        assert!(resp.status().is_success());

        let resp_json: Value = resp.json().await.unwrap();
        let datapoints = resp_json["datapoints"].as_array().unwrap();

        // Should only return the existing datapoint
        assert_eq!(datapoints.len(), 1);
        assert_eq!(datapoints[0]["id"], existing_id.to_string());
    }

    #[tokio::test]
    async fn test_get_datapoints_returns_stale_datapoints() {
        let http_client = Client::new();
        let clickhouse = get_clickhouse().await;
        let dataset_name = format!("test-get-dp-stale-{}", Uuid::now_v7());

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
                    text: "Original output".to_string(),
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

        // Mark it as stale
        clickhouse
            .delete_datapoints(&dataset_name, Some(&[datapoint_id]))
            .await
            .unwrap();

        tokio::time::sleep(Duration::from_millis(500)).await;

        // get_datapoints should return stale datapoints (unlike list_datapoints)
        let resp = http_client
            .post(get_gateway_endpoint(&format!(
                "/v1/datasets/{dataset_name}/get_datapoints"
            )))
            .json(&json!({
                "ids": [datapoint_id.to_string()]
            }))
            .send()
            .await
            .unwrap();

        assert!(resp.status().is_success());

        let resp_json: Value = resp.json().await.unwrap();
        let datapoints = resp_json["datapoints"].as_array().unwrap();
        assert_eq!(datapoints.len(), 1);
        assert_eq!(datapoints[0]["id"], datapoint_id.to_string());
        // The staled_at field should be present
        assert!(datapoints[0]["staled_at"].is_string());
    }

    #[tokio::test]
    async fn test_get_datapoints_empty_ids_list() {
        let http_client = Client::new();
        let clickhouse = get_clickhouse().await;
        let dataset_name = format!("test-get-dp-empty-ids-list-{}", Uuid::now_v7());

        // Create a datapoint so we have a valid dataset name.
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
                    text: "Original output".to_string(),
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

        let resp = http_client
            .post(get_gateway_endpoint(&format!(
                "/v1/datasets/{dataset_name}/get_datapoints"
            )))
            .json(&json!({
                "ids": []
            }))
            .send()
            .await
            .unwrap();

        assert!(resp.status().is_success());

        let resp_json: Value = resp.json().await.unwrap();
        let datapoints = resp_json["datapoints"].as_array().unwrap();
        assert_eq!(datapoints.len(), 0);
    }

    #[tokio::test]
    async fn test_get_datapoints_invalid_uuid() {
        // Create a valid dataset name so we have a valid dataset name.
        let dataset_name = format!("test-get-dp-invalid-uuid-{}", Uuid::now_v7());
        let http_client = Client::new();
        let clickhouse = get_clickhouse().await;

        // Create a datapoint so we have a valid dataset name.
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
                    text: "Original output".to_string(),
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

        let resp = http_client
            .post(get_gateway_endpoint(&format!(
                "/v1/datasets/{dataset_name}/get_datapoints"
            )))
            .json(&json!({
                "ids": ["not-a-valid-uuid"]
            }))
            .send()
            .await
            .unwrap();

        // Should return a 400 error for invalid UUID
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    }
}

/// Tests for the /v1/datasets/{dataset_name}/list_datapoints endpoint.
mod list_datapoints_tests {
    use super::*;

    #[tokio::test]
    async fn test_list_datapoints_basic_pagination() {
        let http_client = Client::new();
        let clickhouse = get_clickhouse().await;
        let dataset_name = format!("test-list-dp-pagination-{}", Uuid::now_v7());

        // Create 5 datapoints
        let mut inserts = vec![];
        for i in 0..5 {
            inserts.push(DatapointInsert::Chat(ChatInferenceDatapointInsert {
                dataset_name: dataset_name.clone(),
                function_name: "basic_test".to_string(),
                name: Some(format!("Datapoint {i}")),
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
                output: Some(vec![
                    tensorzero_core::inference::types::ContentBlockChatOutput::Text(Text {
                        text: format!("Response {i}"),
                    }),
                ]),
                tool_params: None,
                tags: None,
                auxiliary: String::new(),
                staled_at: None,
                source_inference_id: None,
                is_custom: true,
            }));
        }

        clickhouse.insert_datapoints(&inserts).await.unwrap();
        tokio::time::sleep(Duration::from_millis(500)).await;

        // Test default pagination (limit: 20, offset: 0)
        let resp = http_client
            .post(get_gateway_endpoint(&format!(
                "/v1/datasets/{dataset_name}/list_datapoints",
            )))
            .json(&json!({}))
            .send()
            .await
            .unwrap();

        assert!(resp.status().is_success());
        let resp_json: Value = resp.json().await.unwrap();
        let datapoints = resp_json["datapoints"].as_array().unwrap();
        assert_eq!(datapoints.len(), 5);

        // Test limit = 2
        let resp = http_client
            .post(get_gateway_endpoint(&format!(
                "/v1/datasets/{dataset_name}/list_datapoints"
            )))
            .json(&json!({
                "limit": 2
            }))
            .send()
            .await
            .unwrap();

        assert!(resp.status().is_success());
        let resp_json: Value = resp.json().await.unwrap();
        let datapoints = resp_json["datapoints"].as_array().unwrap();

        // Ordering is not guaranteed since they are all inserted at the same time, so we just check the length.
        assert_eq!(datapoints.len(), 2);

        // Test offset = 2
        let resp = http_client
            .post(get_gateway_endpoint(&format!(
                "/v1/datasets/{dataset_name}/list_datapoints"
            )))
            .json(&json!({
                "limit": 3,
                "offset": 2
            }))
            .send()
            .await
            .unwrap();

        assert!(resp.status().is_success());
        let resp_json: Value = resp.json().await.unwrap();
        let datapoints = resp_json["datapoints"].as_array().unwrap();
        assert_eq!(datapoints.len(), 3);

        // Test offset beyond available items
        let resp = http_client
            .post(get_gateway_endpoint(&format!(
                "/v1/datasets/{dataset_name}/list_datapoints"
            )))
            .json(&json!({
                "offset": 100
            }))
            .send()
            .await
            .unwrap();

        assert!(resp.status().is_success());
        let resp_json: Value = resp.json().await.unwrap();
        let datapoints = resp_json["datapoints"].as_array().unwrap();
        assert_eq!(datapoints.len(), 0);
    }

    #[tokio::test]
    async fn test_list_datapoints_filter_by_function_name() {
        let http_client = Client::new();
        let clickhouse = get_clickhouse().await;
        let dataset_name = format!("test-list-dp-function-{}", Uuid::now_v7());

        // Create datapoints with different function names
        let function1_id = Uuid::now_v7();
        let function1_insert = DatapointInsert::Chat(ChatInferenceDatapointInsert {
            dataset_name: dataset_name.clone(),
            function_name: "basic_test".to_string(),
            name: None,
            id: function1_id,
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

        let function2_id = Uuid::now_v7();
        let function2_insert = DatapointInsert::Chat(ChatInferenceDatapointInsert {
            dataset_name: dataset_name.clone(),
            function_name: "weather_helper".to_string(),
            name: None,
            id: function2_id,
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

        clickhouse
            .insert_datapoints(&[function1_insert, function2_insert])
            .await
            .unwrap();

        tokio::time::sleep(Duration::from_millis(500)).await;

        // List without function filter - should get both
        let resp = http_client
            .post(get_gateway_endpoint(&format!(
                "/v1/datasets/{dataset_name}/list_datapoints"
            )))
            .json(&json!({}))
            .send()
            .await
            .unwrap();

        assert!(resp.status().is_success());
        let resp_json: Value = resp.json().await.unwrap();
        let datapoints = resp_json["datapoints"].as_array().unwrap();
        assert_eq!(datapoints.len(), 2);

        // List with function_name = "function_one"
        let resp = http_client
            .post(get_gateway_endpoint(&format!(
                "/v1/datasets/{dataset_name}/list_datapoints"
            )))
            .json(&json!({
                "function_name": "basic_test"
            }))
            .send()
            .await
            .unwrap();

        assert!(resp.status().is_success());
        let resp_json: Value = resp.json().await.unwrap();
        let datapoints = resp_json["datapoints"].as_array().unwrap();
        assert_eq!(datapoints.len(), 1);
        assert_eq!(datapoints[0]["id"], function1_id.to_string());
        assert_eq!(datapoints[0]["function_name"], "basic_test");
    }

    #[tokio::test]
    async fn test_list_datapoints_filter_by_tags() {
        let http_client = Client::new();
        let clickhouse = get_clickhouse().await;
        let dataset_name = format!("test-list-dp-tags-{}", Uuid::now_v7());

        // Create datapoints with different tags
        let mut tags1 = HashMap::new();
        tags1.insert("env".to_string(), "production".to_string());
        tags1.insert("version".to_string(), "v1".to_string());

        let mut tags2 = HashMap::new();
        tags2.insert("env".to_string(), "staging".to_string());

        let datapoint1_id = Uuid::now_v7();
        let datapoint1 = DatapointInsert::Chat(ChatInferenceDatapointInsert {
            dataset_name: dataset_name.clone(),
            function_name: "basic_test".to_string(),
            name: None,
            id: datapoint1_id,
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
            output: Some(vec![
                tensorzero_core::inference::types::ContentBlockChatOutput::Text(Text {
                    text: "Response 1".to_string(),
                }),
            ]),
            tool_params: None,
            tags: Some(tags1),
            auxiliary: String::new(),
            staled_at: None,
            source_inference_id: None,
            is_custom: true,
        });

        let datapoint2_id = Uuid::now_v7();
        let datapoint2 = DatapointInsert::Chat(ChatInferenceDatapointInsert {
            dataset_name: dataset_name.clone(),
            function_name: "basic_test".to_string(),
            name: None,
            id: datapoint2_id,
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
            output: Some(vec![
                tensorzero_core::inference::types::ContentBlockChatOutput::Text(Text {
                    text: "Response 2".to_string(),
                }),
            ]),
            tool_params: None,
            tags: Some(tags2),
            auxiliary: String::new(),
            staled_at: None,
            source_inference_id: None,
            is_custom: true,
        });

        clickhouse
            .insert_datapoints(&[datapoint1, datapoint2])
            .await
            .unwrap();

        tokio::time::sleep(Duration::from_millis(500)).await;

        // Filter by env = production
        let resp = http_client
            .post(get_gateway_endpoint(&format!(
                "/v1/datasets/{dataset_name}/list_datapoints"
            )))
            .json(&json!({
                "filter": {
                    "type": "tag",
                    "key": "env",
                    "comparison_operator": "=",
                    "value": "production"
                }
            }))
            .send()
            .await
            .unwrap();

        assert!(resp.status().is_success());
        let resp_json: Value = resp.json().await.unwrap();
        let datapoints = resp_json["datapoints"].as_array().unwrap();
        assert_eq!(datapoints.len(), 1);
        assert_eq!(datapoints[0]["id"], datapoint1_id.to_string());

        // Filter by version exists
        let resp = http_client
            .post(get_gateway_endpoint(&format!(
                "/v1/datasets/{dataset_name}/list_datapoints"
            )))
            .json(&json!({
                "filter": {
                    "type": "tag",
                    "key": "env",
                    "comparison_operator": "!=",
                    "value": "production"
                }
            }))
            .send()
            .await
            .unwrap();

        assert!(resp.status().is_success());
        let resp_json: Value = resp.json().await.unwrap();
        let datapoints = resp_json["datapoints"].as_array().unwrap();
        assert_eq!(datapoints.len(), 1);
        assert_eq!(datapoints[0]["id"], datapoint2_id.to_string());
    }

    #[tokio::test]
    async fn test_list_datapoints_filter_by_time() {
        let http_client = Client::new();
        let clickhouse = get_clickhouse().await;
        let dataset_name = format!("test-list-dp-time-{}", Uuid::now_v7());

        // Create a datapoint
        let datapoint_id = Uuid::now_v7();
        let datapoint = DatapointInsert::Chat(ChatInferenceDatapointInsert {
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

        clickhouse.insert_datapoints(&[datapoint]).await.unwrap();
        tokio::time::sleep(Duration::from_millis(500)).await;

        // Filter with time before (should not return the datapoint)
        // Use a time in the past
        let time_before = chrono::Utc::now() - chrono::Duration::hours(24);
        let resp = http_client
            .post(get_gateway_endpoint(&format!(
                "/v1/datasets/{dataset_name}/list_datapoints"
            )))
            .json(&json!({
                "filter": {
                    "type": "time",
                    "comparison_operator": "<",
                    "time": time_before.to_rfc3339()
                }
            }))
            .send()
            .await
            .unwrap();

        assert!(resp.status().is_success());
        let resp_json: Value = resp.json().await.unwrap();
        let datapoints = resp_json["datapoints"].as_array().unwrap();
        assert_eq!(datapoints.len(), 0);

        // Filter with time after (should return the datapoint)
        // Use a time well in the past
        let time_after = chrono::Utc::now() - chrono::Duration::hours(24);
        let resp = http_client
            .post(get_gateway_endpoint(&format!(
                "/v1/datasets/{dataset_name}/list_datapoints"
            )))
            .json(&json!({
                "filter": {
                    "type": "time",
                    "comparison_operator": ">",
                    "time": time_after.to_rfc3339()
                }
            }))
            .send()
            .await
            .unwrap();

        assert!(resp.status().is_success());
        let resp_json: Value = resp.json().await.unwrap();
        let datapoints = resp_json["datapoints"].as_array().unwrap();
        assert_eq!(datapoints.len(), 1);
        assert_eq!(datapoints[0]["id"], datapoint_id.to_string());
    }

    #[tokio::test]
    async fn test_list_datapoints_complex_filters() {
        let http_client = Client::new();
        let clickhouse = get_clickhouse().await;
        let dataset_name = format!("test-list-dp-complex-{}", Uuid::now_v7());

        // Create datapoints with various tags
        let mut tags1 = HashMap::new();
        tags1.insert("env".to_string(), "production".to_string());
        tags1.insert("region".to_string(), "us-east".to_string());

        let mut tags2 = HashMap::new();
        tags2.insert("env".to_string(), "production".to_string());
        tags2.insert("region".to_string(), "us-west".to_string());

        let mut tags3 = HashMap::new();
        tags3.insert("env".to_string(), "staging".to_string());
        tags3.insert("region".to_string(), "us-east".to_string());

        let datapoint1_id = Uuid::now_v7();
        let datapoint1 = DatapointInsert::Chat(ChatInferenceDatapointInsert {
            dataset_name: dataset_name.clone(),
            function_name: "basic_test".to_string(),
            name: None,
            id: datapoint1_id,
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
            output: Some(vec![
                tensorzero_core::inference::types::ContentBlockChatOutput::Text(Text {
                    text: "Response 1".to_string(),
                }),
            ]),
            tool_params: None,
            tags: Some(tags1),
            auxiliary: String::new(),
            staled_at: None,
            source_inference_id: None,
            is_custom: true,
        });

        let datapoint2_id = Uuid::now_v7();
        let datapoint2 = DatapointInsert::Chat(ChatInferenceDatapointInsert {
            dataset_name: dataset_name.clone(),
            function_name: "basic_test".to_string(),
            name: None,
            id: datapoint2_id,
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
            output: Some(vec![
                tensorzero_core::inference::types::ContentBlockChatOutput::Text(Text {
                    text: "Response 2".to_string(),
                }),
            ]),
            tool_params: None,
            tags: Some(tags2),
            auxiliary: String::new(),
            staled_at: None,
            source_inference_id: None,
            is_custom: true,
        });

        let datapoint3_id = Uuid::now_v7();
        let datapoint3 = DatapointInsert::Chat(ChatInferenceDatapointInsert {
            dataset_name: dataset_name.clone(),
            function_name: "basic_test".to_string(),
            name: None,
            id: datapoint3_id,
            episode_id: None,
            input: StoredInput {
                system: None,
                messages: vec![StoredInputMessage {
                    role: Role::User,
                    content: vec![StoredInputMessageContent::Text(Text {
                        text: "Test 3".to_string(),
                    })],
                }],
            },
            output: Some(vec![
                tensorzero_core::inference::types::ContentBlockChatOutput::Text(Text {
                    text: "Response 3".to_string(),
                }),
            ]),
            tool_params: None,
            tags: Some(tags3),
            auxiliary: String::new(),
            staled_at: None,
            source_inference_id: None,
            is_custom: true,
        });

        clickhouse
            .insert_datapoints(&[datapoint1, datapoint2, datapoint3])
            .await
            .unwrap();

        tokio::time::sleep(Duration::from_millis(500)).await;

        // AND filter: env = production AND region = us-east
        let resp = http_client
            .post(get_gateway_endpoint(&format!(
                "/v1/datasets/{dataset_name}/list_datapoints"
            )))
            .json(&json!({
                "filter": {
                    "type": "and",
                    "children": [
                        {
                            "type": "tag",
                            "key": "env",
                            "comparison_operator": "=",
                            "value": "production"
                        },
                        {
                            "type": "tag",
                            "key": "region",
                            "comparison_operator": "=",
                            "value": "us-east"
                        }
                    ]
                }
            }))
            .send()
            .await
            .unwrap();

        assert!(resp.status().is_success());
        let resp_json: Value = resp.json().await.unwrap();
        let datapoints = resp_json["datapoints"].as_array().unwrap();
        assert_eq!(datapoints.len(), 1);
        assert_eq!(datapoints[0]["id"], datapoint1_id.to_string());

        // OR filter: env = production OR region = us-east
        let resp = http_client
            .post(get_gateway_endpoint(&format!(
                "/v1/datasets/{dataset_name}/list_datapoints"
            )))
            .json(&json!({
                "filter": {
                    "type": "or",
                    "children": [
                        {
                            "type": "tag",
                            "key": "env",
                            "comparison_operator": "=",
                            "value": "production"
                        },
                        {
                            "type": "tag",
                            "key": "region",
                            "comparison_operator": "=",
                            "value": "us-east"
                        }
                    ]
                }
            }))
            .send()
            .await
            .unwrap();

        assert!(resp.status().is_success());
        let resp_json: Value = resp.json().await.unwrap();
        let datapoints = resp_json["datapoints"].as_array().unwrap();
        // Should return all 3 datapoints
        assert_eq!(datapoints.len(), 3);
    }

    #[tokio::test]
    async fn test_list_datapoints_empty_dataset() {
        let http_client = Client::new();
        let dataset_name = format!("test-list-dp-empty-{}", Uuid::now_v7());

        // List from a dataset that doesn't exist / has no datapoints
        let resp = http_client
            .post(get_gateway_endpoint(&format!(
                "/v1/datasets/{dataset_name}/list_datapoints"
            )))
            .json(&json!({}))
            .send()
            .await
            .unwrap();

        assert!(resp.status().is_success());
        let resp_json: Value = resp.json().await.unwrap();
        let datapoints = resp_json["datapoints"].as_array().unwrap();
        assert_eq!(datapoints.len(), 0);
    }

    #[tokio::test]
    async fn test_list_datapoints_does_not_return_stale() {
        let http_client = Client::new();
        let clickhouse = get_clickhouse().await;
        let dataset_name = format!("test-list-dp-no-stale-{}", Uuid::now_v7());

        // Create a datapoint
        let datapoint_id = Uuid::now_v7();
        let datapoint = DatapointInsert::Chat(ChatInferenceDatapointInsert {
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
                    text: "Original output".to_string(),
                }),
            ]),
            tool_params: None,
            tags: None,
            auxiliary: String::new(),
            staled_at: None,
            source_inference_id: None,
            is_custom: true,
        });

        clickhouse.insert_datapoints(&[datapoint]).await.unwrap();
        tokio::time::sleep(Duration::from_millis(500)).await;

        // Verify it's returned before staling
        let resp = http_client
            .post(get_gateway_endpoint(&format!(
                "/v1/datasets/{dataset_name}/list_datapoints"
            )))
            .json(&json!({}))
            .send()
            .await
            .unwrap();

        assert!(resp.status().is_success());
        let resp_json: Value = resp.json().await.unwrap();
        let datapoints = resp_json["datapoints"].as_array().unwrap();
        assert_eq!(datapoints.len(), 1);
        assert_eq!(datapoints[0]["id"], datapoint_id.to_string());

        // Mark it as stale
        clickhouse
            .delete_datapoints(&dataset_name, Some(&[datapoint_id]))
            .await
            .unwrap();

        tokio::time::sleep(Duration::from_millis(500)).await;

        // Verify list_datapoints no longer returns it
        let resp = http_client
            .post(get_gateway_endpoint(&format!(
                "/v1/datasets/{dataset_name}/list_datapoints"
            )))
            .json(&json!({}))
            .send()
            .await
            .unwrap();

        assert!(resp.status().is_success());
        let resp_json: Value = resp.json().await.unwrap();
        let datapoints = resp_json["datapoints"].as_array().unwrap();
        assert_eq!(datapoints.len(), 0);
    }

    #[tokio::test]
    async fn test_list_datapoints_mixed_chat_and_json() {
        let http_client = Client::new();
        let clickhouse = get_clickhouse().await;
        let dataset_name = format!("test-list-dp-mixed-{}", Uuid::now_v7());

        // Create both chat and JSON datapoints
        let chat_id = Uuid::now_v7();
        let chat_insert = DatapointInsert::Chat(ChatInferenceDatapointInsert {
            dataset_name: dataset_name.clone(),
            function_name: "basic_test".to_string(),
            name: None,
            id: chat_id,
            episode_id: None,
            input: StoredInput {
                system: None,
                messages: vec![StoredInputMessage {
                    role: Role::User,
                    content: vec![StoredInputMessageContent::Text(Text {
                        text: "Chat test".to_string(),
                    })],
                }],
            },
            output: Some(vec![
                tensorzero_core::inference::types::ContentBlockChatOutput::Text(Text {
                    text: "Chat response".to_string(),
                }),
            ]),
            tool_params: None,
            tags: None,
            auxiliary: String::new(),
            staled_at: None,
            source_inference_id: None,
            is_custom: true,
        });

        let json_id = Uuid::now_v7();
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
                        text: "JSON test".to_string(),
                    })],
                }],
            },
            output: Some(JsonInferenceOutput {
                raw: Some(r#"{"result":"success"}"#.to_string()),
                parsed: Some(json!({"result": "success"})),
            }),
            output_schema: json!({"type": "object"}),
            tags: None,
            auxiliary: String::new(),
            staled_at: None,
            source_inference_id: None,
            is_custom: true,
        });

        clickhouse
            .insert_datapoints(&[chat_insert, json_insert])
            .await
            .unwrap();

        tokio::time::sleep(Duration::from_millis(500)).await;

        // List all datapoints
        let resp = http_client
            .post(get_gateway_endpoint(&format!(
                "/v1/datasets/{dataset_name}/list_datapoints"
            )))
            .json(&json!({}))
            .send()
            .await
            .unwrap();

        assert!(resp.status().is_success());
        let resp_json: Value = resp.json().await.unwrap();
        let datapoints = resp_json["datapoints"].as_array().unwrap();
        assert_eq!(datapoints.len(), 2);

        // Ordering is not guaranteed, so we check for containment
        let ids = datapoints
            .iter()
            .map(|dp| {
                let Value::String(id_str) = dp["id"].clone() else {
                    panic!("id is not a string")
                };
                id_str
            })
            .collect::<Vec<_>>();

        assert!(ids.contains(&chat_id.to_string()));
        assert!(ids.contains(&json_id.to_string()));
    }

    #[tokio::test]
    async fn test_list_datapoints_with_large_limit() {
        let http_client = Client::new();
        let clickhouse = get_clickhouse().await;
        let dataset_name = format!("test-list-dp-large-page-{}", Uuid::now_v7());

        // Create 3 datapoints
        let mut inserts = vec![];
        for i in 0..3 {
            inserts.push(DatapointInsert::Chat(ChatInferenceDatapointInsert {
                dataset_name: dataset_name.clone(),
                function_name: "basic_test".to_string(),
                name: None,
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
                output: Some(vec![
                    tensorzero_core::inference::types::ContentBlockChatOutput::Text(Text {
                        text: format!("Response {i}"),
                    }),
                ]),
                tool_params: None,
                tags: None,
                auxiliary: String::new(),
                staled_at: None,
                source_inference_id: None,
                is_custom: true,
            }));
        }

        clickhouse.insert_datapoints(&inserts).await.unwrap();
        tokio::time::sleep(Duration::from_millis(500)).await;

        // Request with limit larger than available datapoints
        let resp = http_client
            .post(get_gateway_endpoint(&format!(
                "/v1/datasets/{dataset_name}/list_datapoints"
            )))
            .json(&json!({
                "limit": 100
            }))
            .send()
            .await
            .unwrap();

        assert!(resp.status().is_success());
        let resp_json: Value = resp.json().await.unwrap();
        let datapoints = resp_json["datapoints"].as_array().unwrap();
        // Should return all 3 datapoints, not 100
        assert_eq!(datapoints.len(), 3);
    }
}
