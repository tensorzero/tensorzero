/// These tests directly talk to ClickHouse to insert datapoints, and use
/// the HTTP endpoint to perform updates to validate that updates are implemented
/// correctly.
use reqwest::{Client, StatusCode};
use serde_json::{json, Value};
use std::collections::HashMap;
use std::time::Duration;
use tensorzero::{FunctionTool, GetDatapointParams, StoredDatapoint};
use uuid::Uuid;

use tensorzero_core::db::clickhouse::test_helpers::{
    clickhouse_flush_async_insert, get_clickhouse,
};
use tensorzero_core::db::datasets::{
    ChatInferenceDatapointInsert, DatapointInsert, DatasetQueries, JsonInferenceDatapointInsert,
};
use tensorzero_core::inference::types::{
    Arguments, ContentBlockChatOutput, JsonInferenceOutput, Role, StoredInput, StoredInputMessage,
    StoredInputMessageContent, System, Template, Text,
};
use tensorzero_core::tool::{
    AllowedTools, AllowedToolsChoice, ProviderTool, ProviderToolScope, Tool,
    ToolCallConfigDatabaseInsert, ToolChoice,
};

use crate::common::get_gateway_endpoint;

#[tokio::test]
async fn test_update_chat_datapoint_output() {
    let http_client = Client::new();
    let clickhouse = get_clickhouse().await;
    let dataset_name = format!("test-update-chat-{}", Uuid::now_v7());

    // First, create a datapoint
    let datapoint_id = Uuid::now_v7();
    let mut tags = std::collections::HashMap::new();
    tags.insert("version".to_string(), "1".to_string());

    let datapoint_insert = DatapointInsert::Chat(ChatInferenceDatapointInsert {
        dataset_name: dataset_name.clone(),
        function_name: "basic_test".to_string(),
        name: None,
        id: datapoint_id,
        episode_id: None,
        input: StoredInput {
            system: Some(System::Template(Arguments(
                json!({"assistant_name": "Test"})
                    .as_object()
                    .unwrap()
                    .clone(),
            ))),
            messages: vec![StoredInputMessage {
                role: Role::User,
                content: vec![StoredInputMessageContent::Text(Text {
                    text: "Original message".to_string(),
                })],
            }],
        },
        output: Some(vec![ContentBlockChatOutput::Text(Text {
            text: "Original output".to_string(),
        })]),
        tool_params: None,
        tags: Some(tags),
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

    // Now update the datapoint with new output and tags
    let resp = http_client
        .patch(get_gateway_endpoint(&format!(
            "/v1/datasets/{dataset_name}/datapoints",
        )))
        .json(&json!({
            "datapoints": [{
                "type": "chat",
                "id": datapoint_id.to_string(),
                "output": [{"type": "text", "text": "Updated output"}],
                "tags": {"version": "2"}
            }]
        }))
        .send()
        .await
        .unwrap();

    assert!(
        resp.status().is_success(),
        "Update request failed: {:?}",
        resp.status()
    );
    let resp_json: Value = resp.json().await.unwrap();
    let new_ids = resp_json["ids"].as_array().unwrap();
    assert_eq!(new_ids.len(), 1);
    let new_id: Uuid = new_ids[0].as_str().unwrap().parse().unwrap();
    assert_ne!(new_id, datapoint_id, "Should create a new datapoint ID");

    // Wait for async inserts and give ClickHouse time to merge the staled version

    clickhouse_flush_async_insert(&clickhouse).await;
    tokio::time::sleep(Duration::from_millis(1000)).await;

    // Verify the old datapoint is staled
    let old_datapoint = clickhouse
        .get_datapoint(&GetDatapointParams {
            dataset_name: dataset_name.clone(),
            datapoint_id,
            allow_stale: Some(true),
        })
        .await
        .unwrap();
    let StoredDatapoint::Chat(chat_datapoint) = old_datapoint else {
        panic!("Expected chat datapoint");
    };

    assert!(chat_datapoint.staled_at.is_some());
    assert_eq!(
        chat_datapoint.output,
        Some(vec![ContentBlockChatOutput::Text(Text {
            text: "Original output".to_string(),
        })])
    );

    // Verify the new datapoint has updated values
    let new_datapoint = clickhouse
        .get_datapoint(&GetDatapointParams {
            dataset_name: dataset_name.clone(),
            datapoint_id: new_id,
            allow_stale: Some(false),
        })
        .await
        .unwrap();
    let StoredDatapoint::Chat(chat_datapoint) = new_datapoint else {
        panic!("Expected chat datapoint");
    };
    assert!(chat_datapoint.staled_at.is_none());
    assert_eq!(
        chat_datapoint.output,
        Some(vec![ContentBlockChatOutput::Text(Text {
            text: "Updated output".to_string(),
        })])
    );
    assert_eq!(
        chat_datapoint.tags,
        Some(HashMap::from([("version".to_string(), "2".to_string())]))
    );
}

#[tokio::test]
async fn test_update_json_datapoint_output() {
    let http_client = Client::new();
    let clickhouse = get_clickhouse().await;
    let dataset_name = format!("test-update-json-{}", Uuid::now_v7());

    // First, create a JSON datapoint
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
                json!({"assistant_name": "Test"})
                    .as_object()
                    .unwrap()
                    .clone(),
            ))),
            messages: vec![StoredInputMessage {
                role: Role::User,
                content: vec![StoredInputMessageContent::Text(Text {
                    text: json!({"country": "US"}).to_string(),
                })],
            }],
        },
        output: Some(JsonInferenceOutput {
            raw: Some(r#"{"answer":"original"}"#.to_string()),
            parsed: Some(json!({"answer": "original"})),
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

    // Update the datapoint with new output
    let resp = http_client
        .patch(get_gateway_endpoint(&format!(
            "/v1/datasets/{dataset_name}/datapoints",
        )))
        .json(&json!({
            "datapoints": [{
                "type": "json",
                "id": datapoint_id.to_string(),
                "output": {
                    "raw": "{\"answer\": \"updated\"}",
                },
            }]
        }))
        .send()
        .await
        .unwrap();

    if !resp.status().is_success() {
        let status = resp.status();
        let error_text = resp.text().await.unwrap();
        panic!("Update request failed with status {status}: {error_text}");
    }
    let resp_json: Value = resp.json().await.unwrap();
    let new_ids = resp_json["ids"].as_array().unwrap();
    assert_eq!(new_ids.len(), 1);
    let new_id: Uuid = new_ids[0].as_str().unwrap().parse().unwrap();

    tokio::time::sleep(Duration::from_millis(500)).await;

    // Verify the new datapoint has updated output
    let new_datapoint = clickhouse
        .get_datapoint(&GetDatapointParams {
            dataset_name: dataset_name.clone(),
            datapoint_id: new_id,
            allow_stale: Some(false),
        })
        .await
        .unwrap();
    let StoredDatapoint::Json(json_datapoint) = new_datapoint else {
        panic!("Expected json datapoint");
    };
    assert!(json_datapoint.staled_at.is_none());
    let output: JsonInferenceOutput = json_datapoint.output.unwrap();
    assert_eq!(output.parsed.unwrap(), json!({"answer": "updated"}));
}

#[tokio::test]
async fn test_update_multiple_datapoints() {
    let http_client = Client::new();
    let clickhouse = get_clickhouse().await;
    let dataset_name = format!("test-update-batch-{}", Uuid::now_v7());

    // Create two datapoints
    let datapoint_id1 = Uuid::now_v7();
    let datapoint_id2 = Uuid::now_v7();

    let datapoint_insert1 = DatapointInsert::Chat(ChatInferenceDatapointInsert {
        dataset_name: dataset_name.clone(),
        function_name: "basic_test".to_string(),
        name: None,
        id: datapoint_id1,
        episode_id: None,
        input: StoredInput {
            system: Some(System::Template(Arguments(
                json!({"assistant_name": "Test"})
                    .as_object()
                    .unwrap()
                    .clone(),
            ))),
            messages: vec![StoredInputMessage {
                role: Role::User,
                content: vec![StoredInputMessageContent::Text(Text {
                    text: "Message 1".to_string(),
                })],
            }],
        },
        output: Some(vec![ContentBlockChatOutput::Text(Text {
            text: "Output 1".to_string(),
        })]),
        tool_params: None,
        tags: None,
        auxiliary: String::new(),
        staled_at: None,
        source_inference_id: None,
        is_custom: true,
    });

    let datapoint_insert2 = DatapointInsert::Chat(ChatInferenceDatapointInsert {
        dataset_name: dataset_name.clone(),
        function_name: "basic_test".to_string(),
        name: None,
        id: datapoint_id2,
        episode_id: None,
        input: StoredInput {
            system: Some(System::Template(Arguments(
                json!({"assistant_name": "Test"})
                    .as_object()
                    .unwrap()
                    .clone(),
            ))),
            messages: vec![StoredInputMessage {
                role: Role::User,
                content: vec![StoredInputMessageContent::Text(Text {
                    text: "Message 2".to_string(),
                })],
            }],
        },
        output: Some(vec![ContentBlockChatOutput::Text(Text {
            text: "Output 2".to_string(),
        })]),
        tool_params: None,
        tags: None,
        auxiliary: String::new(),
        staled_at: None,
        source_inference_id: None,
        is_custom: true,
    });

    let insert_result = clickhouse
        .insert_datapoints(&[datapoint_insert1, datapoint_insert2])
        .await
        .unwrap();
    assert_eq!(
        insert_result, 2,
        "Should insert 2 datapoints in preparation for batch update"
    );

    tokio::time::sleep(Duration::from_millis(500)).await;

    // Update both datapoints in one request
    let resp = http_client
        .patch(get_gateway_endpoint(&format!(
            "/v1/datasets/{dataset_name}/datapoints",
        )))
        .json(&json!({
            "datapoints": [
                {
                    "type": "chat",
                    "id": datapoint_id1.to_string(),
                    "output": [{"type": "text", "text": "Updated output 1"}],
                },
                {
                    "type": "chat",
                    "id": datapoint_id2.to_string(),
                    "output": [{"type": "text", "text": "Updated output 2"}],
                }
            ]
        }))
        .send()
        .await
        .unwrap();

    assert!(resp.status().is_success());
    let resp_json: Value = resp.json().await.unwrap();
    let new_ids = resp_json["ids"].as_array().unwrap();
    assert_eq!(new_ids.len(), 2);

    // Wait for async inserts and give ClickHouse time to merge the staled versions
    tokio::time::sleep(Duration::from_millis(1000)).await;
    clickhouse_flush_async_insert(&clickhouse).await;
    tokio::time::sleep(Duration::from_millis(1000)).await;

    // Verify both old datapoints are staled
    let old_dp1 = clickhouse
        .get_datapoint(&GetDatapointParams {
            dataset_name: dataset_name.clone(),
            datapoint_id: datapoint_id1,
            allow_stale: Some(true),
        })
        .await
        .unwrap();
    let StoredDatapoint::Chat(chat_datapoint) = old_dp1 else {
        panic!("Expected chat datapoint");
    };
    assert!(chat_datapoint.staled_at.is_some());

    let old_dp2 = clickhouse
        .get_datapoint(&GetDatapointParams {
            dataset_name: dataset_name.clone(),
            datapoint_id: datapoint_id2,
            allow_stale: Some(true),
        })
        .await
        .unwrap();
    let StoredDatapoint::Chat(chat_datapoint) = old_dp2 else {
        panic!("Expected chat datapoint");
    };
    assert!(chat_datapoint.staled_at.is_some());
}

#[tokio::test]
async fn test_update_datapoint_not_found() {
    let http_client = Client::new();
    let dataset_name = format!("test-update-not-found-{}", Uuid::now_v7());
    let nonexistent_id = Uuid::now_v7();

    let resp = http_client
        .patch(get_gateway_endpoint(&format!(
            "/v1/datasets/{dataset_name}/datapoints",
        )))
        .json(&json!({
            "datapoints": [{
                "type": "chat",
                "id": nonexistent_id.to_string(),
                "output": [{"type": "text", "text": "New output"}],
            }]
        }))
        .send()
        .await
        .unwrap();

    assert_eq!(resp.status(), StatusCode::NOT_FOUND);
}

#[tokio::test]
async fn test_update_datapoint_type_mismatch() {
    let http_client = Client::new();
    let clickhouse = get_clickhouse().await;
    let dataset_name = format!("test-update-type-mismatch-{}", Uuid::now_v7());

    // Create a chat datapoint
    let datapoint_id = Uuid::now_v7();

    let datapoint_insert = DatapointInsert::Chat(ChatInferenceDatapointInsert {
        dataset_name: dataset_name.clone(),
        function_name: "basic_test".to_string(),
        name: None,
        id: datapoint_id,
        episode_id: None,
        input: StoredInput {
            system: Some(System::Template(Arguments(
                json!({"assistant_name": "Test"})
                    .as_object()
                    .unwrap()
                    .clone(),
            ))),
            messages: vec![StoredInputMessage {
                role: Role::User,
                content: vec![StoredInputMessageContent::Text(Text {
                    text: "Test".to_string(),
                })],
            }],
        },
        output: Some(vec![ContentBlockChatOutput::Text(Text {
            text: "Output".to_string(),
        })]),
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

    // Try to update it as a JSON datapoint
    let resp = http_client
        .patch(get_gateway_endpoint(&format!(
            "/v1/datasets/{dataset_name}/datapoints",
        )))
        .json(&json!({
            "datapoints": [{
                "type": "json",
                "id": datapoint_id.to_string(),
                "output": {"answer": "test"},
                "output_schema": {"type": "object"}
            }]
        }))
        .send()
        .await
        .unwrap();

    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn test_update_datapoint_with_metadata() {
    let http_client = Client::new();
    let clickhouse = get_clickhouse().await;
    let dataset_name = format!("test-update-metadata-{}", Uuid::now_v7());

    // Create a datapoint
    let datapoint_id = Uuid::now_v7();

    let datapoint_insert = DatapointInsert::Chat(ChatInferenceDatapointInsert {
        dataset_name: dataset_name.clone(),
        function_name: "basic_test".to_string(),
        name: None,
        id: datapoint_id,
        episode_id: None,
        input: StoredInput {
            system: Some(System::Template(Arguments(
                json!({"assistant_name": "Test"})
                    .as_object()
                    .unwrap()
                    .clone(),
            ))),
            messages: vec![StoredInputMessage {
                role: Role::User,
                content: vec![StoredInputMessageContent::Text(Text {
                    text: "Test".to_string(),
                })],
            }],
        },
        output: Some(vec![ContentBlockChatOutput::Text(Text {
            text: "Output".to_string(),
        })]),
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

    // Update with new name metadata
    let resp = http_client
        .patch(get_gateway_endpoint(&format!(
            "/v1/datasets/{dataset_name}/datapoints",
        )))
        .json(&json!({
            "datapoints": [{
                "type": "chat",
                "id": datapoint_id.to_string(),
                "name": "Test Datapoint Name"
            }]
        }))
        .send()
        .await
        .unwrap();

    assert!(resp.status().is_success());
    let resp_json: Value = resp.json().await.unwrap();
    let new_id: Uuid = resp_json["ids"][0].as_str().unwrap().parse().unwrap();

    tokio::time::sleep(Duration::from_millis(500)).await;

    let new_datapoint = clickhouse
        .get_datapoint(&GetDatapointParams {
            dataset_name: dataset_name.clone(),
            datapoint_id: new_id,
            allow_stale: None,
        })
        .await
        .unwrap();

    let StoredDatapoint::Chat(chat_datapoint) = new_datapoint else {
        panic!("Expected chat datapoint");
    };
    assert_eq!(chat_datapoint.name, Some("Test Datapoint Name".to_string()));
}

#[tokio::test]
async fn test_update_datapoint_empty_request() {
    let http_client = Client::new();
    let dataset_name = format!("test-update-empty-{}", Uuid::now_v7());

    let resp = http_client
        .patch(get_gateway_endpoint(&format!(
            "/v1/datasets/{dataset_name}/datapoints",
        )))
        .json(&json!({
            "datapoints": []
        }))
        .send()
        .await
        .unwrap();

    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn test_update_datapoint_duplicate_ids() {
    let http_client = Client::new();
    let dataset_name = format!("test-update-duplicate-{}", Uuid::now_v7());
    let datapoint_id = Uuid::now_v7();

    let resp = http_client
        .patch(get_gateway_endpoint(&format!(
            "/v1/datasets/{dataset_name}/datapoints",
        )))
        .json(&json!({
            "datapoints": [
                {
                    "type": "chat",
                    "id": datapoint_id.to_string(),
                    "output": [{"type": "text", "text": "Output 1"}],
                },
                {
                    "type": "chat",
                    "id": datapoint_id.to_string(),
                    "output": [{"type": "text", "text": "Output 2"}],
                }
            ]
        }))
        .send()
        .await
        .unwrap();

    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn test_update_chat_datapoint_set_output_to_null() {
    let http_client = Client::new();
    let clickhouse = get_clickhouse().await;
    let dataset_name = format!("test-update-chat-output-null-{}", Uuid::now_v7());

    // Create a datapoint with output
    let datapoint_id = Uuid::now_v7();

    let datapoint_insert = DatapointInsert::Chat(ChatInferenceDatapointInsert {
        dataset_name: dataset_name.clone(),
        function_name: "basic_test".to_string(),
        name: None,
        id: datapoint_id,
        episode_id: None,
        input: StoredInput {
            system: Some(System::Template(Arguments(
                json!({"assistant_name": "Test"})
                    .as_object()
                    .unwrap()
                    .clone(),
            ))),
            messages: vec![StoredInputMessage {
                role: Role::User,
                content: vec![StoredInputMessageContent::Text(Text {
                    text: "Test message".to_string(),
                })],
            }],
        },
        output: Some(vec![ContentBlockChatOutput::Text(Text {
            text: "Original output".to_string(),
        })]),
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

    // Update to set output to empty
    let resp = http_client
        .patch(get_gateway_endpoint(&format!(
            "/v1/datasets/{dataset_name}/datapoints",
        )))
        .json(&json!({
            "datapoints": [{
                "type": "chat",
                "id": datapoint_id.to_string(),
                "output": [],
            }]
        }))
        .send()
        .await
        .unwrap();

    assert!(resp.status().is_success());
    let resp_json: Value = resp.json().await.unwrap();
    let new_id: Uuid = resp_json["ids"][0].as_str().unwrap().parse().unwrap();

    tokio::time::sleep(Duration::from_millis(500)).await;

    // Verify the new datapoint has output set to None
    let new_datapoint = clickhouse
        .get_datapoint(&GetDatapointParams {
            dataset_name: dataset_name.clone(),
            datapoint_id: new_id,
            allow_stale: Some(false),
        })
        .await
        .unwrap();

    let StoredDatapoint::Chat(chat_datapoint) = new_datapoint else {
        panic!("Expected chat datapoint");
    };
    assert!(chat_datapoint.staled_at.is_none());
    assert_eq!(chat_datapoint.output, Some(vec![]));
}

#[tokio::test]
async fn test_update_chat_datapoint_set_tool_params_to_null() {
    let http_client = Client::new();
    let clickhouse = get_clickhouse().await;
    let dataset_name = format!("test-update-chat-tool-params-null-{}", Uuid::now_v7());

    // Create a datapoint with tool_params
    let datapoint_id = Uuid::now_v7();

    let datapoint_insert = DatapointInsert::Chat(ChatInferenceDatapointInsert {
        dataset_name: dataset_name.clone(),
        function_name: "basic_test".to_string(),
        name: None,
        id: datapoint_id,
        episode_id: None,
        input: StoredInput {
            system: Some(System::Template(Arguments(
                json!({"assistant_name": "Test"})
                    .as_object()
                    .unwrap()
                    .clone(),
            ))),
            messages: vec![StoredInputMessage {
                role: Role::User,
                content: vec![StoredInputMessageContent::Text(Text {
                    text: "Test message".to_string(),
                })],
            }],
        },
        output: Some(vec![ContentBlockChatOutput::Text(Text {
            text: "Output".to_string(),
        })]),
        tool_params: Some(ToolCallConfigDatabaseInsert::new_for_test(
            vec![Tool::Function(FunctionTool {
                name: "test_tool".to_string(),
                description: "Test tool".to_string(),
                parameters: json!({}),
                strict: false,
            })],
            vec![ProviderTool {
                scope: ProviderToolScope::Unscoped,
                tool: json!({"foo": "bar"}),
            }],
            AllowedTools {
                tools: vec!["test_tool".to_string()],
                choice: AllowedToolsChoice::Explicit,
            },
            ToolChoice::Auto,
            None,
        )),
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

    // Update to set tool_params fields to null
    let resp = http_client
        .patch(get_gateway_endpoint(&format!(
            "/v1/datasets/{dataset_name}/datapoints",
        )))
        .json(&json!({
            "datapoints": [{
                "type": "chat",
                "id": datapoint_id.to_string(),
                "allowed_tools": null,
                "additional_tools": [],
                "provider_tools": [],
            }]
        }))
        .send()
        .await
        .unwrap();

    assert!(resp.status().is_success());
    let resp_json: Value = resp.json().await.unwrap();
    let new_id: Uuid = resp_json["ids"][0].as_str().unwrap().parse().unwrap();

    tokio::time::sleep(Duration::from_millis(500)).await;

    // Verify the new datapoint has tool_params fields cleared appropriately
    let new_datapoint = clickhouse
        .get_datapoint(&GetDatapointParams {
            dataset_name: dataset_name.clone(),
            datapoint_id: new_id,
            allow_stale: Some(false),
        })
        .await
        .unwrap();

    let StoredDatapoint::Chat(chat_datapoint) = new_datapoint else {
        panic!("Expected chat datapoint");
    };
    assert!(chat_datapoint.staled_at.is_none());
    assert!(
        chat_datapoint.tool_params.is_none(),
        "When all tool params are cleared, tool params should be None (through deserialize_optional_tool_info)"
    );
}

#[tokio::test]
async fn test_update_chat_datapoint_set_tags_to_empty() {
    let http_client = Client::new();
    let clickhouse = get_clickhouse().await;
    let dataset_name = format!("test-update-chat-tags-null-{}", Uuid::now_v7());

    // Create a datapoint with tags
    let datapoint_id = Uuid::now_v7();
    let mut tags = HashMap::new();
    tags.insert("key1".to_string(), "value1".to_string());
    tags.insert("key2".to_string(), "value2".to_string());

    let datapoint_insert = DatapointInsert::Chat(ChatInferenceDatapointInsert {
        dataset_name: dataset_name.clone(),
        function_name: "basic_test".to_string(),
        name: None,
        id: datapoint_id,
        episode_id: None,
        input: StoredInput {
            system: Some(System::Template(Arguments(
                json!({"assistant_name": "Test"})
                    .as_object()
                    .unwrap()
                    .clone(),
            ))),
            messages: vec![StoredInputMessage {
                role: Role::User,
                content: vec![StoredInputMessageContent::Text(Text {
                    text: "Test message".to_string(),
                })],
            }],
        },
        output: Some(vec![ContentBlockChatOutput::Text(Text {
            text: "Output".to_string(),
        })]),
        tool_params: None,
        tags: Some(tags),
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

    // Update to set tags to empty
    let resp = http_client
        .patch(get_gateway_endpoint(&format!(
            "/v1/datasets/{dataset_name}/datapoints",
        )))
        .json(&json!({
            "datapoints": [{
                "type": "chat",
                "id": datapoint_id.to_string(),
                "tags": {},
            }]
        }))
        .send()
        .await
        .unwrap();

    assert!(resp.status().is_success());
    let resp_json: Value = resp.json().await.unwrap();
    let new_id: Uuid = resp_json["ids"][0].as_str().unwrap().parse().unwrap();

    tokio::time::sleep(Duration::from_millis(500)).await;

    // Verify the new datapoint has tags set to None
    let new_datapoint = clickhouse
        .get_datapoint(&GetDatapointParams {
            dataset_name: dataset_name.clone(),
            datapoint_id: new_id,
            allow_stale: Some(false),
        })
        .await
        .unwrap();

    let StoredDatapoint::Chat(chat_datapoint) = new_datapoint else {
        panic!("Expected chat datapoint");
    };
    assert!(chat_datapoint.staled_at.is_none());
    assert_eq!(chat_datapoint.tags, Some(HashMap::new()));
}

#[tokio::test]
async fn test_update_chat_datapoint_set_name_to_null() {
    let http_client = Client::new();
    let clickhouse = get_clickhouse().await;
    let dataset_name = format!("test-update-chat-name-null-{}", Uuid::now_v7());

    // Create a datapoint with a name
    let datapoint_id = Uuid::now_v7();

    let datapoint_insert = DatapointInsert::Chat(ChatInferenceDatapointInsert {
        dataset_name: dataset_name.clone(),
        function_name: "basic_test".to_string(),
        name: Some("Original Name".to_string()),
        id: datapoint_id,
        episode_id: None,
        input: StoredInput {
            system: Some(System::Template(Arguments(
                json!({"assistant_name": "Test"})
                    .as_object()
                    .unwrap()
                    .clone(),
            ))),
            messages: vec![StoredInputMessage {
                role: Role::User,
                content: vec![StoredInputMessageContent::Text(Text {
                    text: "Test message".to_string(),
                })],
            }],
        },
        output: Some(vec![ContentBlockChatOutput::Text(Text {
            text: "Output".to_string(),
        })]),
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

    // Update to set name to null
    let resp = http_client
        .patch(get_gateway_endpoint(&format!(
            "/v1/datasets/{dataset_name}/datapoints",
        )))
        .json(&json!({
            "datapoints": [{
                "type": "chat",
                "id": datapoint_id.to_string(),
                "name": null
            }]
        }))
        .send()
        .await
        .unwrap();

    assert!(resp.status().is_success());
    let resp_json: Value = resp.json().await.unwrap();
    let new_id: Uuid = resp_json["ids"][0].as_str().unwrap().parse().unwrap();

    tokio::time::sleep(Duration::from_millis(500)).await;

    // Verify the new datapoint has name set to None
    let new_datapoint = clickhouse
        .get_datapoint(&GetDatapointParams {
            dataset_name: dataset_name.clone(),
            datapoint_id: new_id,
            allow_stale: Some(false),
        })
        .await
        .unwrap();

    let StoredDatapoint::Chat(chat_datapoint) = new_datapoint else {
        panic!("Expected chat datapoint");
    };
    assert!(chat_datapoint.staled_at.is_none());
    assert_eq!(chat_datapoint.name, None);
}

#[tokio::test]
async fn test_update_json_datapoint_set_output_to_null() {
    let http_client = Client::new();
    let clickhouse = get_clickhouse().await;
    let dataset_name = format!("test-update-json-output-null-{}", Uuid::now_v7());

    // Create a JSON datapoint with output
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
                json!({"assistant_name": "Test"})
                    .as_object()
                    .unwrap()
                    .clone(),
            ))),
            messages: vec![StoredInputMessage {
                role: Role::User,
                content: vec![StoredInputMessageContent::Text(Text {
                    text: json!({"question": "test"}).to_string(),
                })],
            }],
        },
        output: Some(JsonInferenceOutput {
            raw: Some(r#"{"answer":"original"}"#.to_string()),
            parsed: Some(json!({"answer": "original"})),
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

    // Update to set output to null
    let resp = http_client
        .patch(get_gateway_endpoint(&format!(
            "/v1/datasets/{dataset_name}/datapoints",
        )))
        .json(&json!({
            "datapoints": [{
                "type": "json",
                "id": datapoint_id.to_string(),
                "output": null,
            }]
        }))
        .send()
        .await
        .unwrap();

    assert!(resp.status().is_success());
    let resp_json: Value = resp.json().await.unwrap();
    let new_id: Uuid = resp_json["ids"][0].as_str().unwrap().parse().unwrap();

    tokio::time::sleep(Duration::from_millis(500)).await;

    // Verify the new datapoint has output set to None
    let new_datapoint = clickhouse
        .get_datapoint(&GetDatapointParams {
            dataset_name: dataset_name.clone(),
            datapoint_id: new_id,
            allow_stale: Some(false),
        })
        .await
        .unwrap();

    let StoredDatapoint::Json(json_datapoint) = new_datapoint else {
        panic!("Expected json datapoint");
    };
    assert!(json_datapoint.staled_at.is_none());
    assert_eq!(json_datapoint.output, None);
}

#[tokio::test]
async fn test_update_json_datapoint_set_tags_to_empty() {
    let http_client = Client::new();
    let clickhouse = get_clickhouse().await;
    let dataset_name = format!("test-update-json-tags-null-{}", Uuid::now_v7());

    // Create a JSON datapoint with tags
    let datapoint_id = Uuid::now_v7();
    let mut tags = HashMap::new();
    tags.insert("key1".to_string(), "value1".to_string());
    tags.insert("key2".to_string(), "value2".to_string());

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
                json!({"assistant_name": "Test"})
                    .as_object()
                    .unwrap()
                    .clone(),
            ))),
            messages: vec![StoredInputMessage {
                role: Role::User,
                content: vec![StoredInputMessageContent::Text(Text {
                    text: json!({"question": "test"}).to_string(),
                })],
            }],
        },
        output: Some(JsonInferenceOutput {
            raw: Some(r#"{"answer":"original"}"#.to_string()),
            parsed: Some(json!({"answer": "original"})),
        }),
        output_schema,
        tags: Some(tags),
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

    // Update to set tags to null
    let resp = http_client
        .patch(get_gateway_endpoint(&format!(
            "/v1/datasets/{dataset_name}/datapoints",
        )))
        .json(&json!({
            "datapoints": [{
                "type": "json",
                "id": datapoint_id.to_string(),
                "tags": {},
            }]
        }))
        .send()
        .await
        .unwrap();

    assert!(resp.status().is_success());
    let resp_json: Value = resp.json().await.unwrap();
    let new_id: Uuid = resp_json["ids"][0].as_str().unwrap().parse().unwrap();

    tokio::time::sleep(Duration::from_millis(500)).await;

    // Verify the new datapoint has tags set to None
    let new_datapoint = clickhouse
        .get_datapoint(&GetDatapointParams {
            dataset_name: dataset_name.clone(),
            datapoint_id: new_id,
            allow_stale: Some(false),
        })
        .await
        .unwrap();

    let StoredDatapoint::Json(json_datapoint) = new_datapoint else {
        panic!("Expected json datapoint");
    };
    assert!(json_datapoint.staled_at.is_none());
    assert_eq!(json_datapoint.tags, Some(HashMap::new()));
}

// ============================================================================
// Tests for update_datapoints_metadata endpoint
// ============================================================================

#[tokio::test]
async fn test_update_metadata_chat_datapoint() {
    let http_client = Client::new();
    let clickhouse = get_clickhouse().await;
    let dataset_name = format!("test-update-metadata-chat-{}", Uuid::now_v7());

    // Create a chat datapoint
    let datapoint_id = Uuid::now_v7();
    let datapoint_insert = DatapointInsert::Chat(ChatInferenceDatapointInsert {
        dataset_name: dataset_name.clone(),
        function_name: "basic_test".to_string(),
        name: Some("original_name".to_string()),
        id: datapoint_id,
        episode_id: None,
        input: StoredInput {
            system: None,
            messages: vec![StoredInputMessage {
                role: Role::User,
                content: vec![StoredInputMessageContent::Text(Text {
                    text: "Test message".to_string(),
                })],
            }],
        },
        output: Some(vec![ContentBlockChatOutput::Text(Text {
            text: "Test output".to_string(),
        })]),
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

    // Update metadata
    let resp = http_client
        .patch(get_gateway_endpoint(&format!(
            "/v1/datasets/{dataset_name}/datapoints/metadata",
        )))
        .json(&json!({
            "datapoints": [{
                "id": datapoint_id.to_string(),
                "name": "updated_name"
            }]
        }))
        .send()
        .await
        .unwrap();

    assert!(
        resp.status().is_success(),
        "Update metadata request failed: {:?}",
        resp.status()
    );
    let resp_json: Value = resp.json().await.unwrap();
    let returned_ids = resp_json["ids"].as_array().unwrap();
    assert_eq!(returned_ids.len(), 1);
    let returned_id: Uuid = returned_ids[0].as_str().unwrap().parse().unwrap();
    assert_eq!(
        returned_id, datapoint_id,
        "Should return the same ID, not create a new one"
    );

    // Wait for async inserts
    tokio::time::sleep(Duration::from_millis(1000)).await;
    clickhouse_flush_async_insert(&clickhouse).await;
    tokio::time::sleep(Duration::from_millis(1000)).await;

    // Verify the datapoint has updated name
    let updated_datapoint = clickhouse
        .get_datapoint(&GetDatapointParams {
            dataset_name: dataset_name.clone(),
            datapoint_id,
            allow_stale: Some(false),
        })
        .await
        .unwrap();

    let StoredDatapoint::Chat(chat_datapoint) = updated_datapoint else {
        panic!("Expected chat datapoint");
    };
    assert_eq!(chat_datapoint.name, Some("updated_name".to_string()));
    assert!(chat_datapoint.staled_at.is_none(), "Should not be staled");
}

#[tokio::test]
async fn test_update_metadata_json_datapoint() {
    let http_client = Client::new();
    let clickhouse = get_clickhouse().await;
    let dataset_name = format!("test-update-metadata-json-{}", Uuid::now_v7());

    // Create a JSON datapoint
    let datapoint_id = Uuid::now_v7();
    let output_schema = json!({
        "type": "object",
        "properties": {"result": {"type": "string"}},
        "required": ["result"],
        "additionalProperties": false
    });

    let datapoint_insert = DatapointInsert::Json(JsonInferenceDatapointInsert {
        dataset_name: dataset_name.clone(),
        function_name: "json_success".to_string(),
        name: Some("original_json_name".to_string()),
        id: datapoint_id,
        episode_id: None,
        input: StoredInput {
            system: None,
            messages: vec![StoredInputMessage {
                role: Role::User,
                content: vec![StoredInputMessageContent::Text(Text {
                    text: json!({"query": "test"}).to_string(),
                })],
            }],
        },
        output: Some(JsonInferenceOutput {
            raw: Some(r#"{"result":"test"}"#.to_string()),
            parsed: Some(json!({"result": "test"})),
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

    // Update metadata
    let resp = http_client
        .patch(get_gateway_endpoint(&format!(
            "/v1/datasets/{dataset_name}/datapoints/metadata",
        )))
        .json(&json!({
            "datapoints": [{
                "id": datapoint_id.to_string(),
                "name": "updated_json_name"
            }]
        }))
        .send()
        .await
        .unwrap();

    assert!(resp.status().is_success());
    let resp_json: Value = resp.json().await.unwrap();
    let returned_ids = resp_json["ids"].as_array().unwrap();
    assert_eq!(returned_ids.len(), 1);
    let returned_id: Uuid = returned_ids[0].as_str().unwrap().parse().unwrap();
    assert_eq!(returned_id, datapoint_id);

    // Wait for async inserts
    tokio::time::sleep(Duration::from_millis(1000)).await;
    clickhouse_flush_async_insert(&clickhouse).await;
    tokio::time::sleep(Duration::from_millis(1000)).await;

    // Verify the datapoint has updated name
    let updated_datapoint = clickhouse
        .get_datapoint(&GetDatapointParams {
            dataset_name: dataset_name.clone(),
            datapoint_id,
            allow_stale: Some(false),
        })
        .await
        .unwrap();

    let StoredDatapoint::Json(json_datapoint) = updated_datapoint else {
        panic!("Expected json datapoint");
    };
    assert_eq!(json_datapoint.name, Some("updated_json_name".to_string()));
    assert!(json_datapoint.staled_at.is_none());
}

#[tokio::test]
async fn test_update_metadata_set_name_to_null() {
    let http_client = Client::new();
    let clickhouse = get_clickhouse().await;
    let dataset_name = format!("test-update-metadata-null-{}", Uuid::now_v7());

    // Create a chat datapoint with a name
    let datapoint_id = Uuid::now_v7();
    let datapoint_insert = DatapointInsert::Chat(ChatInferenceDatapointInsert {
        dataset_name: dataset_name.clone(),
        function_name: "basic_test".to_string(),
        name: Some("to_be_removed".to_string()),
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
        output: None,
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

    // Set name to null
    let resp = http_client
        .patch(get_gateway_endpoint(&format!(
            "/v1/datasets/{dataset_name}/datapoints/metadata",
        )))
        .json(&json!({
            "datapoints": [{
                "id": datapoint_id.to_string(),
                "name": null,
            }]
        }))
        .send()
        .await
        .unwrap();

    assert!(resp.status().is_success());

    // Wait for async inserts
    clickhouse_flush_async_insert(&clickhouse).await;
    tokio::time::sleep(Duration::from_millis(1000)).await;

    // Verify name is now null
    let updated_datapoint = clickhouse
        .get_datapoint(&GetDatapointParams {
            dataset_name: dataset_name.clone(),
            datapoint_id,
            allow_stale: Some(false),
        })
        .await
        .unwrap();

    let StoredDatapoint::Chat(chat_datapoint) = updated_datapoint else {
        panic!("Expected chat datapoint");
    };
    assert_eq!(chat_datapoint.name, None);
}

#[tokio::test]
async fn test_update_metadata_batch() {
    let http_client = Client::new();
    let clickhouse = get_clickhouse().await;
    let dataset_name = format!("test-update-metadata-batch-{}", Uuid::now_v7());

    // Create multiple datapoints
    let datapoint_id1 = Uuid::now_v7();
    let datapoint_id2 = Uuid::now_v7();

    let datapoint1 = DatapointInsert::Chat(ChatInferenceDatapointInsert {
        dataset_name: dataset_name.clone(),
        function_name: "basic_test".to_string(),
        name: Some("name1".to_string()),
        id: datapoint_id1,
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
        output: None,
        tool_params: None,
        tags: None,
        auxiliary: String::new(),
        staled_at: None,
        source_inference_id: None,
        is_custom: true,
    });

    let datapoint2 = DatapointInsert::Chat(ChatInferenceDatapointInsert {
        dataset_name: dataset_name.clone(),
        function_name: "basic_test".to_string(),
        name: Some("name2".to_string()),
        id: datapoint_id2,
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
        output: None,
        tool_params: None,
        tags: None,
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

    // Update both datapoints' metadata
    let resp = http_client
        .patch(get_gateway_endpoint(&format!(
            "/v1/datasets/{dataset_name}/datapoints/metadata",
        )))
        .json(&json!({
            "datapoints": [
                {
                    "id": datapoint_id1.to_string(),
                    "metadata": {"name": "updated_name1"}
                },
                {
                    "id": datapoint_id2.to_string(),
                    "metadata": {"name": "updated_name2"}
                }
            ]
        }))
        .send()
        .await
        .unwrap();

    assert!(resp.status().is_success());
    let resp_json: Value = resp.json().await.unwrap();
    let returned_ids = resp_json["ids"].as_array().unwrap();
    assert_eq!(returned_ids.len(), 2);

    // Wait for async inserts
    clickhouse_flush_async_insert(&clickhouse).await;
    tokio::time::sleep(Duration::from_millis(1000)).await;

    // Verify both datapoints have updated names
    let datapoint1 = clickhouse
        .get_datapoint(&GetDatapointParams {
            dataset_name: dataset_name.clone(),
            datapoint_id: datapoint_id1,
            allow_stale: Some(false),
        })
        .await
        .unwrap();

    let StoredDatapoint::Chat(chat_datapoint) = datapoint1 else {
        panic!("Expected chat datapoint");
    };
    assert_eq!(chat_datapoint.name, Some("updated_name1".to_string()));

    let datapoint2 = clickhouse
        .get_datapoint(&GetDatapointParams {
            dataset_name: dataset_name.clone(),
            datapoint_id: datapoint_id2,
            allow_stale: Some(false),
        })
        .await
        .unwrap();

    let StoredDatapoint::Chat(chat_datapoint) = datapoint2 else {
        panic!("Expected chat datapoint");
    };
    assert_eq!(chat_datapoint.name, Some("updated_name2".to_string()));
}

#[tokio::test]
async fn test_update_metadata_datapoint_not_found() {
    let http_client = Client::new();
    let dataset_name = format!("test-update-metadata-notfound-{}", Uuid::now_v7());
    let non_existent_id = Uuid::now_v7();

    let resp = http_client
        .patch(get_gateway_endpoint(&format!(
            "/v1/datasets/{dataset_name}/datapoints/metadata",
        )))
        .json(&json!({
            "datapoints": [{
                "id": non_existent_id.to_string(),
                "metadata": {"name": "new_name"}
            }]
        }))
        .send()
        .await
        .unwrap();

    assert_eq!(resp.status(), StatusCode::NOT_FOUND);
}

#[tokio::test]
async fn test_update_metadata_duplicate_ids() {
    let http_client = Client::new();
    let dataset_name = format!("test-update-metadata-duplicate-{}", Uuid::now_v7());
    let duplicate_id = Uuid::now_v7();

    let resp = http_client
        .patch(get_gateway_endpoint(&format!(
            "/v1/datasets/{dataset_name}/datapoints/metadata",
        )))
        .json(&json!({
            "datapoints": [
                {
                    "id": duplicate_id.to_string(),
                    "metadata": {"name": "name1"}
                },
                {
                    "id": duplicate_id.to_string(),
                    "metadata": {"name": "name2"}
                }
            ]
        }))
        .send()
        .await
        .unwrap();

    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn test_get_chat_datapoint_modify_and_update_roundtrip() {
    let http_client = Client::new();
    let clickhouse = get_clickhouse().await;
    let dataset_name = format!("test-roundtrip-{}", Uuid::now_v7());

    // Create a datapoint
    let datapoint_id = Uuid::now_v7();
    let mut tags = HashMap::new();
    tags.insert("env".to_string(), "test".to_string());

    let datapoint_insert = DatapointInsert::Chat(ChatInferenceDatapointInsert {
        dataset_name: dataset_name.clone(),
        function_name: "basic_test".to_string(),
        name: Some("Original Name".to_string()),
        id: datapoint_id,
        episode_id: None,
        input: StoredInput {
            system: Some(System::Template(Arguments(
                json!({"assistant_name": "Test"})
                    .as_object()
                    .unwrap()
                    .clone(),
            ))),
            messages: vec![StoredInputMessage {
                role: Role::User,
                content: vec![StoredInputMessageContent::Text(Text {
                    text: "Original message".to_string(),
                })],
            }],
        },
        output: Some(vec![ContentBlockChatOutput::Text(Text {
            text: "Original output".to_string(),
        })]),
        tool_params: None,
        tags: Some(tags),
        auxiliary: String::new(),
        staled_at: None,
        source_inference_id: None,
        // Make a non-custom datapoint so that it can be modified
        is_custom: false,
    });

    clickhouse
        .insert_datapoints(&[datapoint_insert])
        .await
        .unwrap();

    tokio::time::sleep(Duration::from_millis(500)).await;

    // Get the datapoint via HTTP API
    let get_resp = http_client
        .post(get_gateway_endpoint("/v1/datasets/get_datapoints"))
        .json(&json!({
            "ids": [datapoint_id.to_string()]
        }))
        .send()
        .await
        .unwrap();

    assert!(get_resp.status().is_success());
    let get_datapoints_response: Value = get_resp.json().await.unwrap();

    // Modify the output field in the JSON
    assert!(
        get_datapoints_response["datapoints"]
            .as_array()
            .unwrap()
            .len()
            == 1
    );
    let mut first_datapoint = get_datapoints_response["datapoints"][0].clone();
    first_datapoint["output"][0]["text"] = json!("Modified output");

    // Round-trip to update_datapoints
    let update_resp = http_client
        .patch(get_gateway_endpoint(&format!(
            "/v1/datasets/{dataset_name}/datapoints",
        )))
        .json(&json!({
            "datapoints": [first_datapoint]
        }))
        .send()
        .await
        .unwrap();

    assert!(
        update_resp.status().is_success(),
        "Update request failed: {:?}",
        update_resp.status()
    );
    let resp_json: Value = update_resp.json().await.unwrap();
    let new_id: Uuid = resp_json["ids"][0].as_str().unwrap().parse().unwrap();
    assert_ne!(new_id, datapoint_id, "Should create a new datapoint ID");

    // Wait for async inserts
    clickhouse_flush_async_insert(&clickhouse).await;
    tokio::time::sleep(Duration::from_millis(1000)).await;

    // Verify the new datapoint has the modified output
    let new_datapoint = clickhouse
        .get_datapoint(&GetDatapointParams {
            dataset_name: dataset_name.clone(),
            datapoint_id: new_id,
            allow_stale: Some(false),
        })
        .await
        .unwrap();

    let StoredDatapoint::Chat(chat_datapoint) = new_datapoint else {
        panic!("Expected chat datapoint");
    };
    assert!(chat_datapoint.staled_at.is_none());
    assert_eq!(
        chat_datapoint.output,
        Some(vec![ContentBlockChatOutput::Text(Text {
            text: "Modified output".to_string(),
        })]),
        "Output should be modified"
    );
    // Verify it's now marked as "custom"
    assert!(
        chat_datapoint.is_custom,
        "Datapoint should be marked as custom"
    );
    // Verify other fields remain unchanged
    assert_eq!(
        chat_datapoint.name,
        Some("Original Name".to_string()),
        "Name should remain unchanged"
    );
    assert_eq!(
        chat_datapoint.tags,
        Some(HashMap::from([("env".to_string(), "test".to_string())])),
        "Tags should remain unchanged"
    );
}

#[tokio::test]
async fn test_get_json_datapoint_modify_and_update_roundtrip() {
    let http_client = Client::new();
    let clickhouse = get_clickhouse().await;
    let dataset_name = format!("test-json-roundtrip-{}", Uuid::now_v7());

    // Create a JSON datapoint
    let datapoint_id = Uuid::now_v7();
    let mut tags = HashMap::new();
    tags.insert("env".to_string(), "test".to_string());

    let output_schema = json!({
        "type": "object",
        "properties": {"answer": {"type": "string"}},
        "required": ["answer"],
        "additionalProperties": false
    });

    let datapoint_insert = DatapointInsert::Json(JsonInferenceDatapointInsert {
        dataset_name: dataset_name.clone(),
        function_name: "json_success".to_string(),
        name: Some("Original Name".to_string()),
        id: datapoint_id,
        episode_id: None,
        input: StoredInput {
            system: Some(System::Template(Arguments(
                json!({"assistant_name": "Alfred Pennyworth"})
                    .as_object()
                    .unwrap()
                    .clone(),
            ))),
            messages: vec![StoredInputMessage {
                role: Role::User,
                content: vec![StoredInputMessageContent::Template(Template {
                    name: "user".to_string(),
                    arguments: Arguments(json!({"country": "Japan"}).as_object().unwrap().clone()),
                })],
            }],
        },
        output: Some(JsonInferenceOutput {
            raw: Some(r#"{"answer":"4"}"#.to_string()),
            parsed: Some(json!({"answer": "4"})),
        }),
        output_schema,
        tags: Some(tags),
        auxiliary: String::new(),
        staled_at: None,
        source_inference_id: None,
        // Mark as non-custom so that it can be modified
        is_custom: false,
    });

    clickhouse
        .insert_datapoints(&[datapoint_insert])
        .await
        .unwrap();

    tokio::time::sleep(Duration::from_millis(500)).await;

    // Get the datapoint via HTTP API
    let get_resp = http_client
        .post(get_gateway_endpoint("/v1/datasets/get_datapoints"))
        .json(&json!({
            "ids": [datapoint_id.to_string()]
        }))
        .send()
        .await
        .unwrap();

    assert!(get_resp.status().is_success());
    let get_datapoints_response: Value = get_resp.json().await.unwrap();

    // Modify the output field in the JSON
    assert!(
        get_datapoints_response["datapoints"]
            .as_array()
            .unwrap()
            .len()
            == 1
    );
    let mut first_datapoint = get_datapoints_response["datapoints"][0].clone();
    first_datapoint["output"] = json!({
        "raw": r#"{"answer":"5"}"#
    });
    first_datapoint["name"] = json!("Modified Name");

    // Send it back to update_datapoints
    let update_resp = http_client
        .patch(get_gateway_endpoint(&format!(
            "/v1/datasets/{dataset_name}/datapoints",
        )))
        .json(&json!({
            "datapoints": [first_datapoint]
        }))
        .send()
        .await
        .unwrap();

    assert!(
        update_resp.status().is_success(),
        "Update request failed: {:?}",
        update_resp.status()
    );
    let resp_json: Value = update_resp.json().await.unwrap();
    let new_id: Uuid = resp_json["ids"][0].as_str().unwrap().parse().unwrap();
    assert_ne!(new_id, datapoint_id, "Should create a new datapoint ID");

    // Wait for async inserts
    clickhouse_flush_async_insert(&clickhouse).await;
    tokio::time::sleep(Duration::from_millis(1000)).await;

    // Verify the new datapoint has the modified output and name
    let new_datapoint = clickhouse
        .get_datapoint(&GetDatapointParams {
            dataset_name: dataset_name.clone(),
            datapoint_id: new_id,
            allow_stale: Some(false),
        })
        .await
        .unwrap();

    let StoredDatapoint::Json(json_datapoint) = new_datapoint else {
        panic!("Expected json datapoint");
    };
    assert!(json_datapoint.staled_at.is_none());

    let output = json_datapoint.output.unwrap();
    assert_eq!(output.parsed.unwrap(), json!({"answer": "5"}));

    // Verify other fields remain unchanged
    assert_eq!(json_datapoint.name, Some("Modified Name".to_string()));
    assert_eq!(
        json_datapoint.tags,
        Some(HashMap::from([("env".to_string(), "test".to_string())]))
    );
}
