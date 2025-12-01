#![expect(clippy::print_stdout)]

use std::time::Duration;

use reqwest::{Client, StatusCode};
use serde_json::{json, Value};
use tensorzero::{
    ClientExt, InputMessageContent, JsonInferenceDatapoint, Role, StoredDatapoint, System,
};
use tensorzero_core::endpoints::datasets::ChatInferenceDatapoint;
use tensorzero_core::{
    db::{
        clickhouse::test_helpers::{
            select_chat_dataset_clickhouse, select_json_dataset_clickhouse,
        },
        datasets::{DatasetQueries, GetDatapointsParams},
    },
    endpoints::datasets::{DatapointKind, CLICKHOUSE_DATETIME_FORMAT},
    inference::types::{ContentBlockChatOutput, StoredInputMessageContent},
    tool::Tool,
};

use uuid::Uuid;

use crate::common::{delete_datapoint, get_gateway_endpoint};
use tensorzero_core::db::clickhouse::test_helpers::{
    get_clickhouse, select_chat_datapoint_clickhouse, select_json_datapoint_clickhouse,
};

#[tokio::test]
async fn test_datapoint_insert_synthetic_chat() {
    let clickhouse = get_clickhouse().await;
    let client = Client::new();
    let dataset_name = format!("test-dataset-{}", Uuid::now_v7());
    let source_inference_id = Uuid::now_v7();

    let datapoint_id = Uuid::now_v7();

    let resp = client
        .put(get_gateway_endpoint(&format!(
            "/internal/datasets/{dataset_name}/datapoints/{datapoint_id}",
        )))
        .json(&json!({
            "function_name": "basic_test",
            "input": {"system": {"assistant_name": "Dummy"}, "messages": [{"role": "user", "content": [{"type": "text", "text": "My synthetic input"}]}]},
            "output": [{"type": "text", "text": "My synthetic output"}],
            "source_inference_id": source_inference_id,
            "is_custom": true,
        }))
        .send()
        .await
        .unwrap();

    let status = resp.status();
    let resp_json = resp.json::<Value>().await.unwrap();

    assert!(status.is_success(), "Bad request: {resp_json:?}");

    let id: Uuid = resp_json
        .get("id")
        .unwrap()
        .as_str()
        .unwrap()
        .parse()
        .unwrap();

    assert_eq!(id, datapoint_id);

    let mut datapoint = select_chat_datapoint_clickhouse(&clickhouse, id)
        .await
        .unwrap();

    let updated_at = datapoint
        .as_object_mut()
        .unwrap()
        .remove("updated_at")
        .unwrap();
    let updated_at = chrono::NaiveDateTime::parse_from_str(
        updated_at.as_str().unwrap(),
        CLICKHOUSE_DATETIME_FORMAT,
    )
    .unwrap()
    .and_utc();
    assert!(
        chrono::Utc::now()
            .signed_duration_since(updated_at)
            .num_seconds()
            < 5,
        "Unexpected updated_at: {updated_at:?}"
    );

    let expected = json!({
      "dataset_name": dataset_name,
      "function_name": "basic_test",
      "id": id.to_string(),
      "name": null,
      "episode_id": null,
      "input": "{\"system\":{\"assistant_name\":\"Dummy\"},\"messages\":[{\"role\":\"user\",\"content\":[{\"type\":\"text\",\"text\":\"My synthetic input\"}]}]}",
      "output": "[{\"type\":\"text\",\"text\":\"My synthetic output\"}]",
      "tool_params": "{\"tools_available\":[],\"tool_choice\":\"auto\",\"parallel_tool_calls\":null}",
      "dynamic_tools": [],
      "dynamic_provider_tools": [],
      "tool_choice": "auto",
      "parallel_tool_calls": null,
      "allowed_tools": "{\"tools\":[],\"choice\":\"function_default\"}",
      "tags": {},
      "auxiliary": "",
      "is_deleted": false,
      "is_custom": true,
      "source_inference_id": source_inference_id.to_string(),
      "staled_at": null,
    });
    assert_eq!(datapoint, expected);
}

#[tokio::test]
async fn test_create_delete_datapoint_chat() {
    let clickhouse = get_clickhouse().await;
    let client = Client::new();
    let dataset_name = format!("test-dataset-{}", Uuid::now_v7());
    println!("dataset_name: {dataset_name}");

    let additional_tools = json!([{"description": "Get the current temperature in a given location",
                "parameters": {
                    "$schema": "http://json-schema.org/draft-07/schema#",
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The location to get the temperature for (e.g. \"New York\")"
                        },
                        "units": {
                            "type": "string",
                            "description": "The units to get the temperature in (must be \"fahrenheit\" or \"celsius\")",
                            "enum": ["fahrenheit", "celsius"]
                        }
                    },
                    "required": ["location"],
                    "additionalProperties": false
                },
                "name": "get_temperature",
                "strict": false}]);

    let resp = client
        .post(get_gateway_endpoint(&format!(
            "/datasets/{dataset_name}/datapoints",
        )))
        .json(&json!({
            "datapoints": [
                {
                    "function_name": "basic_test",
                    "input": {
                        "system": { "assistant_name": "foo" },
                        "messages": [
                            {
                                "role": "user",
                                "content": [ { "type": "text", "text": "bar" } ]
                            }
                        ]
                    },
                    "output": [ { "type": "text", "text": "foobar" } ],
                    "additional_tools": null,
                    "tool_choice":   null,
                },
                {
                    "function_name": "basic_test",
                    "input": {
                        "system": { "assistant_name": "Dummy" },
                        "messages": [
                            {
                                "role": "user",
                                "content": [ { "type": "text", "text": "My synthetic input" } ]
                            }
                        ]
                    },
                    "output": [
                        {
                            "type": "tool_call",
                            "id": "call_123",
                            "name": "get_temperature",
                            "arguments": { "location": "New York", "units": "fahrenheit" }
                        }
                    ],
                    "additional_tools": additional_tools,
                    "tool_choice":  "auto"
                },
                {
                    "function_name": "basic_test",
                    "input": {
                        "system": { "assistant_name": "bar" },
                        "messages": [
                            {
                                "role": "user",
                                "content": [ { "type": "text", "text": "foo" } ]
                            }
                        ]
                    },
                    "output": null,
                    "additional_tools": null,
                    "tool_choice":  null
                }
            ]
        }))
        .send()
        .await
        .unwrap();

    let status = resp.status();

    assert_eq!(status, StatusCode::OK);
    tokio::time::sleep(Duration::from_millis(500)).await;

    let datapoints = select_chat_dataset_clickhouse(&clickhouse, &dataset_name)
        .await
        .unwrap();
    assert_eq!(datapoints.len(), 3);

    // Test the getter
    let list_datapoints_response = client
        .get(get_gateway_endpoint(&format!(
            "/datasets/{dataset_name}/datapoints",
        )))
        .send()
        .await
        .unwrap();
    assert!(list_datapoints_response.status().is_success());
    let list_datapoints_json = list_datapoints_response.json::<Vec<Value>>().await.unwrap();
    // Test that the auxiliary field is not returned by the list datapoints API
    for datapoint in &list_datapoints_json {
        assert!(datapoint.get("auxiliary").is_none());
    }
    let list_datapoints = list_datapoints_json
        .into_iter()
        .map(|datapoint| serde_json::from_value::<ChatInferenceDatapoint>(datapoint).unwrap())
        .collect::<Vec<_>>();
    assert_eq!(list_datapoints.len(), 3);

    for datapoint in &datapoints {
        let pretty_datapoint = serde_json::to_string_pretty(&datapoint).unwrap();
        println!("pretty_datapoint: {pretty_datapoint}");
        // Verify the datapoint structure and content
        assert_eq!(datapoint.dataset_name, dataset_name);
        assert_eq!(datapoint.function_name, "basic_test");
        assert!(!datapoint.is_deleted);
        assert!(datapoint.episode_id.is_none());
        assert!(datapoint.tags.as_ref().unwrap().is_empty());
        assert_eq!(datapoint.auxiliary, "");
        assert!(datapoint.staled_at.is_none());
        let datapoint_id = datapoint.id;

        // Find the matching list_datapoint by ID
        let list_datapoint = list_datapoints
            .iter()
            .find(|dp| dp.id == datapoint_id)
            .expect("datapoint from database should be in list response");

        // Test the getter
        let get_datapoint_response = client
            .get(get_gateway_endpoint(&format!(
                "/datasets/{dataset_name}/datapoints/{datapoint_id}",
            )))
            .send()
            .await
            .unwrap();
        assert!(get_datapoint_response.status().is_success());
        let get_datapoint_json = get_datapoint_response.json::<Value>().await.unwrap();
        // Assert that the auxiliary field is not returned by the get datapoint API
        assert!(get_datapoint_json.get("auxiliary").is_none());
        let get_datapoint =
            serde_json::from_value::<ChatInferenceDatapoint>(get_datapoint_json).unwrap();
        assert_eq!(&get_datapoint, list_datapoint);

        // Verify the list datapoint structure and content
        assert_eq!(list_datapoint.dataset_name, dataset_name);
        assert_eq!(list_datapoint.function_name, "basic_test");
        assert!(!list_datapoint.is_deleted);
        assert!(list_datapoint.episode_id.is_none());
        assert!(list_datapoint.tags.as_ref().unwrap().is_empty());
        assert_eq!(list_datapoint.auxiliary, "");

        // Verify input structure
        let input = &datapoint.input;
        assert!(match input.system.as_ref().unwrap() {
            System::Template(arguments) => arguments.0.get("assistant_name"),
            System::Text(_) => panic!("Expected System::Template"),
        }
        .is_some());
        assert!(!input.messages.is_empty());
        let first_message = input.messages[0].clone();
        assert_eq!(first_message.role, Role::User);
        let content = first_message.content;
        assert!(!content.is_empty());
        let first_content = content[0].clone();
        assert!(matches!(first_content, StoredInputMessageContent::Text(_)));

        // Verify the list datapoint input structure and content
        let input = &list_datapoint.input;
        assert!(match input.system.as_ref().unwrap() {
            System::Template(arguments) => arguments.0.get("assistant_name"),
            System::Text(_) => panic!("Expected System::Template"),
        }
        .is_some());
        assert!(!input.messages.is_empty());
        let first_message = input.messages[0].clone();
        assert_eq!(first_message.role, Role::User);
        let content = first_message.content;
        assert!(!content.is_empty());
        let first_content = content[0].clone();
        assert!(matches!(first_content, InputMessageContent::Text(_)));

        // Verify output if present
        if let Some(output) = &datapoint.output {
            assert!(!output.is_empty());
            let first_output = output[0].clone();
            if matches!(first_output, ContentBlockChatOutput::Text { .. }) {
                assert!(matches!(first_output, ContentBlockChatOutput::Text { .. }));
            } else if matches!(first_output, ContentBlockChatOutput::ToolCall { .. }) {
                assert!(matches!(
                    first_output,
                    ContentBlockChatOutput::ToolCall { .. }
                ));
            }
        }

        // Verify output if present for the list datapoint
        if let Some(output) = &list_datapoint.output {
            assert!(!output.is_empty());
            let first_output = output[0].clone();
            if matches!(first_output, ContentBlockChatOutput::Text { .. }) {
                assert!(matches!(first_output, ContentBlockChatOutput::Text { .. }));
            } else if matches!(first_output, ContentBlockChatOutput::ToolCall { .. }) {
                assert!(matches!(
                    first_output,
                    ContentBlockChatOutput::ToolCall { .. }
                ));
            }
        }

        // Verify tool_params if present for the list datapoint
        if let Some(additional_tools) = &list_datapoint.tool_params.additional_tools {
            assert!(!additional_tools.is_empty());
            let first_tool = &additional_tools[0];
            match &first_tool {
                Tool::Function(tool) => {
                    assert_eq!(tool.name, "get_temperature");
                    assert_eq!(
                        tool.description,
                        "Get the current temperature in a given location"
                    );
                    assert_eq!(
                        tool.parameters,
                        json!({
                            "$schema": "http://json-schema.org/draft-07/schema#",
                            "type": "object",
                            "properties": {
                                "location": {
                                    "type": "string",
                                    "description": "The location to get the temperature for (e.g. \"New York\")"
                                },
                                "units": {
                                    "type": "string",
                                    "description": "The units to get the temperature in (must be \"fahrenheit\" or \"celsius\")",
                                    "enum": ["fahrenheit", "celsius"]
                                }
                            },
                            "required": ["location"],
                            "additionalProperties": false
                        })
                    );
                }
                Tool::OpenAICustom(_) => panic!("Expected Function tool"),
            }
        }

        let datapoint_id = datapoint.id;
        let resp = client
            .delete(get_gateway_endpoint(&format!(
                "/datasets/{dataset_name}/datapoints/{datapoint_id}",
            )))
            .send()
            .await
            .unwrap();

        let status = resp.status();
        resp.text().await.unwrap();
        assert_eq!(status, StatusCode::OK);
    }
    tokio::time::sleep(Duration::from_millis(500)).await;

    let datapoints = select_chat_dataset_clickhouse(&clickhouse, &dataset_name)
        .await
        .unwrap();
    assert!(datapoints.is_empty());
}

#[tokio::test]
async fn test_insert_datapoint_chat_bad_request() {
    let client = Client::new();
    let dataset_name = format!("test-dataset-{}", Uuid::now_v7());
    println!("dataset_name: {dataset_name}");

    let resp = client
        .post(get_gateway_endpoint(&format!(
            "/datasets/{dataset_name}/datapoints",
        )))
        .json(&json!({
            "datapoints": [
                {
                    "function_name": "basic_test",
                    "input": {
                        "system": {"assistant_name": "Dummy"},
                        "messages": [
                            {
                                "role": "user",
                                "content": [
                                    {"type": "text", "text": "My synthetic input"}
                                ]
                            }
                        ]
                    },
                    "output": [
                        {"type": "tool_call", "id": "call_123", "name": "get_temperature", "arguments": {"location": "New York", "units": "fahrenheit"}}
                    ]
                }
            ]
        }))
        .send()
        .await
        .unwrap();

    let status = resp.status();
    assert_eq!(status, StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn test_datapoint_insert_synthetic_chat_with_tools() {
    let clickhouse = get_clickhouse().await;
    let client = Client::new();
    let dataset_name = format!("test-dataset-{}", Uuid::now_v7());
    let datapoint_id = Uuid::now_v7();

    // Define the tool params once to avoid duplication
    let tool_params = json!({
        "tools_available": [
            {
                "description": "Get the current temperature in a given location",
                "parameters": {
                    "$schema": "http://json-schema.org/draft-07/schema#",
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The location to get the temperature for (e.g. \"New York\")"
                        },
                        "units": {
                            "type": "string",
                            "description": "The units to get the temperature in (must be \"fahrenheit\" or \"celsius\")",
                            "enum": ["fahrenheit", "celsius"]
                        }
                    },
                    "required": ["location"],
                    "additionalProperties": false
                },
                "name": "get_temperature",
                "strict": false
            }
        ],
        "tool_choice": "auto",
        "parallel_tool_calls": false
    });

    let resp = client
        .put(get_gateway_endpoint(&format!(
            "/internal/datasets/{dataset_name}/datapoints/{datapoint_id}",
        )))
        .json(&json!({
            "function_name": "basic_test",
            "input": {
                "system": {"assistant_name": "Dummy"},
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "My synthetic input"}
                        ]
                    }
                ]
            },
            "output": [
                {"type": "tool_call", "id": "call_123", "name": "get_humidity", "arguments": {"location": "New York", "units": "fahrenheit"}}
            ],
            "tool_params": tool_params,
            "is_custom": true,
        }))
        .send()
        .await
        .unwrap();

    let status = resp.status();
    let resp_json = resp.json::<Value>().await.unwrap();

    // This should fail because the tool call is not in the tool_params
    assert_eq!(status, StatusCode::BAD_REQUEST);
    let err_msg = resp_json.get("error").unwrap().as_str().unwrap();
    println!("Error: {err_msg}");
    assert!(
        err_msg.contains("Demonstration contains invalid tool name"),
        "Unexpected error message: {err_msg}"
    );

    // Next we check invalid arguments
    let resp = client
        .put(get_gateway_endpoint(&format!(
            "/internal/datasets/{dataset_name}/datapoints/{datapoint_id}",
        )))
        .json(&json!({
            "function_name": "basic_test",
            "input": {
                "system": {"assistant_name": "Dummy"},
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "My synthetic input"}
                        ]
                    }
                ]
            },
            "output": [
                {"type": "tool_call", "id": "call_123", "name": "get_temperature", "arguments": {"city": "New York", "units": "fahrenheit"}}
            ],
            "tool_params": tool_params,
            "is_custom": true,
        }))
        .send()
        .await
        .unwrap();

    let status = resp.status();
    let resp_json = resp.json::<Value>().await.unwrap();

    // This request is incorrect
    assert_eq!(status, StatusCode::BAD_REQUEST);
    let err_msg = resp_json.get("error").unwrap().as_str().unwrap();
    println!("Error: {err_msg}");
    assert!(
        err_msg.contains("Demonstration contains invalid tool call arguments"),
        "Unexpected error message: {err_msg}"
    );

    let resp = client
    .put(get_gateway_endpoint(&format!(
        "/internal/datasets/{dataset_name}/datapoints/{datapoint_id}",
    )))
    .json(&json!({
        "function_name": "basic_test",
        "input": {
            "system": {"assistant_name": "Dummy"},
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "My synthetic input"}
                    ]
                }
            ]
        },
        "output": [
            {"type": "tool_call", "id": "call_123", "name": "get_temperature", "arguments": {"location": "New York", "units": "fahrenheit"}}
        ],
        "tool_params": tool_params,
        "is_custom": true,
    }))
    .send()
    .await
    .unwrap();

    let status = resp.status();
    assert!(status.is_success());
    let resp_json = resp.json::<Value>().await.unwrap();

    let id: Uuid = resp_json
        .get("id")
        .unwrap()
        .as_str()
        .unwrap()
        .parse()
        .unwrap();

    assert_eq!(id, datapoint_id);

    let mut datapoint = select_chat_datapoint_clickhouse(&clickhouse, id)
        .await
        .unwrap();

    let updated_at = datapoint
        .as_object_mut()
        .unwrap()
        .remove("updated_at")
        .unwrap();
    let updated_at = chrono::NaiveDateTime::parse_from_str(
        updated_at.as_str().unwrap(),
        CLICKHOUSE_DATETIME_FORMAT,
    )
    .unwrap()
    .and_utc();
    assert!(
        chrono::Utc::now()
            .signed_duration_since(updated_at)
            .num_seconds()
            < 5,
        "Unexpected updated_at: {updated_at:?}"
    );
    let expected = json!({
      "dataset_name": dataset_name,
      "function_name": "basic_test",
      "id": id.to_string(),
      "name": null,
      "episode_id": null,
      "input": "{\"system\":{\"assistant_name\":\"Dummy\"},\"messages\":[{\"role\":\"user\",\"content\":[{\"type\":\"text\",\"text\":\"My synthetic input\"}]}]}",
      "output": "[{\"type\":\"tool_call\",\"id\":\"call_123\",\"raw_name\":\"get_temperature\",\"raw_arguments\":\"{\\\"location\\\":\\\"New York\\\",\\\"units\\\":\\\"fahrenheit\\\"}\",\"name\":\"get_temperature\",\"arguments\":{\"location\":\"New York\",\"units\":\"fahrenheit\"}}]",
      "tool_params": "{\"tools_available\":[{\"description\":\"Get the current temperature in a given location\",\"parameters\":{\"$schema\":\"http://json-schema.org/draft-07/schema#\",\"type\":\"object\",\"properties\":{\"location\":{\"type\":\"string\",\"description\":\"The location to get the temperature for (e.g. \\\"New York\\\")\"},\"units\":{\"type\":\"string\",\"description\":\"The units to get the temperature in (must be \\\"fahrenheit\\\" or \\\"celsius\\\")\",\"enum\":[\"fahrenheit\",\"celsius\"]}},\"required\":[\"location\"],\"additionalProperties\":false},\"name\":\"get_temperature\",\"strict\":false}],\"tool_choice\":\"auto\",\"parallel_tool_calls\":false}",
      "dynamic_tools": ["{\"type\":\"function\",\"description\":\"Get the current temperature in a given location\",\"parameters\":{\"$schema\":\"http://json-schema.org/draft-07/schema#\",\"type\":\"object\",\"properties\":{\"location\":{\"type\":\"string\",\"description\":\"The location to get the temperature for (e.g. \\\"New York\\\")\"},\"units\":{\"type\":\"string\",\"description\":\"The units to get the temperature in (must be \\\"fahrenheit\\\" or \\\"celsius\\\")\",\"enum\":[\"fahrenheit\",\"celsius\"]}},\"required\":[\"location\"],\"additionalProperties\":false},\"name\":\"get_temperature\",\"strict\":false}"],
      "dynamic_provider_tools": [],
      "tool_choice": "auto",
      "parallel_tool_calls": false,
      "allowed_tools": "{\"tools\":[\"get_temperature\"],\"choice\":\"function_default\"}",
      "tags": {},
      "auxiliary": "",
      "is_deleted": false,
      "is_custom": true,
      "source_inference_id": null,
      "staled_at": null,
    });
    assert_eq!(datapoint, expected);
}

#[tokio::test]
async fn test_datapoint_insert_synthetic_json() {
    let clickhouse = get_clickhouse().await;
    let client = Client::new();
    let dataset_name = format!("test-dataset-{}", Uuid::now_v7());
    let datapoint_id = Uuid::now_v7();
    let source_inference_id = Uuid::now_v7();

    let resp = client
        .put(get_gateway_endpoint(&format!(
            "/internal/datasets/{dataset_name}/datapoints/{datapoint_id}",
        )))
        .json(&json!({
            "function_name": "json_success",
            "input": {"system": {"assistant_name": "Dummy"}, "messages": [{"role": "user", "content": [{"type": "template", "name": "user", "arguments": {"country": "US"}}]}]},
            "output": {"answer": "Hello"},
            "output_schema": {},
            "source_inference_id": source_inference_id,
            "is_custom": true,
        }))
        .send()
        .await
        .unwrap();

    let status = resp.status();
    let resp_json = resp.json::<Value>().await.unwrap();

    assert!(status.is_success(), "Bad request: {resp_json:?}");

    let id: Uuid = resp_json
        .get("id")
        .unwrap()
        .as_str()
        .unwrap()
        .parse()
        .unwrap();
    assert_eq!(id, datapoint_id);

    let mut datapoint = select_json_datapoint_clickhouse(&clickhouse, id)
        .await
        .unwrap();

    let updated_at = datapoint
        .as_object_mut()
        .unwrap()
        .remove("updated_at")
        .unwrap();
    let updated_at = chrono::NaiveDateTime::parse_from_str(
        updated_at.as_str().unwrap(),
        CLICKHOUSE_DATETIME_FORMAT,
    )
    .unwrap()
    .and_utc();
    assert!(
        chrono::Utc::now()
            .signed_duration_since(updated_at)
            .num_seconds()
            < 5,
        "Unexpected updated_at: {updated_at:?}"
    );

    let expected = json!({
      "dataset_name": dataset_name,
      "function_name": "json_success",
      "id": id,
      "episode_id": null,
      "input": "{\"system\":{\"assistant_name\":\"Dummy\"},\"messages\":[{\"role\":\"user\",\"content\":[{\"type\":\"template\",\"name\":\"user\",\"arguments\":{\"country\":\"US\"}}]}]}",
      "output": "{\"raw\":\"{\\\"answer\\\":\\\"Hello\\\"}\",\"parsed\":{\"answer\":\"Hello\"}}",
      "output_schema": "{}",
      "tags": {},
      "auxiliary": "",
      "is_deleted": false,
      "is_custom": true,
      "staled_at": null,
      "source_inference_id": source_inference_id.to_string(),
      "name": null,
    });
    assert_eq!(datapoint, expected);

    // Sleep to ensure that we get a different `updated_at` timestamp
    tokio::time::sleep(Duration::from_millis(1500)).await;

    // Now, update the existing datapoint and verify that it changes in clickhouse
    // Test updating with a bad output schema the output won't match (this should fail)
    let new_resp = client
    .put(get_gateway_endpoint(&format!(
        "/internal/datasets/{dataset_name}/datapoints/{datapoint_id}",
    )))
    .json(&json!({
        "function_name": "json_success",
        "input": {"system": {"assistant_name": "Dummy"}, "messages": [{"role": "user", "content": [{"type": "template", "name": "user", "arguments": {"country": "US"}}]}]},
        "output": {"answer": "New answer"},
        "output_schema": {"type": "object", "properties": {"confidence": {"type": "number"}}, "required": ["confidence"]},
        "is_custom": true,
    }))
    .send()
    .await
    .unwrap();
    assert_eq!(new_resp.status(), StatusCode::BAD_REQUEST);
    let resp_json = new_resp.json::<Value>().await.unwrap();
    let err_msg = resp_json.get("error").unwrap().as_str().unwrap();
    assert!(
        err_msg.contains("Demonstration does not fit function output schema"),
        "Unexpected error message: {err_msg}"
    );
    assert!(
        err_msg.contains("\"confidence\" is a required property"),
        "Error should mention the missing required property: {err_msg}"
    );

    // Sleep to ensure that we get a different `updated_at` timestamp
    tokio::time::sleep(Duration::from_millis(1500)).await;

    // Now try with the correct schema
    // NOTE: This tests the case where the source_inference_id is the same as the existing datapoint
    // but we are overwriting the same datapoint with a new one
    let new_resp = client
    .put(get_gateway_endpoint(&format!(
        "/internal/datasets/{dataset_name}/datapoints/{datapoint_id}",
    )))
    .json(&json!({
        "function_name": "json_success",
        "input": {"system": {"assistant_name": "Dummy"}, "messages": [{"role": "user", "content": [{"type": "template", "name": "user", "arguments": {"country": "US"}}]}]},
        "output": {"answer": "New answer"},
        "output_schema": {
            "type": "object",
            "properties": {
                "answer": {"type": "string"}
            },
            "required": ["answer"],
            "additionalProperties": false
        },
        "source_inference_id": source_inference_id,
        "is_custom": true,
    }))
    .send()
    .await
    .unwrap();

    assert!(
        new_resp.status().is_success(),
        "Bad request: {:?}",
        new_resp.text().await.unwrap()
    );

    // Force deduplication to run
    clickhouse
        .run_query_synchronous_no_params("OPTIMIZE TABLE JsonInferenceDatapoint".to_string())
        .await
        .unwrap();

    let mut datapoint = select_json_datapoint_clickhouse(&clickhouse, id)
        .await
        .unwrap();

    let new_updated_at = datapoint
        .as_object_mut()
        .unwrap()
        .remove("updated_at")
        .unwrap();
    let new_updated_at = chrono::NaiveDateTime::parse_from_str(
        new_updated_at.as_str().unwrap(),
        CLICKHOUSE_DATETIME_FORMAT,
    )
    .unwrap()
    .and_utc();
    assert!(
        chrono::Utc::now()
            .signed_duration_since(new_updated_at)
            .num_seconds()
            < 5,
        "Unexpected updated_at: {new_updated_at:?}"
    );

    assert!(
        new_updated_at > updated_at,
        "Expected updated_at to change: new_updated_at={new_updated_at:?} updated_at={updated_at:?}"
    );

    let expected = json!({
      "dataset_name": dataset_name,
      "function_name": "json_success",
      "id": id,
      "episode_id": null,
      "input": "{\"system\":{\"assistant_name\":\"Dummy\"},\"messages\":[{\"role\":\"user\",\"content\":[{\"type\":\"template\",\"name\":\"user\",\"arguments\":{\"country\":\"US\"}}]}]}",
      "output": "{\"raw\":\"{\\\"answer\\\":\\\"New answer\\\"}\",\"parsed\":{\"answer\":\"New answer\"}}",
      "output_schema": "{\"type\":\"object\",\"properties\":{\"answer\":{\"type\":\"string\"}},\"required\":[\"answer\"],\"additionalProperties\":false}",
      "tags": {},
      "auxiliary": "",
      "is_deleted": false,
      "is_custom": true,
      "staled_at": null,
      "source_inference_id": source_inference_id.to_string(),
      "name": null,
    });
    assert_eq!(datapoint, expected);

    // Try with a new datapoint_id and the same source_inference_id
    let new_datapoint_id = Uuid::now_v7();
    let new_resp = client
    .put(get_gateway_endpoint(&format!(
        "/internal/datasets/{dataset_name}/datapoints/{new_datapoint_id}",
    )))
    .json(&json!({
        "function_name": "json_success",
        "input": {"system": {"assistant_name": "Dummy"}, "messages": [{"role": "user", "content": [{"type": "template", "name": "user", "arguments": {"country": "US"}}]}]},
        "output": {"answer": "New answer"},
        "output_schema": {
            "type": "object",
            "properties": {
                "answer": {"type": "string"}
            },
            "required": ["answer"],
            "additionalProperties": false
        },
        "source_inference_id": source_inference_id,
        "is_custom": true,
    }))
    .send()
    .await
    .unwrap();
    let status = new_resp.status();
    assert_eq!(status, StatusCode::OK);

    // Check that the datapoint was inserted into clickhouse
    let datapoint = select_json_datapoint_clickhouse(&clickhouse, new_datapoint_id).await;
    assert!(datapoint.is_some());
    let datapoint = datapoint.unwrap();
    let datapoint = serde_json::from_value::<JsonInferenceDatapoint>(datapoint).unwrap();
    assert_eq!(datapoint.source_inference_id, Some(source_inference_id));
    assert!(datapoint.is_custom);
    assert_eq!(datapoint.id, new_datapoint_id);

    // Let's stale the old datapoint and try again
    clickhouse
        .delete_datapoints(&dataset_name, Some(&[datapoint_id]))
        .await
        .unwrap();

    // Try a new insert with the same source_inference_id but a new datapoint id
    let new_datapoint_id = Uuid::now_v7();
    let resp = client
       .put(get_gateway_endpoint(&format!(
           "/internal/datasets/{dataset_name}/datapoints/{new_datapoint_id}",
       )))
       .json(&json!({
           "function_name": "json_success",
           "input": {"system": {"assistant_name": "Dummy"}, "messages": [{"role": "user", "content": [{"type": "template", "name": "user", "arguments": {"country": "US"}}]}]},
           "output": {"answer": "New answer"},
           "output_schema": {
               "type": "object",
               "properties": {
                   "answer": {"type": "string"}
               },
               "required": ["answer"],
               "additionalProperties": false
           },
           "source_inference_id": source_inference_id,
           "is_custom": true,
       }))
       .send()
       .await.unwrap();

    let status = resp.status();
    assert_eq!(status, StatusCode::OK);

    let mut datapoint = select_json_datapoint_clickhouse(&clickhouse, new_datapoint_id)
        .await
        .unwrap();

    let updated_at = datapoint
        .as_object_mut()
        .unwrap()
        .remove("updated_at")
        .unwrap();
    let updated_at = chrono::NaiveDateTime::parse_from_str(
        updated_at.as_str().unwrap(),
        CLICKHOUSE_DATETIME_FORMAT,
    )
    .unwrap()
    .and_utc();
    assert!(
        chrono::Utc::now()
            .signed_duration_since(updated_at)
            .num_seconds()
            < 5,
        "Unexpected updated_at: {updated_at:?}"
    );

    let expected = json!({
      "dataset_name": dataset_name,
      "function_name": "json_success",
      "id": new_datapoint_id.to_string(),
      "episode_id": null,
      "input": "{\"system\":{\"assistant_name\":\"Dummy\"},\"messages\":[{\"role\":\"user\",\"content\":[{\"type\":\"template\",\"name\":\"user\",\"arguments\":{\"country\":\"US\"}}]}]}",
      "output": "{\"raw\":\"{\\\"answer\\\":\\\"New answer\\\"}\",\"parsed\":{\"answer\":\"New answer\"}}",
      "output_schema": "{\"type\":\"object\",\"properties\":{\"answer\":{\"type\":\"string\"}},\"required\":[\"answer\"],\"additionalProperties\":false}",
      "tags": {},
      "auxiliary": "",
      "is_deleted": false,
      "is_custom": true,
      "staled_at": null,
      "source_inference_id": source_inference_id.to_string(),
      "name": null,
    });
    assert_eq!(datapoint, expected);
}

#[tokio::test]
async fn test_create_delete_datapoint_json() {
    let clickhouse = get_clickhouse().await;
    let client = Client::new();
    let dataset_name = format!("test-dataset-{}", Uuid::now_v7());

    let alternate_output_schema = json!({
    "type": "object",
    "properties": {
        "response": {
            "type": "string"
        }
    },
    "required": ["response"],
    "additionalProperties": false
    });
    let resp = client
        .post(get_gateway_endpoint(&format!(
            "/datasets/{dataset_name}/datapoints",
        )))
        .json(&json!({
            "datapoints": [
                {
                    "function_name": "json_success",
                    "input": {"system": {"assistant_name": "Dummy"}, "messages": [{"role": "user", "content": [{"type": "template", "name": "user", "arguments": {"country": "US"}}]}]},
                    "output": {"answer": "Hello"},
                },
                {
                    "function_name": "json_success",
                    "input": {"system": {"assistant_name": "Dummy"}, "messages": [{"role": "user", "content": [{"type": "template", "name": "user", "arguments": {"country": "US"}}]}]},
                    "output": {"response": "Hello"},
                    "output_schema": alternate_output_schema
                }
            ]
        }))
        .send()
        .await
        .unwrap();

    let status = resp.status();

    assert!(status.is_success());
    // Sleep for a little so the getter can get
    tokio::time::sleep(Duration::from_millis(500)).await;

    // Test the lister
    let list_datapoints_response = client
        .get(get_gateway_endpoint(&format!(
            "/datasets/{dataset_name}/datapoints",
        )))
        .send()
        .await
        .unwrap();
    assert!(list_datapoints_response.status().is_success());
    let list_datapoints = list_datapoints_response.json::<Vec<Value>>().await.unwrap();
    assert_eq!(list_datapoints.len(), 2);
    for datapoint in &list_datapoints {
        // Test that the auxiliary field is not returned by the list datapoints API
        assert!(datapoint.get("auxiliary").is_none());
    }
    let list_datapoints = list_datapoints
        .into_iter()
        .map(|datapoint| serde_json::from_value::<JsonInferenceDatapoint>(datapoint).unwrap())
        .collect::<Vec<_>>();
    let datapoints = select_json_dataset_clickhouse(&clickhouse, &dataset_name)
        .await
        .unwrap();
    assert_eq!(datapoints.len(), 2);

    for (datapoint, list_datapoint) in datapoints.iter().zip(list_datapoints.iter()) {
        let pretty_datapoint = serde_json::to_string_pretty(&datapoint).unwrap();
        println!("pretty_datapoint: {pretty_datapoint}");
        // Verify the datapoint structure and content
        assert_eq!(datapoint.dataset_name, dataset_name);
        assert_eq!(datapoint.function_name, "json_success");
        assert!(!datapoint.is_deleted);
        assert!(datapoint.episode_id.is_none());
        assert!(datapoint.tags.as_ref().unwrap().is_empty());
        assert_eq!(datapoint.auxiliary, "");
        assert!(datapoint.staled_at.is_none());
        let datapoint_id = datapoint.id;

        // Test the getter
        let get_datapoint_response = client
            .get(get_gateway_endpoint(&format!(
                "/datasets/{dataset_name}/datapoints/{datapoint_id}",
            )))
            .send()
            .await
            .unwrap();
        assert!(get_datapoint_response.status().is_success());
        let get_datapoint_json = get_datapoint_response.json::<Value>().await.unwrap();
        // Assert that the auxiliary field is not returned by the get datapoint API
        assert!(get_datapoint_json.get("auxiliary").is_none());
        let get_datapoint =
            serde_json::from_value::<JsonInferenceDatapoint>(get_datapoint_json).unwrap();
        assert_eq!(&get_datapoint, datapoint);

        // Verify the list datapoint structure and content
        assert_eq!(list_datapoint.dataset_name, dataset_name);
        assert_eq!(list_datapoint.function_name, "json_success");
        assert!(!list_datapoint.is_deleted);
        assert!(list_datapoint.episode_id.is_none());
        assert!(list_datapoint.tags.as_ref().unwrap().is_empty());
        assert_eq!(list_datapoint.auxiliary, "");

        // Verify input structure
        let input = &datapoint.input;
        assert!(match input.system.as_ref().unwrap() {
            System::Template(arguments) => arguments.0.get("assistant_name"),
            System::Text(_) => panic!("Expected System::Template"),
        }
        .is_some());
        assert!(!input.messages.is_empty());
        let first_message = input.messages[0].clone();
        assert_eq!(first_message.role, Role::User);
        let content = first_message.content;
        assert!(!content.is_empty());
        let first_content = content[0].clone();
        assert!(matches!(
            first_content,
            InputMessageContent::Template { .. }
        ));

        // Verify the list datapoint input structure and content
        let input = &list_datapoint.input;
        assert!(match input.system.as_ref().unwrap() {
            System::Template(arguments) => arguments.0.get("assistant_name"),
            System::Text(_) => panic!("Expected System::Template"),
        }
        .is_some());
        assert!(!input.messages.is_empty());
        let first_message = input.messages[0].clone();
        assert_eq!(first_message.role, Role::User);
        let content = first_message.content;
        assert!(!content.is_empty());
        let first_content = content[0].clone();
        assert!(matches!(
            first_content,
            InputMessageContent::Template { .. }
        ));

        // Get the output schema
        let output_schema = &datapoint.output_schema;
        let output_schema_str = serde_json::to_string(&output_schema).unwrap();

        // Verify output if present
        if let Some(output) = &datapoint.output {
            if output_schema_str.contains("response") {
                assert_eq!(output.raw, Some("{\"response\":\"Hello\"}".to_string()));
                assert_eq!(output.parsed, Some(json!({"response": "Hello"})));
            } else {
                assert_eq!(output.raw, Some("{\"answer\":\"Hello\"}".to_string()));
                assert_eq!(output.parsed, Some(json!({"answer": "Hello"})));
            }
        }

        // Get the output schema from the list datapoint
        let output_schema = &list_datapoint.output_schema;
        let output_schema_str = serde_json::to_string(&output_schema).unwrap();

        // Verify output if present for the list datapoint
        if let Some(output) = &list_datapoint.output {
            if output_schema_str.contains("response") {
                assert_eq!(output.raw, Some("{\"response\":\"Hello\"}".to_string()));
                assert_eq!(output.parsed, Some(json!({"response": "Hello"})));
            } else {
                assert_eq!(output.raw, Some("{\"answer\":\"Hello\"}".to_string()));
                assert_eq!(output.parsed, Some(json!({"answer": "Hello"})));
            }
        }

        let datapoint_id = datapoint.id;
        let resp = client
            .delete(get_gateway_endpoint(&format!(
                "/datasets/{dataset_name}/datapoints/{datapoint_id}",
            )))
            .send()
            .await
            .unwrap();

        let status = resp.status();
        resp.text().await.unwrap();
        assert_eq!(status, StatusCode::OK);
    }
    tokio::time::sleep(Duration::from_millis(500)).await;

    let datapoints = select_json_dataset_clickhouse(&clickhouse, &dataset_name)
        .await
        .unwrap();
    assert!(datapoints.is_empty());
}

#[tokio::test]
async fn test_insert_datapoint_json_bad_output() {
    let client = Client::new();
    let dataset_name = format!("test-dataset-{}", Uuid::now_v7());

    let resp = client
        .post(get_gateway_endpoint(&format!(
            "/datasets/{dataset_name}/datapoints",
        )))
        .json(&json!({
            "datapoints": [
                {
                    "function_name": "json_success",
                    "input": {
                        "system": { "assistant_name": "Dummy" },
                        "messages": [
                            {
                                "role": "user",
                                "content": [
                                    {
                                        "type": "text",
                                        "arguments": { "country": "US" }
                                    }
                                ]
                            }
                        ]
                    },
                    "output": [
                        { "response": "Hello" }
                    ]
                }
            ]
        }))
        .send()
        .await
        .unwrap();

    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn test_delete_nonexistent_datapoint() {
    let client = Client::new();
    let dataset_name = format!("test-dataset-{}", Uuid::now_v7());
    let datapoint_id = Uuid::now_v7();

    let resp = client
        .delete(get_gateway_endpoint(&format!(
            "/datasets/{dataset_name}/datapoints/{datapoint_id}",
        )))
        .send()
        .await
        .unwrap();

    let status = resp.status();
    // For now, we don't care if the datapoint doesn't exist.
    assert_eq!(status, StatusCode::OK);
}

#[tokio::test]
async fn test_datapoint_insert_bad_name() {
    let client = Client::new();
    let dataset_name = "builder";
    let datapoint_id = Uuid::now_v7();

    let resp = client
        .put(get_gateway_endpoint(&format!(
            "/internal/datasets/{dataset_name}/datapoints/{datapoint_id}",
        )))
        .json(&json!({
            "function_name": "json_success",
            "input": {"system": {"assistant_name": "Dummy"}, "messages": [{"role": "user", "content": [{"type": "template", "name": "user", "arguments": {"country": "US"}}]}]},
            "output": {"answer": "Hello"},
            "output_schema": {},
        }))
        .send()
        .await
        .unwrap();

    let status = resp.status();
    let resp_json = resp.json::<Value>().await.unwrap();

    assert_eq!(status, StatusCode::BAD_REQUEST);
    let err_msg = resp_json.get("error").unwrap().as_str().unwrap();
    assert!(
        err_msg.contains("Invalid dataset name"),
        "Unexpected error message: {err_msg}"
    );
}

#[tokio::test]
async fn test_datapoint_insert_invalid_input_synthetic_chat() {
    let client = Client::new();
    let dataset_name = format!("test-dataset-{}", Uuid::now_v7());
    let datapoint_id = Uuid::now_v7();

    let resp = client
        .put(get_gateway_endpoint(&format!(
            "/internal/datasets/{dataset_name}/datapoints/{datapoint_id}",
        )))
        .json(&json!({
            "function_name": "variant_failover",
            "input": {"system": {"assistant_name": "Ferris"}, "messages": [{"role": "user", "content": [{"type": "text", "text": "My synthetic input"}]}]},
            "output": "Not a json object",
            "is_custom": false,
        }))
        .send()
        .await
        .unwrap();

    let status = resp.status();
    let resp_json = resp.json::<Value>().await.unwrap();
    let err_msg = resp_json
        .get("error")
        .unwrap()
        .as_str()
        .unwrap()
        .to_string();
    assert!(
        err_msg.contains("\"My synthetic input\" is not of type \"object\""),
        "Unexpected response: {resp_json}"
    );
    assert_eq!(status, StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn test_datapoint_insert_invalid_input_synthetic_json() {
    let client = Client::new();
    let dataset_name = format!("test-dataset-{}", Uuid::now_v7());
    let datapoint_id = Uuid::now_v7();

    let resp = client
        .put(get_gateway_endpoint(&format!(
            "/internal/datasets/{dataset_name}/datapoints/{datapoint_id}",
        )))
        .json(&json!({
            "function_name": "json_success",
            "input": {"system": {"assistant_name": "Ferris"}, "messages": [{"role": "user", "content": [{"type": "text", "text": "My synthetic input"}]}]},
            "output": "Not a json object",
            "output_schema": {},
            "is_custom": false,
        }))
        .send()
        .await
        .unwrap();

    let status = resp.status();
    let resp_json = resp.json::<Value>().await.unwrap();
    let err_msg = resp_json
        .get("error")
        .unwrap()
        .as_str()
        .unwrap()
        .to_string();
    assert!(
        err_msg.contains("\"My synthetic input\" is not of type \"object\""),
        "Unexpected response: {resp_json}"
    );
    assert_eq!(status, StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn test_datapoint_insert_invalid_output_synthetic_json() {
    let client = Client::new();
    let dataset_name = format!("test-dataset-{}", Uuid::now_v7());
    let datapoint_id = Uuid::now_v7();

    let resp = client
        .put(get_gateway_endpoint(&format!(
            "/internal/datasets/{dataset_name}/datapoints/{datapoint_id}",
        )))
        .json(&json!({
            "function_name": "json_success",
            "input": {"system": {"assistant_name": "Ferris"}, "messages": [{"role": "user", "content": [{"type": "template", "name": "user", "arguments": {"country": "US"}}]}]},
            "output": "Not a json object",
            "output_schema": {"type": "object", "properties": {"answer": {"type": "string"}}, "required": ["answer"]},
            "is_custom": false,
        }))
        .send()
        .await
        .unwrap();

    let status = resp.status();
    let resp_json = resp.json::<Value>().await.unwrap();
    let err_msg = resp_json
        .get("error")
        .unwrap()
        .as_str()
        .unwrap()
        .to_string();
    assert!(
        err_msg.contains("is not of type \"object\""),
        "Unexpected response: {resp_json}"
    );
    assert!(
        err_msg.contains("\"Not a json object\" is not of type \"object\""),
        "Unexpected response: {resp_json}"
    );
    assert_eq!(status, StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn test_datapoint_insert_synthetic_bad_uuid() {
    let client = Client::new();
    let dataset_name = format!("test-dataset-{}", Uuid::now_v7());
    let datapoint_uuid_v4 = "1b3e1480-a24a-4a69-942e-da960f9045fc";

    let resp = client
        .put(get_gateway_endpoint(&format!(
            "/internal/datasets/{dataset_name}/datapoints/{datapoint_uuid_v4}",
        )))
        .json(&json!({
            "function_name": "basic_test",
        }))
        .send()
        .await
        .unwrap();

    let status = resp.status();
    let resp_json = resp.json::<Value>().await.unwrap();
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(
        resp_json,
        json!({
            "error": "Invalid Datapoint ID: Version must be 7, got 4",
            "error_json": {
                "InvalidTensorzeroUuid": {
                    "kind": "Datapoint",
                    "message": "Version must be 7, got 4",
                }
            }
        })
    );
}

#[tokio::test]
async fn test_datapoint_insert_synthetic_bad_params() {
    let client = Client::new();
    let dataset_name = format!("test-dataset-{}", Uuid::now_v7());
    let datapoint_id = Uuid::now_v7();

    let resp = client
        .put(get_gateway_endpoint(&format!(
            "/internal/datasets/{dataset_name}/datapoints/{datapoint_id}",
        )))
        .json(&json!({
            "function_name": "basic_test",
        }))
        .send()
        .await
        .unwrap();

    let status = resp.status();
    let resp_json = resp.json::<Value>().await.unwrap();
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(
        resp_json,
        json!({
            "error": "Failed to deserialize chat datapoint: missing field `input`",
            "error_json": {
                "InvalidRequest": {
                    "message": "Failed to deserialize chat datapoint: missing field `input`"
                }
            }
        })
    );
}

#[tokio::test]
async fn test_datapoint_insert_output_inherit_chat() {
    let clickhouse = get_clickhouse().await;
    let client = Client::new();
    // Run inference (standard, no dryrun) to get an episode_id.
    let inference_payload = json!({
        "function_name": "basic_test",
        "input": {
            "system": {"assistant_name": "Alfred Pennyworth"},
            "messages": [{"role": "user", "content": "Hello, world!"}]
        },
        "stream": false,
    });

    let response = client
        .post(get_gateway_endpoint("/inference"))
        .json(&inference_payload)
        .send()
        .await
        .unwrap();
    assert!(response.status().is_success());
    let response_json = response.json::<Value>().await.unwrap();
    let episode_id = response_json.get("episode_id").unwrap().as_str().unwrap();
    let episode_id = Uuid::parse_str(episode_id).unwrap();
    let inference_id = response_json.get("inference_id").unwrap().as_str().unwrap();
    let inference_id = Uuid::parse_str(inference_id).unwrap();
    let variant_name = response_json.get("variant_name").unwrap().as_str().unwrap();

    let dataset_name = format!("test-dataset-{}", Uuid::now_v7());
    tokio::time::sleep(Duration::from_secs(1)).await;

    let resp = client
        .post(get_gateway_endpoint(&format!(
            "/internal/datasets/{dataset_name}/datapoints"
        )))
        .json(&json!({
            "inference_id": inference_id,
            "output": "inherit",
            "function_name": "basic_test",
            "variant_name": variant_name,
            "episode_id": episode_id,
        }))
        .send()
        .await
        .unwrap();

    assert!(
        resp.status().is_success(),
        "Bad request: {:?}",
        resp.text().await.unwrap()
    );
    let datapoint_id =
        Uuid::parse_str(resp.json::<Value>().await.unwrap()["id"].as_str().unwrap()).unwrap();
    assert_ne!(datapoint_id, inference_id);

    let mut datapoint = select_chat_datapoint_clickhouse(&clickhouse, datapoint_id)
        .await
        .unwrap();

    let updated_at = datapoint
        .as_object_mut()
        .unwrap()
        .remove("updated_at")
        .unwrap();
    let updated_at = chrono::NaiveDateTime::parse_from_str(
        updated_at.as_str().unwrap(),
        CLICKHOUSE_DATETIME_FORMAT,
    )
    .unwrap()
    .and_utc();
    assert!(
        chrono::Utc::now()
            .signed_duration_since(updated_at)
            .num_seconds()
            < 5,
        "Unexpected updated_at: {updated_at:?}"
    );
    let expected = json!({
      "dataset_name": dataset_name,
      "function_name": "basic_test",
      "id": datapoint_id.to_string(),
      "episode_id": episode_id.to_string(),
      "input": "{\"system\":{\"assistant_name\":\"Alfred Pennyworth\"},\"messages\":[{\"role\":\"user\",\"content\":[{\"type\":\"text\",\"text\":\"Hello, world!\"}]}]}",
      "output": "[{\"type\":\"text\",\"text\":\"Megumin gleefully chanted her spell, unleashing a thunderous explosion that lit up the sky and left a massive crater in its wake.\"}]",
      "tool_params": "",
      "dynamic_tools": [],
      "dynamic_provider_tools": [],
      "tool_choice": null,
      "parallel_tool_calls": null,
      "allowed_tools": null,
      "tags": {},
      "auxiliary": "",
      "is_deleted": false,
      "is_custom": false,
      "staled_at": null,
      "source_inference_id": inference_id.to_string(),
      "name": null,
    });
    assert_eq!(datapoint, expected);

    delete_datapoint(
        &clickhouse,
        DatapointKind::Chat,
        "basic_test",
        &dataset_name,
        datapoint_id,
    )
    .await;

    // Force deduplication to run
    clickhouse
        .run_query_synchronous_no_params("OPTIMIZE TABLE ChatInferenceDatapoint".to_string())
        .await
        .unwrap();

    assert!(select_chat_datapoint_clickhouse(&clickhouse, datapoint_id)
        .await
        .is_none());
}

#[tokio::test]
async fn test_datapoint_insert_output_none_chat() {
    let clickhouse = get_clickhouse().await;
    let client = Client::new();
    // Run inference (standard, no dryrun) to get an episode_id.
    let inference_payload = json!({
        "function_name": "basic_test",
        "input": {
            "system": {"assistant_name": "Alfred Pennyworth"},
            "messages": [{"role": "user", "content": "Hello, world!"}]
        },
        "stream": false,
    });

    let response = client
        .post(get_gateway_endpoint("/inference"))
        .json(&inference_payload)
        .send()
        .await
        .unwrap();
    assert!(response.status().is_success());
    let response_json = response.json::<Value>().await.unwrap();
    let episode_id = response_json.get("episode_id").unwrap().as_str().unwrap();
    let episode_id = Uuid::parse_str(episode_id).unwrap();
    let inference_id = response_json.get("inference_id").unwrap().as_str().unwrap();
    let inference_id = Uuid::parse_str(inference_id).unwrap();
    let variant_name = response_json.get("variant_name").unwrap().as_str().unwrap();
    tokio::time::sleep(Duration::from_secs(1)).await;

    let dataset_name = format!("test-dataset-{}", Uuid::now_v7());

    let resp = client
        .post(get_gateway_endpoint(&format!(
            "/internal/datasets/{dataset_name}/datapoints"
        )))
        .json(&json!({
            "inference_id": inference_id,
            "output": "none",
            "function_name": "basic_test",
            "variant_name": variant_name,
            "episode_id": episode_id,
        }))
        .send()
        .await
        .unwrap();

    assert!(
        resp.status().is_success(),
        "Bad request: {:?}",
        resp.text().await.unwrap()
    );
    let datapoint_id =
        Uuid::parse_str(resp.json::<Value>().await.unwrap()["id"].as_str().unwrap()).unwrap();
    assert_ne!(datapoint_id, inference_id);
    tokio::time::sleep(Duration::from_millis(500)).await;

    let mut datapoint = select_chat_datapoint_clickhouse(&clickhouse, datapoint_id)
        .await
        .unwrap();

    let updated_at = datapoint
        .as_object_mut()
        .unwrap()
        .remove("updated_at")
        .unwrap();
    let updated_at = chrono::NaiveDateTime::parse_from_str(
        updated_at.as_str().unwrap(),
        CLICKHOUSE_DATETIME_FORMAT,
    )
    .unwrap()
    .and_utc();
    assert!(
        chrono::Utc::now()
            .signed_duration_since(updated_at)
            .num_seconds()
            < 5,
        "Unexpected updated_at: {updated_at:?}"
    );

    let expected = json!({
      "dataset_name": dataset_name,
      "function_name": "basic_test",
      "id": datapoint_id.to_string(),
      "episode_id": episode_id.to_string(),
      "input": "{\"system\":{\"assistant_name\":\"Alfred Pennyworth\"},\"messages\":[{\"role\":\"user\",\"content\":[{\"type\":\"text\",\"text\":\"Hello, world!\"}]}]}",
      "output": null,
      "tool_params": "",
      "dynamic_tools": [],
      "dynamic_provider_tools": [],
      "tool_choice": null,
      "parallel_tool_calls": null,
      "allowed_tools": null,
      "tags": {},
      "auxiliary": "",
      "is_deleted": false,
      "is_custom": false,
      "staled_at": null,
      "source_inference_id": inference_id.to_string(),
      "name": null,
    });
    assert_eq!(datapoint, expected);
}

#[tokio::test]
async fn test_datapoint_create_bad_name() {
    let client = Client::new();
    // Run inference (standard, no dryrun) to get an episode_id.
    let inference_payload = json!({
        "function_name": "basic_test",
        "input": {
            "system": {"assistant_name": "Alfred Pennyworth"},
            "messages": [{"role": "user", "content": "Hello, world!"}]
        },
        "stream": false,
    });

    let response = client
        .post(get_gateway_endpoint("/inference"))
        .json(&inference_payload)
        .send()
        .await
        .unwrap();
    assert!(response.status().is_success());
    let response_json = response.json::<Value>().await.unwrap();
    let episode_id = response_json.get("episode_id").unwrap().as_str().unwrap();
    let episode_id = Uuid::parse_str(episode_id).unwrap();
    let inference_id = response_json.get("inference_id").unwrap().as_str().unwrap();
    let inference_id = Uuid::parse_str(inference_id).unwrap();
    let variant_name = response_json.get("variant_name").unwrap().as_str().unwrap();

    let dataset_name = "builder";

    let resp = client
        .post(get_gateway_endpoint(&format!(
            "/internal/datasets/{dataset_name}/datapoints"
        )))
        .json(&json!({
            "inference_id": inference_id,
            "output": "none",
            "function_name": "basic_test",
            "variant_name": variant_name,
            "episode_id": episode_id,
        }))
        .send()
        .await
        .unwrap();

    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    let err_msg = resp.text().await.unwrap();
    assert!(
        err_msg.contains("Invalid dataset name"),
        "Unexpected error message: {err_msg}"
    );
}

#[tokio::test]
async fn test_datapoint_insert_output_demonstration_chat() {
    let clickhouse = get_clickhouse().await;
    let client = Client::new();
    let function_name = "basic_test";
    // Run inference (standard, no dryrun) to get an episode_id.
    let inference_payload = json!({
        "function_name": function_name,
        "input": {
            "system": {"assistant_name": "Alfred Pennyworth"},
            "messages": [{"role": "user", "content": "Hello, world!"}]
        },
        "stream": false,
    });

    let response = client
        .post(get_gateway_endpoint("/inference"))
        .json(&inference_payload)
        .send()
        .await
        .unwrap();
    assert!(response.status().is_success());
    let response_json = response.json::<Value>().await.unwrap();
    let episode_id = response_json.get("episode_id").unwrap().as_str().unwrap();
    let episode_id = Uuid::parse_str(episode_id).unwrap();
    let inference_id = response_json.get("inference_id").unwrap().as_str().unwrap();
    let inference_id = Uuid::parse_str(inference_id).unwrap();
    let variant_name = response_json.get("variant_name").unwrap().as_str().unwrap();

    let dataset_name = format!("test-dataset-{}", Uuid::now_v7());

    let response = client
        .post(get_gateway_endpoint("/feedback"))
        .json(&json!({
            "inference_id": inference_id,
            "metric_name": "demonstration",
            "value": [{"type": "text", "text": "My demonstration chat answer"}],
        }))
        .send()
        .await
        .unwrap();
    assert!(
        response.status().is_success(),
        "Bad request: {:?}",
        response.text().await.unwrap()
    );

    // Sleep to ensure that we wrote to ClickHouse
    tokio::time::sleep(std::time::Duration::from_millis(500)).await;

    let resp = client
        .post(get_gateway_endpoint(&format!(
            "/internal/datasets/{dataset_name}/datapoints"
        )))
        .json(&json!({
            "inference_id": inference_id,
            "output": "demonstration",
            "function_name": function_name,
            "variant_name": variant_name,
            "episode_id": episode_id,
        }))
        .send()
        .await
        .unwrap();

    assert!(
        resp.status().is_success(),
        "Bad request: {:?}",
        resp.text().await.unwrap()
    );
    let datapoint_id =
        Uuid::parse_str(resp.json::<Value>().await.unwrap()["id"].as_str().unwrap()).unwrap();
    assert_ne!(datapoint_id, inference_id);

    let mut datapoint = select_chat_datapoint_clickhouse(&clickhouse, datapoint_id)
        .await
        .unwrap();

    let updated_at = datapoint
        .as_object_mut()
        .unwrap()
        .remove("updated_at")
        .unwrap();
    let updated_at = chrono::NaiveDateTime::parse_from_str(
        updated_at.as_str().unwrap(),
        CLICKHOUSE_DATETIME_FORMAT,
    )
    .unwrap()
    .and_utc();
    assert!(
        chrono::Utc::now()
            .signed_duration_since(updated_at)
            .num_seconds()
            < 5,
        "Unexpected updated_at: {updated_at:?}"
    );

    println!(
        "Datapoint: {}",
        serde_json::to_string_pretty(&datapoint).unwrap()
    );

    let expected = json!({
      "dataset_name": dataset_name,
      "function_name": "basic_test",
      "id": datapoint_id.to_string(),
      "episode_id": episode_id.to_string(),
      "input": "{\"system\":{\"assistant_name\":\"Alfred Pennyworth\"},\"messages\":[{\"role\":\"user\",\"content\":[{\"type\":\"text\",\"text\":\"Hello, world!\"}]}]}",
      "output": "[{\"type\":\"text\",\"text\":\"My demonstration chat answer\"}]",
      "tool_params": "",
      "dynamic_tools": [],
      "dynamic_provider_tools": [],
      "tool_choice": null,
      "parallel_tool_calls": null,
      "allowed_tools": null,
      "tags": {},
      "auxiliary": "",
      "is_deleted": false,
      "staled_at": null,
      "is_custom": false,
      "source_inference_id": inference_id.to_string(),
      "name": null,
    });
    assert_eq!(datapoint, expected);
}

#[tokio::test]
async fn test_datapoint_insert_output_inherit_json() {
    let clickhouse = get_clickhouse().await;
    let client = Client::new();
    // Run inference (standard, no dryrun) to get an episode_id.
    let inference_payload = json!({
        "function_name": "json_success",
        "input": {
            "system": {"assistant_name": "Alfred Pennyworth"},
            "messages": [{"role": "user", "content": [
                {"type": "template", "name": "user", "arguments": {"country": "Japan"}}
            ]}],
        },
        "stream": false,
    });

    let response = client
        .post(get_gateway_endpoint("/inference"))
        .json(&inference_payload)
        .send()
        .await
        .unwrap();
    assert!(response.status().is_success());
    let response_json = response.json::<Value>().await.unwrap();
    let episode_id = response_json.get("episode_id").unwrap().as_str().unwrap();
    let episode_id = Uuid::parse_str(episode_id).unwrap();
    let inference_id = response_json.get("inference_id").unwrap().as_str().unwrap();
    let inference_id = Uuid::parse_str(inference_id).unwrap();
    let variant_name = response_json.get("variant_name").unwrap().as_str().unwrap();
    tokio::time::sleep(Duration::from_secs(1)).await;

    let dataset_name = format!("test-dataset-{}", Uuid::now_v7());

    let resp = client
        .post(get_gateway_endpoint(&format!(
            "/internal/datasets/{dataset_name}/datapoints"
        )))
        .json(&json!({
            "inference_id": inference_id,
            "output": "inherit",
            "function_name": "json_success",
            "variant_name": variant_name,
            "episode_id": episode_id,
        }))
        .send()
        .await
        .unwrap();

    assert!(
        resp.status().is_success(),
        "Bad request: {:?}",
        resp.text().await.unwrap()
    );
    let datapoint_id =
        Uuid::parse_str(resp.json::<Value>().await.unwrap()["id"].as_str().unwrap()).unwrap();
    assert_ne!(datapoint_id, inference_id);

    let mut datapoint = select_json_datapoint_clickhouse(&clickhouse, datapoint_id)
        .await
        .unwrap();

    let updated_at = datapoint
        .as_object_mut()
        .unwrap()
        .remove("updated_at")
        .unwrap();
    let updated_at = chrono::NaiveDateTime::parse_from_str(
        updated_at.as_str().unwrap(),
        CLICKHOUSE_DATETIME_FORMAT,
    )
    .unwrap()
    .and_utc();
    assert!(
        chrono::Utc::now()
            .signed_duration_since(updated_at)
            .num_seconds()
            < 5,
        "Unexpected updated_at: {updated_at:?}"
    );

    let expected = json!({
      "dataset_name": dataset_name,
      "function_name": "json_success",
      "id": datapoint_id.to_string(),
      "episode_id": episode_id.to_string(),
      "input": "{\"system\":{\"assistant_name\":\"Alfred Pennyworth\"},\"messages\":[{\"role\":\"user\",\"content\":[{\"type\":\"template\",\"name\":\"user\",\"arguments\":{\"country\":\"Japan\"}}]}]}",
      "output": "{\"raw\":\"{\\\"answer\\\":\\\"Hello\\\"}\",\"parsed\":{\"answer\":\"Hello\"}}",
      "output_schema": "{\"type\":\"object\",\"properties\":{\"answer\":{\"type\":\"string\"}},\"required\":[\"answer\"],\"additionalProperties\":false}",
      "tags": {},
      "auxiliary": "",
      "is_deleted": false,
      "staled_at": null,
      "is_custom": false,
      "source_inference_id": inference_id.to_string(),
      "name": null,
    });
    assert_eq!(datapoint, expected);

    delete_datapoint(
        &clickhouse,
        DatapointKind::Json,
        "json_success",
        &dataset_name,
        datapoint_id,
    )
    .await;

    // Force deduplication to run
    clickhouse
        .run_query_synchronous_no_params("OPTIMIZE TABLE JsonInferenceDatapoint".to_string())
        .await
        .unwrap();

    assert!(select_json_datapoint_clickhouse(&clickhouse, datapoint_id)
        .await
        .is_none());
}

#[tokio::test]
async fn test_datapoint_insert_output_none_json() {
    let clickhouse = get_clickhouse().await;
    let client = Client::new();
    // Run inference (standard, no dryrun) to get an episode_id.
    let inference_payload = json!({
        "function_name": "json_success",
        "input": {
            "system": {"assistant_name": "Alfred Pennyworth"},
            "messages": [{"role": "user", "content": [{"type": "template", "name": "user", "arguments": {"country": "Japan"}}]}]
        },
        "stream": false,
    });

    let response = client
        .post(get_gateway_endpoint("/inference"))
        .json(&inference_payload)
        .send()
        .await
        .unwrap();
    assert!(response.status().is_success());
    let response_json = response.json::<Value>().await.unwrap();
    let episode_id = response_json.get("episode_id").unwrap().as_str().unwrap();
    let episode_id = Uuid::parse_str(episode_id).unwrap();
    let inference_id = response_json.get("inference_id").unwrap().as_str().unwrap();
    let inference_id = Uuid::parse_str(inference_id).unwrap();
    let variant_name = response_json.get("variant_name").unwrap().as_str().unwrap();
    tokio::time::sleep(Duration::from_secs(1)).await;

    let dataset_name = format!("test-dataset-{}", Uuid::now_v7());

    let resp = client
        .post(get_gateway_endpoint(&format!(
            "/internal/datasets/{dataset_name}/datapoints"
        )))
        .json(&json!({
            "inference_id": inference_id,
            "output": "none",
            "function_name": "json_success",
            "variant_name": variant_name,
            "episode_id": episode_id,
        }))
        .send()
        .await
        .unwrap();

    assert!(
        resp.status().is_success(),
        "Bad request: {:?}",
        resp.text().await.unwrap()
    );
    let datapoint_id =
        Uuid::parse_str(resp.json::<Value>().await.unwrap()["id"].as_str().unwrap()).unwrap();
    assert_ne!(datapoint_id, inference_id);

    let mut datapoint = select_json_datapoint_clickhouse(&clickhouse, datapoint_id)
        .await
        .unwrap();

    let updated_at = datapoint
        .as_object_mut()
        .unwrap()
        .remove("updated_at")
        .unwrap();
    let updated_at = chrono::NaiveDateTime::parse_from_str(
        updated_at.as_str().unwrap(),
        CLICKHOUSE_DATETIME_FORMAT,
    )
    .unwrap()
    .and_utc();
    assert!(
        chrono::Utc::now()
            .signed_duration_since(updated_at)
            .num_seconds()
            < 5,
        "Unexpected updated_at: {updated_at:?}"
    );

    let expected = json!({
      "dataset_name": dataset_name,
      "function_name": "json_success",
      "id": datapoint_id.to_string(),
      "episode_id": episode_id.to_string(),
      "input": "{\"system\":{\"assistant_name\":\"Alfred Pennyworth\"},\"messages\":[{\"role\":\"user\",\"content\":[{\"type\":\"template\",\"name\":\"user\",\"arguments\":{\"country\":\"Japan\"}}]}]}",
      "output":null,
      "output_schema": "{\"type\":\"object\",\"properties\":{\"answer\":{\"type\":\"string\"}},\"required\":[\"answer\"],\"additionalProperties\":false}",
      "tags": {},
      "auxiliary": "",
      "is_deleted": false,
      "is_custom": false,
      "staled_at": null,
      "source_inference_id": inference_id.to_string(),
      "name": null,
    });
    assert_eq!(datapoint, expected);
}

#[tokio::test]
async fn test_datapoint_insert_output_demonstration_json() {
    let clickhouse = get_clickhouse().await;
    let client = Client::new();
    // Run inference (standard, no dryrun) to get an episode_id.
    let inference_payload = json!({
        "function_name": "json_success",
        "input": {
            "system": {"assistant_name": "Alfred Pennyworth"},
            "messages": [{"role": "user", "content": [{"type": "template", "name": "user", "arguments": {"country": "Japan"}}]}]
        },
        "stream": false,
    });

    let response = client
        .post(get_gateway_endpoint("/inference"))
        .json(&inference_payload)
        .send()
        .await
        .unwrap();
    assert!(response.status().is_success());
    let response_json = response.json::<Value>().await.unwrap();
    let episode_id = response_json.get("episode_id").unwrap().as_str().unwrap();
    let episode_id = Uuid::parse_str(episode_id).unwrap();
    let inference_id = response_json.get("inference_id").unwrap().as_str().unwrap();
    let inference_id = Uuid::parse_str(inference_id).unwrap();
    let variant_name = response_json.get("variant_name").unwrap().as_str().unwrap();

    let dataset_name = format!("test-dataset-{}", Uuid::now_v7());

    let response = client
        .post(get_gateway_endpoint("/feedback"))
        .json(&json!({
            "inference_id": inference_id,
            "metric_name": "demonstration",
            "value": {"answer": "My demonstration answer"},
        }))
        .send()
        .await
        .unwrap();
    assert!(
        response.status().is_success(),
        "Bad request: {:?}",
        response.text().await.unwrap()
    );

    // Sleep to allow writing demonstration before making datapoint
    tokio::time::sleep(std::time::Duration::from_secs(1)).await;

    let resp = client
        .post(get_gateway_endpoint(&format!(
            "/internal/datasets/{dataset_name}/datapoints"
        )))
        .json(&json!({
            "inference_id": inference_id,
            "output": "demonstration",
            "function_name": "json_success",
            "variant_name": variant_name,
            "episode_id": episode_id,
        }))
        .send()
        .await
        .unwrap();

    assert!(
        resp.status().is_success(),
        "Bad request: {:?}",
        resp.text().await.unwrap()
    );
    let datapoint_id =
        Uuid::parse_str(resp.json::<Value>().await.unwrap()["id"].as_str().unwrap()).unwrap();
    assert_ne!(datapoint_id, inference_id);

    let mut datapoint = select_json_datapoint_clickhouse(&clickhouse, datapoint_id)
        .await
        .unwrap();

    let updated_at = datapoint
        .as_object_mut()
        .unwrap()
        .remove("updated_at")
        .unwrap();
    let updated_at = chrono::NaiveDateTime::parse_from_str(
        updated_at.as_str().unwrap(),
        CLICKHOUSE_DATETIME_FORMAT,
    )
    .unwrap()
    .and_utc();
    assert!(
        chrono::Utc::now()
            .signed_duration_since(updated_at)
            .num_seconds()
            < 5,
        "Unexpected updated_at: {updated_at:?}"
    );

    println!(
        "Datapoint: {}",
        serde_json::to_string_pretty(&datapoint).unwrap()
    );

    let expected = json!({
      "dataset_name": dataset_name,
      "function_name": "json_success",
      "id": datapoint_id.to_string(),
      "episode_id": episode_id.to_string(),
      "input": "{\"system\":{\"assistant_name\":\"Alfred Pennyworth\"},\"messages\":[{\"role\":\"user\",\"content\":[{\"type\":\"template\",\"name\":\"user\",\"arguments\":{\"country\":\"Japan\"}}]}]}",
      "output": "{\"raw\":\"{\\\"answer\\\":\\\"My demonstration answer\\\"}\",\"parsed\":{\"answer\":\"My demonstration answer\"}}",
      "output_schema": "{\"type\":\"object\",\"properties\":{\"answer\":{\"type\":\"string\"}},\"required\":[\"answer\"],\"additionalProperties\":false}",
      "tags": {},
      "auxiliary": "",
      "is_deleted": false,
      "is_custom": false,
      "staled_at": null,
      "source_inference_id": inference_id.to_string(),
      "name": null,
    });
    assert_eq!(datapoint, expected);
}

#[tokio::test]
async fn test_missing_inference_id() {
    let client = Client::new();
    let fake_inference_id = Uuid::now_v7();
    let resp = client
        .post(get_gateway_endpoint(
            "/internal/datasets/dummy-dataset/datapoints",
        ))
        .json(&json!({
            "inference_id": fake_inference_id,
            "output": "inherit",
            "function_name": "basic_test",
            "variant_name": "test",
            "episode_id": Uuid::now_v7(),
        }))
        .send()
        .await
        .unwrap();

    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    let body = resp.text().await.unwrap();
    assert!(
        body.contains(&format!("Inference `{fake_inference_id}` not found")),
        "Unexpected response: {body}"
    );
}

#[tokio::test]
async fn test_datapoint_missing_demonstration() {
    let client = Client::new();
    // Run inference (standard, no dryrun) to get an episode_id.
    let inference_payload = json!({
        "function_name": "json_success",
        "input": {
            "system": {"assistant_name": "Alfred Pennyworth"},
            "messages": [{"role": "user", "content": [{"type": "template", "name": "user", "arguments": {"country": "Japan"}}]}]
        },
        "stream": false,
    });

    let response = client
        .post(get_gateway_endpoint("/inference"))
        .json(&inference_payload)
        .send()
        .await
        .unwrap();
    assert!(response.status().is_success());
    tokio::time::sleep(Duration::from_secs(1)).await;
    let response_json = response.json::<Value>().await.unwrap();
    let episode_id = response_json.get("episode_id").unwrap().as_str().unwrap();
    let episode_id = Uuid::parse_str(episode_id).unwrap();
    let inference_id = response_json.get("inference_id").unwrap().as_str().unwrap();
    let inference_id = Uuid::parse_str(inference_id).unwrap();
    let variant_name = response_json.get("variant_name").unwrap().as_str().unwrap();

    let dataset_name = format!("test-dataset-{}", Uuid::now_v7());

    let resp = client
        .post(get_gateway_endpoint(&format!(
            "/internal/datasets/{dataset_name}/datapoints"
        )))
        .json(&json!({
            "inference_id": inference_id,
            "output": "demonstration",
            "function_name": "json_success",
            "variant_name": variant_name,
            "episode_id": episode_id,
        }))
        .send()
        .await
        .unwrap();

    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    let body = resp.text().await.unwrap();
    assert!(
        body.contains(&format!(
            "No demonstration found for inference `{inference_id}`"
        )),
        "Unexpected response: {body}"
    );
}

#[tokio::test]
async fn test_datapoint_insert_missing_output_chat() {
    let clickhouse = get_clickhouse().await;
    let client = Client::new();
    let dataset_name = format!("test-dataset-{}", Uuid::now_v7());
    let datapoint_id = Uuid::now_v7();

    let resp = client
        .put(get_gateway_endpoint(&format!(
            "/internal/datasets/{dataset_name}/datapoints/{datapoint_id}",
        )))
        .json(&json!({
            "function_name": "basic_test",
            "input": {"system": {"assistant_name": "Dummy"}, "messages": [{"role": "user", "content": [{"type": "text", "text": "My synthetic input"}]}]},
            // output field is deliberately omitted
            "is_custom": true,
        }))
        .send()
        .await
        .unwrap();

    let status = resp.status();
    assert!(
        status.is_success(),
        "Expected successful response, got: {status}"
    );

    let resp_json = resp.json::<Value>().await.unwrap();
    let id: Uuid = resp_json
        .get("id")
        .unwrap()
        .as_str()
        .unwrap()
        .parse()
        .unwrap();
    assert_eq!(id, datapoint_id);

    let mut datapoint = select_chat_datapoint_clickhouse(&clickhouse, id)
        .await
        .unwrap();

    datapoint
        .as_object_mut()
        .unwrap()
        .remove("updated_at")
        .unwrap();

    let expected = json!({
      "dataset_name": dataset_name,
      "function_name": "basic_test",
      "id": id.to_string(),
      "name": null,
      "episode_id": null,
      "input": "{\"system\":{\"assistant_name\":\"Dummy\"},\"messages\":[{\"role\":\"user\",\"content\":[{\"type\":\"text\",\"text\":\"My synthetic input\"}]}]}",
      "output": null,
      "tool_params": "{\"tools_available\":[],\"tool_choice\":\"auto\",\"parallel_tool_calls\":null}",
      "dynamic_tools": [],
      "dynamic_provider_tools": [],
      "tool_choice": "auto",
      "parallel_tool_calls": null,
      "allowed_tools": "{\"tools\":[],\"choice\":\"function_default\"}",
      "tags": {},
      "auxiliary": "",
      "is_deleted": false,
      "is_custom": true,
      "source_inference_id": null,
      "staled_at": null,
    });
    assert_eq!(datapoint, expected);
}

#[tokio::test]
async fn test_datapoint_insert_null_output_chat() {
    let clickhouse = get_clickhouse().await;
    let client = Client::new();
    let dataset_name = format!("test-dataset-{}", Uuid::now_v7());
    let datapoint_id = Uuid::now_v7();

    let resp = client
        .put(get_gateway_endpoint(&format!(
            "/internal/datasets/{dataset_name}/datapoints/{datapoint_id}",
        )))
        .json(&json!({
            "function_name": "basic_test",
            "input": {"system": {"assistant_name": "Dummy"}, "messages": [{"role": "user", "content": [{"type": "text", "text": "My synthetic input"}]}]},
            "output": null, // explicitly null output
            "is_custom": true,
        }))
        .send()
        .await
        .unwrap();

    let status = resp.status();
    assert!(
        status.is_success(),
        "Expected successful response, got: {status}"
    );

    let resp_json = resp.json::<Value>().await.unwrap();
    let id: Uuid = resp_json
        .get("id")
        .unwrap()
        .as_str()
        .unwrap()
        .parse()
        .unwrap();
    assert_eq!(id, datapoint_id);

    let mut datapoint = select_chat_datapoint_clickhouse(&clickhouse, id)
        .await
        .unwrap();

    datapoint
        .as_object_mut()
        .unwrap()
        .remove("updated_at")
        .unwrap();

    let expected = json!({
      "dataset_name": dataset_name,
      "function_name": "basic_test",
      "id": id.to_string(),
      "name": null,
      "episode_id": null,
      "input": "{\"system\":{\"assistant_name\":\"Dummy\"},\"messages\":[{\"role\":\"user\",\"content\":[{\"type\":\"text\",\"text\":\"My synthetic input\"}]}]}",
      "output": null,
      "tool_params": "{\"tools_available\":[],\"tool_choice\":\"auto\",\"parallel_tool_calls\":null}",
      "dynamic_tools": [],
      "dynamic_provider_tools": [],
      "tool_choice": "auto",
      "parallel_tool_calls": null,
      "allowed_tools": "{\"tools\":[],\"choice\":\"function_default\"}",
      "tags": {},
      "auxiliary": "",
      "is_deleted": false,
      "is_custom": true,
      "source_inference_id": null,
      "staled_at": null,
    });
    assert_eq!(datapoint, expected);
}

#[tokio::test]
async fn test_datapoint_insert_missing_output_json() {
    let clickhouse = get_clickhouse().await;
    let client = Client::new();
    let dataset_name = format!("test-dataset-{}", Uuid::now_v7());
    let datapoint_id = Uuid::now_v7();

    let resp = client
        .put(get_gateway_endpoint(&format!(
            "/internal/datasets/{dataset_name}/datapoints/{datapoint_id}",
        )))
        .json(&json!({
            "function_name": "json_success",
            "input": {"system": {"assistant_name": "Dummy"}, "messages": [{"role": "user", "content": [{"type": "template", "name": "user", "arguments": {"country": "US"}}]}]},
            "output_schema": {},
            "is_custom": false,
            // output field is deliberately omitted
        }))
        .send()
        .await
        .unwrap();

    let status = resp.status();
    assert!(
        status.is_success(),
        "Expected successful response, got: {status}"
    );

    let resp_json = resp.json::<Value>().await.unwrap();
    let id: Uuid = resp_json
        .get("id")
        .unwrap()
        .as_str()
        .unwrap()
        .parse()
        .unwrap();
    assert_eq!(id, datapoint_id);

    let mut datapoint = select_json_datapoint_clickhouse(&clickhouse, id)
        .await
        .unwrap();

    datapoint
        .as_object_mut()
        .unwrap()
        .remove("updated_at")
        .unwrap();

    let expected = json!({
      "dataset_name": dataset_name,
      "function_name": "json_success",
      "id": id,
      "episode_id": null,
      "input": "{\"system\":{\"assistant_name\":\"Dummy\"},\"messages\":[{\"role\":\"user\",\"content\":[{\"type\":\"template\",\"name\":\"user\",\"arguments\":{\"country\":\"US\"}}]}]}",
      "output": null,
      "output_schema": "{}",
      "tags": {},
      "auxiliary": "",
      "is_deleted": false,
      "is_custom": false,
      "staled_at": null,
      "source_inference_id": null,
      "name": null,
    });
    assert_eq!(datapoint, expected);
}

#[tokio::test]
async fn test_datapoint_insert_null_output_json() {
    let clickhouse = get_clickhouse().await;
    let client = Client::new();
    let dataset_name = format!("test-dataset-{}", Uuid::now_v7());
    let datapoint_id = Uuid::now_v7();

    let resp = client
        .put(get_gateway_endpoint(&format!(
            "/internal/datasets/{dataset_name}/datapoints/{datapoint_id}",
        )))
        .json(&json!({
            "function_name": "json_success",
            "input": {"system": {"assistant_name": "Dummy"}, "messages": [{"role": "user", "content": [{"type": "template", "name": "user", "arguments": {"country": "US"}}]}]},
            "output": null, // explicitly null output
            "output_schema": {},
            "is_custom": true,
        }))
        .send()
        .await
        .unwrap();

    let status = resp.status();
    assert!(
        status.is_success(),
        "Expected successful response, got: {status}"
    );

    let resp_json = resp.json::<Value>().await.unwrap();
    let id: Uuid = resp_json
        .get("id")
        .unwrap()
        .as_str()
        .unwrap()
        .parse()
        .unwrap();
    assert_eq!(id, datapoint_id);

    let mut datapoint = select_json_datapoint_clickhouse(&clickhouse, id)
        .await
        .unwrap();

    datapoint
        .as_object_mut()
        .unwrap()
        .remove("updated_at")
        .unwrap();

    let expected = json!({
      "dataset_name": dataset_name,
      "function_name": "json_success",
      "id": id,
      "episode_id": null,
      "input": "{\"system\":{\"assistant_name\":\"Dummy\"},\"messages\":[{\"role\":\"user\",\"content\":[{\"type\":\"template\",\"name\":\"user\",\"arguments\":{\"country\":\"US\"}}]}]}",
      "output": null,
      "output_schema": "{}",
      "tags": {},
      "auxiliary": "",
      "is_deleted": false,
      "is_custom": true,
      "staled_at": null,
      "source_inference_id": null,
      "name": null,
    });
    assert_eq!(datapoint, expected);
}

#[tokio::test]
async fn test_list_datapoints_nonexistent_dataset() {
    let client = Client::new();
    let dataset_name = format!("nonexistent-dataset-{}", Uuid::now_v7());

    let resp = client
        .get(get_gateway_endpoint(&format!(
            "/datasets/{dataset_name}/datapoints",
        )))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let datapoints: Vec<ChatInferenceDatapoint> = resp.json().await.unwrap();
    assert!(datapoints.is_empty());
}

#[tokio::test]
async fn test_get_nonexistent_datapoint() {
    let client = Client::new();
    let dataset_name = format!("test-dataset-{}", Uuid::now_v7());
    let datapoint_id = Uuid::now_v7();

    let resp = client
        .get(get_gateway_endpoint(&format!(
            "/datasets/{dataset_name}/datapoints/{datapoint_id}",
        )))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::NOT_FOUND);
}

#[tokio::test]
async fn test_list_datapoints_function_name_filter() {
    let client = Client::new();
    let dataset_name = format!("test-dataset-{}", Uuid::now_v7());

    // Create datapoints for two different functions
    let datapoints_basic = vec![
        json!({
            "function_name": "basic_test",
            "input": {
                "system": {"assistant_name": "Claude"},
                "messages": [{"role": "user", "content": "Hello basic"}]
            },
            "output": [{"type": "text", "text": "Response 1"}],
        }),
        json!({
            "function_name": "basic_test",
            "input": {
                "system": {"assistant_name": "Claude"},
                "messages": [{"role": "user", "content": "Hello basic 2"}]
            },
            "output": [{"type": "text", "text": "Response 2"}],
        }),
    ];

    let datapoints_basic_test_timeout = vec![json!({
        "function_name": "basic_test_timeout",
        "input": {
            "system": "aaa",
            "messages": [{"role": "user", "content": "Hello weather"}]
        },
        "output": [{"type": "text", "text": "Weather response"}],
    })];

    // Insert basic_test datapoints
    let resp = client
        .post(get_gateway_endpoint(&format!(
            "/datasets/{dataset_name}/datapoints",
        )))
        .json(&json!({
            "datapoints": datapoints_basic
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);

    // Insert basic_test_timeout datapoints
    let resp = client
        .post(get_gateway_endpoint(&format!(
            "/datasets/{dataset_name}/datapoints",
        )))
        .json(&json!({
            "datapoints": datapoints_basic_test_timeout
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);

    // Wait for data to be available
    tokio::time::sleep(Duration::from_millis(500)).await;

    // Test listing all datapoints (no filter)
    let resp = client
        .get(get_gateway_endpoint(&format!(
            "/datasets/{dataset_name}/datapoints",
        )))
        .send()
        .await
        .unwrap();
    assert!(resp.status().is_success());
    let all_datapoints: Vec<Value> = resp.json().await.unwrap();
    assert_eq!(all_datapoints.len(), 3);

    // Test filtering by basic_test function
    let resp = client
        .get(get_gateway_endpoint(&format!(
            "/datasets/{dataset_name}/datapoints?function_name=basic_test",
        )))
        .send()
        .await
        .unwrap();
    assert!(resp.status().is_success());
    let basic_datapoints: Vec<Value> = resp.json().await.unwrap();
    assert_eq!(basic_datapoints.len(), 2);

    // Verify all returned datapoints have function_name = "basic_test"
    for datapoint in &basic_datapoints {
        assert_eq!(datapoint["function_name"], "basic_test");
    }

    // Test filtering by basic_test_timeout function
    let resp = client
        .get(get_gateway_endpoint(&format!(
            "/datasets/{dataset_name}/datapoints?function_name=basic_test_timeout",
        )))
        .send()
        .await
        .unwrap();
    assert!(resp.status().is_success());
    let weather_datapoints: Vec<Value> = resp.json().await.unwrap();
    assert_eq!(weather_datapoints.len(), 1);

    // Verify all returned datapoints have function_name = "basic_test_timeout"
    for datapoint in &weather_datapoints {
        assert_eq!(datapoint["function_name"], "basic_test_timeout");
    }

    // Test filtering by non-existent function
    let resp = client
        .get(get_gateway_endpoint(&format!(
            "/datasets/{dataset_name}/datapoints?function_name=nonexistent_function",
        )))
        .send()
        .await
        .unwrap();
    assert!(resp.status().is_success());
    let empty_datapoints: Vec<Value> = resp.json().await.unwrap();
    assert_eq!(empty_datapoints.len(), 0);
}

#[tokio::test(flavor = "multi_thread")]
async fn test_stale_dataset_with_datapoints() {
    let clickhouse = get_clickhouse().await;
    let client = tensorzero::test_helpers::make_embedded_gateway().await;
    let dataset_name = format!("test-stale-dataset-{}", Uuid::now_v7());
    println!("dataset_name: {dataset_name}");

    // First, insert some chat datapoints
    let chat_datapoint_id1 = Uuid::now_v7();
    let chat_datapoint_id2 = Uuid::now_v7();

    let http_client = Client::new();
    let resp = http_client
        .put(get_gateway_endpoint(&format!(
            "/internal/datasets/{dataset_name}/datapoints/{chat_datapoint_id1}",
        )))
        .json(&json!({
            "function_name": "basic_test",
            "input": {"system": {"assistant_name": "Test"}, "messages": [{"role": "user", "content": [{"type": "text", "text": "Chat message 1"}]}]},
            "output": [{"type": "text", "text": "Response 1"}],
            "is_custom": false,
        }))
        .send()
        .await
        .unwrap();
    assert!(resp.status().is_success());

    let resp = http_client
        .put(get_gateway_endpoint(&format!(
            "/internal/datasets/{dataset_name}/datapoints/{chat_datapoint_id2}",
        )))
        .json(&json!({
            "function_name": "basic_test",
            "input": {"system": {"assistant_name": "Test"}, "messages": [{"role": "user", "content": [{"type": "text", "text": "Chat message 2"}]}]},
            "output": [{"type": "text", "text": "Response 2"}],
            "is_custom": false,
        }))
        .send()
        .await
        .unwrap();
    assert!(resp.status().is_success());

    // Insert some JSON datapoints
    let json_datapoint_id1 = Uuid::now_v7();
    let json_datapoint_id2 = Uuid::now_v7();

    let resp = http_client
        .put(get_gateway_endpoint(&format!(
            "/internal/datasets/{dataset_name}/datapoints/{json_datapoint_id1}",
        )))
        .json(&json!({
            "function_name": "json_success",
            "input": {"system": {"assistant_name": "Test"}, "messages": [{"role": "user", "content": [{"type": "template", "name": "user", "arguments": {"country": "Brazil"}}]}]},
            "output": {"answer": "Result 1"},
            "output_schema": {},
            "is_custom": false,
        }))
        .send()
        .await
        .unwrap();
    assert!(resp.status().is_success());

    let resp = http_client
        .put(get_gateway_endpoint(&format!(
            "/internal/datasets/{dataset_name}/datapoints/{json_datapoint_id2}",
        )))
        .json(&json!({
            "function_name": "json_success",
            "input": {"system": {"assistant_name": "Test"}, "messages": [{"role": "user", "content": [{"type": "template", "name": "user", "arguments": {"country": "France"}}]}]},
            "output": {"answer": "Result 2"},
            "output_schema": {},
            "is_custom": false,
        }))
        .send()
        .await
        .unwrap();
    assert!(resp.status().is_success());

    // Sleep for 500ms
    tokio::time::sleep(Duration::from_millis(500)).await;

    // Verify datapoints exist before staling
    let resp = http_client
        .get(get_gateway_endpoint(&format!(
            "/datasets/{dataset_name}/datapoints",
        )))
        .send()
        .await
        .unwrap();
    assert!(resp.status().is_success());
    let datapoints: Vec<Value> = resp.json().await.unwrap();
    assert_eq!(datapoints.len(), 4);

    // Now stale the entire dataset using the Rust client
    #[expect(deprecated)]
    let stale_result = client.stale_dataset(dataset_name.clone()).await.unwrap();
    assert_eq!(stale_result.num_staled_datapoints, 4);

    // Sleep for 500ms
    tokio::time::sleep(Duration::from_millis(500)).await;

    // Verify datapoints are no longer returned after staling
    let resp = http_client
        .get(get_gateway_endpoint(&format!(
            "/datasets/{dataset_name}/datapoints",
        )))
        .send()
        .await
        .unwrap();
    assert!(resp.status().is_success());
    let datapoints_after: Vec<Value> = resp.json().await.unwrap();
    assert_eq!(datapoints_after.len(), 0);

    // Verify the datapoints still exist in the database but are marked as staled
    let chat_datapoint1 = select_chat_datapoint_clickhouse(&clickhouse, chat_datapoint_id1)
        .await
        .unwrap();
    println!("chat_datapoint1: {chat_datapoint1:?}");
    assert!(chat_datapoint1["staled_at"].as_str().is_some());
    assert_ne!(chat_datapoint1["staled_at"].as_str().unwrap(), "");

    let json_datapoint1 = select_json_datapoint_clickhouse(&clickhouse, json_datapoint_id1)
        .await
        .unwrap();
    assert!(json_datapoint1["staled_at"].as_str().is_some());
    assert_ne!(json_datapoint1["staled_at"].as_str().unwrap(), "");
}

#[tokio::test(flavor = "multi_thread")]
async fn test_stale_dataset_empty() {
    let client = tensorzero::test_helpers::make_embedded_gateway().await;
    let dataset_name = format!("test-empty-stale-dataset-{}", Uuid::now_v7());

    // Stale an empty dataset (no datapoints exist)
    #[expect(deprecated)]
    let stale_result = client.stale_dataset(dataset_name.clone()).await.unwrap();
    assert_eq!(stale_result.num_staled_datapoints, 0);
}

#[tokio::test(flavor = "multi_thread")]
async fn test_stale_dataset_already_staled() {
    let client = tensorzero::test_helpers::make_embedded_gateway().await;
    let http_client = Client::new();
    let clickhouse = get_clickhouse().await;
    let dataset_name = format!("test-already-staled-{}", Uuid::now_v7());
    println!("dataset_name: {dataset_name}");

    // Insert a datapoint
    let datapoint_id = Uuid::now_v7();
    let resp = http_client
        .put(get_gateway_endpoint(&format!(
            "/internal/datasets/{dataset_name}/datapoints/{datapoint_id}",
        )))
        .json(&json!({
            "function_name": "basic_test",
            "input": {"system": {"assistant_name": "Test"}, "messages": [{"role": "user", "content": [{"type": "text", "text": "Test message"}]}]},
            "output": [{"type": "text", "text": "Test response"}],
            "is_custom": false,
        }))
        .send()
        .await
        .unwrap();
    assert!(resp.status().is_success());
    let resp_json: Value = resp.json().await.unwrap();
    let id = Uuid::parse_str(resp_json["id"].as_str().unwrap()).unwrap();
    // Sleep for 500ms
    tokio::time::sleep(Duration::from_millis(500)).await;

    println!("staling dataset");
    // Stale the dataset once
    #[expect(deprecated)]
    let stale_result1 = client.stale_dataset(dataset_name.clone()).await.unwrap();
    assert_eq!(stale_result1.num_staled_datapoints, 1);

    // Verify the datapoint is staled
    let datapoint = select_chat_datapoint_clickhouse(&clickhouse, id)
        .await
        .unwrap();
    let staled_at = datapoint["staled_at"].as_str().unwrap();

    // Wait for 500ms
    tokio::time::sleep(Duration::from_millis(500)).await;

    println!("staling dataset again");
    // Try to stale it again - should return 0 since datapoints are already staled (use HTTP for this)
    let resp = http_client
        .delete(get_gateway_endpoint(&format!("/datasets/{dataset_name}",)))
        .send()
        .await
        .unwrap();
    assert!(resp.status().is_success());
    let resp_json: Value = resp.json().await.unwrap();
    let num_staled_datapoints = resp_json["num_staled_datapoints"].as_u64().unwrap();
    assert_eq!(num_staled_datapoints, 0);

    // Verify the datapoint is still staled
    let datapoint = select_chat_datapoint_clickhouse(&clickhouse, id)
        .await
        .unwrap();
    let new_staled_at = datapoint["staled_at"].as_str().unwrap();
    assert_eq!(staled_at, new_staled_at);
}

#[tokio::test(flavor = "multi_thread")]
async fn test_stale_dataset_mixed_staled_fresh() {
    let http_client = Client::new();
    let clickhouse = get_clickhouse().await;
    let dataset_name = format!("test-mixed-stale-{}", Uuid::now_v7());

    // Insert first datapoint
    let datapoint_id1 = Uuid::now_v7();
    let resp = http_client
        .put(get_gateway_endpoint(&format!(
            "/internal/datasets/{dataset_name}/datapoints/{datapoint_id1}",
        )))
        .json(&json!({
            "function_name": "basic_test",
            "input": {"system": {"assistant_name": "Test"}, "messages": [{"role": "user", "content": [{"type": "text", "text": "Message 1"}]}]},
            "output": [{"type": "text", "text": "Response 1"}],
            "is_custom": false,
        }))
        .send()
        .await
        .unwrap();
    assert!(resp.status().is_success());

    // Delete (stale) the first datapoint individually
    let resp = http_client
        .delete(get_gateway_endpoint(&format!(
            "/datasets/{dataset_name}/datapoints/{datapoint_id1}",
        )))
        .send()
        .await
        .unwrap();
    assert!(resp.status().is_success());

    // Verify it's staled
    let datapoint1 = select_chat_datapoint_clickhouse(&clickhouse, datapoint_id1)
        .await
        .unwrap();
    let staled_at = datapoint1["staled_at"].as_str().unwrap();

    // Insert second datapoint after the first was staled
    let datapoint_id2 = Uuid::now_v7();
    let resp = http_client
        .put(get_gateway_endpoint(&format!(
            "/internal/datasets/{dataset_name}/datapoints/{datapoint_id2}",
        )))
        .json(&json!({
            "function_name": "basic_test",
            "input": {"system": {"assistant_name": "Test"}, "messages": [{"role": "user", "content": [{"type": "text", "text": "Message 2"}]}]},
            "output": [{"type": "text", "text": "Response 2"}],
            "is_custom": false,
        }))
        .send()
        .await
        .unwrap();
    assert!(resp.status().is_success());

    // Now stale the entire dataset - should only stale the fresh datapoint (use HTTP for this)
    let resp = http_client
        .delete(get_gateway_endpoint(&format!("/datasets/{dataset_name}",)))
        .send()
        .await
        .unwrap();
    assert!(resp.status().is_success());
    let resp_json: Value = resp.json().await.unwrap();
    let num_staled_datapoints = resp_json["num_staled_datapoints"].as_u64().unwrap();
    assert_eq!(num_staled_datapoints, 1);

    // Verify both datapoints are staled
    let datapoint2 = select_chat_datapoint_clickhouse(&clickhouse, datapoint_id2)
        .await
        .unwrap();
    assert!(datapoint2["staled_at"].as_str().is_some());

    let datapoint1 = select_chat_datapoint_clickhouse(&clickhouse, datapoint_id1)
        .await
        .unwrap();
    let new_staled_at = datapoint1["staled_at"].as_str().unwrap();
    assert_eq!(staled_at, new_staled_at);
}

/// Regression test for bug where tool call IDs were being lost during datapoint updates.
/// This test verifies that when a datapoint with tool calls containing IDs is updated
/// multiple times, the IDs are preserved across all updates.
#[tokio::test]
async fn test_update_datapoint_preserves_tool_call_ids() {
    use tensorzero_core::{
        db::datasets::{ChatInferenceDatapointInsert, DatapointInsert, DatasetQueries},
        inference::types::{ContentBlockChatOutput, StoredInput},
        tool::InferenceResponseToolCall,
    };

    let episode_id = Uuid::now_v7();
    let clickhouse = get_clickhouse().await;
    let client = Client::new();
    let dataset_name = "test_preserve_tool_call_ids";
    let datapoint_id = Uuid::now_v7();

    // Define tool params for the function
    let tool_params_json = json!({
        "tools_available": [
            {
                "name": "load_wikipedia_page",
                "description": "Load a Wikipedia page",
                "parameters": {
                    "$schema": "http://json-schema.org/draft-07/schema#",
                    "type": "object",
                    "properties": {
                        "title": {
                            "type": "string",
                            "description": "The title of the Wikipedia page"
                        }
                    },
                    "required": ["title"],
                    "additionalProperties": false
                },
                "strict": false
            }
        ],
        "tool_choice": "auto",
        "parallel_tool_calls": true
    });

    // Create initial datapoint using ClickHouse directly with tool calls that have IDs
    let initial_datapoint = ChatInferenceDatapointInsert {
        dataset_name: dataset_name.to_string(),
        function_name: "basic_test".to_string(),
        id: datapoint_id,
        name: Some("test_datapoint".to_string()),
        episode_id: Some(episode_id),
        input: StoredInput {
            system: None,
            messages: vec![],
        },
        output: Some(vec![ContentBlockChatOutput::ToolCall(
            InferenceResponseToolCall {
                id: "call_eBDiwZRnNnddB5tjcQbhdY0s".to_string(),
                name: Some("load_wikipedia_page".to_string()),
                raw_name: "load_wikipedia_page".to_string(),
                arguments: Some(json!({"title": "Russell Hoban"})),
                raw_arguments: "{\"title\": \"Russell Hoban\"}".to_string(),
            },
        )]),
        tool_params: None,
        tags: None,
        auxiliary: String::new(),
        staled_at: None,
        source_inference_id: None,
        is_custom: true,
    };

    clickhouse
        .insert_datapoints(&[DatapointInsert::Chat(initial_datapoint)])
        .await
        .unwrap();

    // Sleep for ClickHouse to become consistent
    tokio::time::sleep(std::time::Duration::from_secs(1)).await;

    // Verify the initial datapoint has the correct tool call IDs
    let datapoint = clickhouse
        .get_datapoints(&GetDatapointsParams {
            dataset_name: Some(dataset_name.to_string()),
            function_name: None,
            ids: Some(vec![datapoint_id]),
            limit: 20,
            offset: 0,
            allow_stale: false,
            filter: None,
            order_by: None,
            search_query_experimental: None,
        })
        .await
        .unwrap();
    let StoredDatapoint::Chat(chat_datapoint) = &datapoint[0] else {
        panic!("Datapoint is not a chat datapoint");
    };
    let output = chat_datapoint.output.as_ref().unwrap();
    if let ContentBlockChatOutput::ToolCall(first_tool_call_pre_update) = &output[0] {
        assert_eq!(
            first_tool_call_pre_update.id,
            "call_eBDiwZRnNnddB5tjcQbhdY0s"
        );
    } else {
        panic!("First content block is not a tool call");
    };

    // Update the datapoint (change input slightly) - IDs should be preserved
    let resp = client
        .put(get_gateway_endpoint(&format!(
            "/internal/datasets/{dataset_name}/datapoints/{datapoint_id}",
        )))
        .json(&json!({
            "function_name": "basic_test",
            "input": {
                "system": {"assistant_name": "TestBot"},
                "messages": [
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": "Tell me about Russell Hoban xyz abc"}]
                    }
                ]
            },
            "output": [
                {
                    "type": "tool_call",
                    "id": "call_eBDiwZRnNnddB5tjcQbhdY0s",
                    "name": "load_wikipedia_page",
                    "arguments": {"title": "Russell Hoban xyz abc"}
                },
            ],
            "tool_params": tool_params_json,
            "episode_id": episode_id,
            "is_custom": true,
        }))
        .send()
        .await
        .unwrap();

    assert!(resp.status().is_success());

    // Sleep for ClickHouse to become consistent
    tokio::time::sleep(std::time::Duration::from_secs(1)).await;

    // Verify IDs are still preserved after update
    let datapoint = clickhouse
        .get_datapoints(&GetDatapointsParams {
            dataset_name: Some(dataset_name.to_string()),
            function_name: None,
            ids: Some(vec![datapoint_id]),
            limit: 20,
            offset: 0,
            allow_stale: false,
            filter: None,
            order_by: None,
            search_query_experimental: None,
        })
        .await
        .unwrap();
    let StoredDatapoint::Chat(chat_datapoint) = &datapoint[0] else {
        panic!("Datapoint is not a chat datapoint");
    };
    let output = chat_datapoint.output.as_ref().unwrap();
    let ContentBlockChatOutput::ToolCall(first_tool_call_post_update) = &output[0] else {
        panic!("First content block is not a tool call");
    };
    assert_eq!(
        first_tool_call_post_update.id,
        "call_eBDiwZRnNnddB5tjcQbhdY0s"
    );
}

#[tokio::test]
async fn test_datapoint_update_invalid_output_schema_json() {
    let client = Client::new();
    let dataset_name = format!("test-dataset-{}", Uuid::now_v7());
    let datapoint_id = Uuid::now_v7();

    // Try to create/update a datapoint with an invalid output_schema
    let resp = client
        .put(get_gateway_endpoint(&format!(
            "/internal/datasets/{dataset_name}/datapoints/{datapoint_id}",
        )))
        .json(&json!({
            "function_name": "json_success",
            "input": {"system": {"assistant_name": "Dummy"}, "messages": [{"role": "user", "content": [{"type": "template", "name": "user", "arguments": {"country": "US"}}]}]},
            "output": {"answer": "Hello"},
            "output_schema": {
                "type": "invalid_type",  // This is an invalid JSON Schema type
                "properties": {
                    "answer": {"type": "string"}
                }
            },
            "is_custom": true
        }))
        .send()
        .await
        .unwrap();

    // The request should fail with a client error
    assert_eq!(
        resp.status(),
        StatusCode::BAD_REQUEST,
        "Expected BAD_REQUEST status for invalid output_schema"
    );

    let resp_json = resp.json::<Value>().await.unwrap();
    let error_message = resp_json.to_string();

    // Verify that the error message mentions the schema is invalid
    assert!(
        error_message.contains("invalid") || error_message.contains("schema"),
        "Error message should mention invalid schema: {error_message}"
    );
}

#[tokio::test]
async fn test_datapoint_create_invalid_output_schema_json() {
    let client = Client::new();
    let dataset_name = format!("test-dataset-{}", Uuid::now_v7());

    // Try to create datapoints with an invalid output_schema
    let resp = client
        .post(get_gateway_endpoint(&format!(
            "/datasets/{dataset_name}/datapoints",
        )))
        .json(&json!({
            "datapoints": [{
                "function_name": "json_success",
                "input": {"system": {"assistant_name": "Dummy"}, "messages": [{"role": "user", "content": [{"type": "template", "name": "user", "arguments": {"country": "US"}}]}]},
                "output": {"answer": "Hello"},
                "output_schema": {
                    "type": "invalid_type",  // This is an invalid JSON Schema type
                    "properties": {
                        "answer": {"type": "string"}
                    }
                }
            }]
        }))
        .send()
        .await
        .unwrap();
    // The request should fail with a client error
    assert_eq!(
        resp.status(),
        StatusCode::BAD_REQUEST,
        "Expected BAD_REQUEST status for invalid output_schema"
    );

    let resp_json = resp.json::<Value>().await.unwrap();
    let error_message = resp_json.to_string();

    // Verify that the error message mentions the schema is invalid
    assert!(
        error_message.contains("invalid") || error_message.contains("schema"),
        "Error message should mention invalid schema: {error_message}"
    );
}

pub mod tool_params;
