#![allow(clippy::print_stdout)]

use std::collections::HashMap;

use gateway::{
    inference::types::{ContentBlock, RequestMessage, Role},
    tool::{ToolCall, ToolResult},
};
use reqwest::{Client, StatusCode};
use serde_json::{json, Value};
use uuid::Uuid;

use crate::common::{
    get_clickhouse, get_gateway_endpoint, select_batch_model_inference_clickhouse,
    select_latest_batch_request_clickhouse,
};

use super::common::E2ETestProvider;

#[macro_export]
macro_rules! generate_batch_inference_tests {
    ($func:ident) => {
        use $crate::providers::batch::test_dynamic_json_mode_batch_inference_request_with_provider;
        use $crate::providers::batch::test_dynamic_tool_use_batch_inference_request_with_provider;
        use $crate::providers::batch::test_inference_params_batch_inference_request_with_provider;
        use $crate::providers::batch::test_json_mode_batch_inference_request_with_provider;
        use $crate::providers::batch::test_parallel_tool_use_batch_inference_request_with_provider;
        use $crate::providers::batch::test_simple_batch_inference_request_with_provider;
        use $crate::providers::batch::test_tool_multi_turn_batch_inference_request_with_provider;
        use $crate::providers::batch::test_tool_use_batch_inference_request_with_provider;

        #[tokio::test]
        async fn test_simple_batch_inference_request() {
            let all_providers = $func().await;
            let providers = all_providers.simple_inference;
            if all_providers.supports_batch_inference {
                for provider in providers {
                    test_simple_batch_inference_request_with_provider(provider).await;
                }
            }
        }

        #[tokio::test]
        async fn test_inference_params_batch_inference_request() {
            let all_providers = $func().await;
            let providers = all_providers.simple_inference;
            for provider in providers {
                test_inference_params_batch_inference_request_with_provider(provider).await;
            }
        }

        #[tokio::test]
        async fn test_tool_use_batch_inference_request() {
            let all_providers = $func().await;
            let providers = all_providers.simple_inference;
            for provider in providers {
                test_tool_use_batch_inference_request_with_provider(provider).await;
            }
        }

        #[tokio::test]
        async fn test_tool_multi_turn_batch_inference_request() {
            let all_providers = $func().await;
            let providers = all_providers.simple_inference;
            for provider in providers {
                test_tool_multi_turn_batch_inference_request_with_provider(provider).await;
            }
        }

        #[tokio::test]
        async fn test_dynamic_tool_use_batch_inference_request() {
            let all_providers = $func().await;
            let providers = all_providers.simple_inference;
            for provider in providers {
                test_dynamic_tool_use_batch_inference_request_with_provider(provider).await;
            }
        }

        #[tokio::test]
        async fn test_parallel_tool_use_batch_inference_request() {
            let all_providers = $func().await;
            let providers = all_providers.simple_inference;
            for provider in providers {
                test_parallel_tool_use_batch_inference_request_with_provider(provider).await;
            }
        }

        #[tokio::test]
        async fn test_json_mode_batch_inference_request() {
            let all_providers = $func().await;
            let providers = all_providers.simple_inference;
            for provider in providers {
                test_json_mode_batch_inference_request_with_provider(provider).await;
            }
        }

        #[tokio::test]
        async fn test_dynamic_json_mode_batch_inference_request() {
            let all_providers = $func().await;
            let providers = all_providers.simple_inference;
            for provider in providers {
                test_dynamic_json_mode_batch_inference_request_with_provider(provider).await;
            }
        }
    };
}

pub async fn test_simple_batch_inference_request_with_provider(provider: E2ETestProvider) {
    let episode_id = Uuid::now_v7();
    let tag_value = Uuid::now_v7().to_string();

    let payload = json!({
        "function_name": "basic_test",
        "variant_name": provider.variant_name,
        "episode_ids": [episode_id],
        "inputs":
            [{
               "system": {"assistant_name": "Dr. Mehta"},
               "messages": [
                {
                    "role": "user",
                    "content": "What is the capital city of Japan?"
                }
            ]}],
        "tags": [{"key": tag_value}],
    });

    let response = Client::new()
        .post(get_gateway_endpoint("/batch_inference"))
        .json(&payload)
        .send()
        .await
        .unwrap();

    // Check that the API response is ok
    assert_eq!(response.status(), StatusCode::OK);
    let response_json = response.json::<Value>().await.unwrap();

    println!("API response: {response_json:#?}");
    let batch_id = response_json.get("batch_id").unwrap().as_str().unwrap();
    let batch_id = Uuid::parse_str(batch_id).unwrap();

    let inference_ids = response_json
        .get("inference_ids")
        .unwrap()
        .as_array()
        .unwrap();
    assert_eq!(inference_ids.len(), 1);
    let inference_id = inference_ids.first().unwrap().as_str().unwrap();
    let inference_id = Uuid::parse_str(inference_id).unwrap();

    let episode_ids = response_json
        .get("episode_ids")
        .unwrap()
        .as_array()
        .unwrap();
    assert_eq!(episode_ids.len(), 1);
    let returned_episode_id = episode_ids.first().unwrap().as_str().unwrap();
    let returned_episode_id = Uuid::parse_str(returned_episode_id).unwrap();
    assert_eq!(returned_episode_id, episode_id);

    // Sleep to allow time for data to be inserted into ClickHouse (trailing writes from API)
    tokio::time::sleep(std::time::Duration::from_millis(500)).await;

    // Check if ClickHouse is ok - BatchModelInference Table
    let clickhouse = get_clickhouse().await;
    let result = select_batch_model_inference_clickhouse(&clickhouse, inference_id)
        .await
        .unwrap();

    println!("ClickHouse - BatchModelInference: {result:#?}");

    let id = result.get("id").unwrap().as_str().unwrap();
    let id = Uuid::parse_str(id).unwrap();
    assert_eq!(id, inference_id);

    let retrieved_batch_id = result.get("batch_id").unwrap().as_str().unwrap();
    let retrieved_batch_id = Uuid::parse_str(retrieved_batch_id).unwrap();
    assert_eq!(retrieved_batch_id, batch_id);

    let function_name = result.get("function_name").unwrap().as_str().unwrap();
    assert_eq!(function_name, payload["function_name"]);

    let variant_name = result.get("variant_name").unwrap().as_str().unwrap();
    assert_eq!(variant_name, provider.variant_name);

    let retrieved_episode_id = result.get("episode_id").unwrap().as_str().unwrap();
    let retrieved_episode_id = Uuid::parse_str(retrieved_episode_id).unwrap();
    assert_eq!(retrieved_episode_id, episode_id);

    let input: Value =
        serde_json::from_str(result.get("input").unwrap().as_str().unwrap()).unwrap();
    let correct_input = json!({
        "system": {"assistant_name": "Dr. Mehta"},
        "messages": [
            {
                "role": "user",
                "content": [{"type": "text", "value": "What is the capital city of Japan?"}]
            }
        ]
    });
    assert_eq!(input, correct_input);

    let input_messages = result.get("input_messages").unwrap().as_str().unwrap();
    let input_messages: Vec<RequestMessage> = serde_json::from_str(input_messages).unwrap();
    let expected_input_messages = vec![RequestMessage {
        role: Role::User,
        content: vec!["What is the capital city of Japan?".to_string().into()],
    }];
    assert_eq!(input_messages, expected_input_messages);

    let system = result.get("system").unwrap().as_str().unwrap();
    assert_eq!(
        system,
        "You are a helpful and friendly assistant named Dr. Mehta"
    );

    let tool_params = result.get("tool_params");
    assert_eq!(tool_params, Some(&Value::Null));

    let inference_params = result.get("inference_params").unwrap().as_str().unwrap();
    let inference_params: Value = serde_json::from_str(inference_params).unwrap();
    let inference_params = inference_params.get("chat_completion").unwrap();
    assert!(inference_params.get("temperature").is_none());
    assert!(inference_params.get("seed").is_none());
    assert_eq!(
        inference_params
            .get("max_tokens")
            .unwrap()
            .as_u64()
            .unwrap(),
        100
    );

    let model_name = result.get("model_name").unwrap().as_str().unwrap();
    assert_eq!(model_name, provider.model_name);

    let model_provider_name = result.get("model_provider_name").unwrap().as_str().unwrap();
    assert_eq!(model_provider_name, provider.model_provider_name);

    let output_schema = result.get("output_schema");
    assert_eq!(output_schema, Some(&Value::Null));

    let tags = result.get("tags").unwrap().as_object().unwrap();
    assert_eq!(tags.len(), 1);
    assert_eq!(tags.get("key").unwrap().as_str().unwrap(), tag_value);

    // Check if ClickHouse is ok - BatchRequest Table
    let result = select_latest_batch_request_clickhouse(&clickhouse, batch_id)
        .await
        .unwrap();

    println!("ClickHouse - BatchRequest: {result:#?}");

    let id = result.get("id").unwrap().as_str().unwrap();
    Uuid::parse_str(id).unwrap();

    let retrieved_batch_id = result.get("batch_id").unwrap().as_str().unwrap();
    let retrieved_batch_id = Uuid::parse_str(retrieved_batch_id).unwrap();
    assert_eq!(retrieved_batch_id, batch_id);
    let batch_params = result.get("batch_params").unwrap().as_str().unwrap();
    let _batch_params: Value = serde_json::from_str(batch_params).unwrap();
    // We can't check that the batch params are exactly the same because they vary per-provider
    // We will check that they are valid by using them instead.
    let model_name = result.get("model_name").unwrap().as_str().unwrap();
    assert_eq!(model_name, provider.model_name);

    let model_provider_name = result.get("model_provider_name").unwrap().as_str().unwrap();
    assert_eq!(model_provider_name, provider.model_provider_name);

    let status = result.get("status").unwrap().as_str().unwrap();
    assert_eq!(status, "pending");

    let errors = result.get("errors").unwrap().as_object().unwrap();
    assert_eq!(errors.len(), 0);
}

pub async fn test_inference_params_batch_inference_request_with_provider(
    provider: E2ETestProvider,
) {
    let episode_id = Uuid::now_v7();

    let payload = json!({
        "function_name": "basic_test",
        "variant_name": provider.variant_name,
        "episode_ids": [episode_id],
        "inputs":
            [{
               "system": {"assistant_name": "Dr. Mehta"},
               "messages": [
                {
                    "role": "user",
                    "content": "What is the capital city of Japan?"
                }
            ]}],
        "params": {
            "chat_completion": {
                "temperature": [0.9],
                "seed": [1337],
                "max_tokens": [120],
                "top_p": [0.9],
                "presence_penalty": [0.1],
                "frequency_penalty": [0.2],
            },
            "fake_variant_type": {
                "temperature": [0.8],
                "seed": [7331],
                "max_tokens": [80],
            }
        }
    });

    let response = Client::new()
        .post(get_gateway_endpoint("/batch_inference"))
        .json(&payload)
        .send()
        .await
        .unwrap();

    // Check that the API response is ok
    assert_eq!(response.status(), StatusCode::OK);
    let response_json = response.json::<Value>().await.unwrap();

    println!("API response: {response_json:#?}");

    let batch_id = response_json.get("batch_id").unwrap().as_str().unwrap();
    let batch_id = Uuid::parse_str(batch_id).unwrap();

    let inference_ids = response_json
        .get("inference_ids")
        .unwrap()
        .as_array()
        .unwrap();
    assert_eq!(inference_ids.len(), 1);
    let inference_id = inference_ids.first().unwrap().as_str().unwrap();
    let inference_id = Uuid::parse_str(inference_id).unwrap();

    let episode_ids = response_json
        .get("episode_ids")
        .unwrap()
        .as_array()
        .unwrap();
    assert_eq!(episode_ids.len(), 1);
    let returned_episode_id = episode_ids.first().unwrap().as_str().unwrap();
    let returned_episode_id = Uuid::parse_str(returned_episode_id).unwrap();
    assert_eq!(returned_episode_id, episode_id);

    // Sleep to allow time for data to be inserted into ClickHouse (trailing writes from API)
    tokio::time::sleep(std::time::Duration::from_millis(500)).await;

    // Check if ClickHouse is ok - BatchModelInference Table
    let clickhouse = get_clickhouse().await;
    let result = select_batch_model_inference_clickhouse(&clickhouse, inference_id)
        .await
        .unwrap();

    println!("ClickHouse - BatchModelInference: {result:#?}");

    let id = result.get("id").unwrap().as_str().unwrap();
    let id = Uuid::parse_str(id).unwrap();
    assert_eq!(id, inference_id);

    let retrieved_batch_id = result.get("batch_id").unwrap().as_str().unwrap();
    let retrieved_batch_id = Uuid::parse_str(retrieved_batch_id).unwrap();
    assert_eq!(retrieved_batch_id, batch_id);

    let function_name = result.get("function_name").unwrap().as_str().unwrap();
    assert_eq!(function_name, payload["function_name"]);

    let variant_name = result.get("variant_name").unwrap().as_str().unwrap();
    assert_eq!(variant_name, provider.variant_name);

    let retrieved_episode_id = result.get("episode_id").unwrap().as_str().unwrap();
    let retrieved_episode_id = Uuid::parse_str(retrieved_episode_id).unwrap();
    assert_eq!(retrieved_episode_id, episode_id);

    let input: Value =
        serde_json::from_str(result.get("input").unwrap().as_str().unwrap()).unwrap();
    let correct_input = json!({
        "system": {"assistant_name": "Dr. Mehta"},
        "messages": [
            {
                "role": "user",
                "content": [{"type": "text", "value": "What is the capital city of Japan?"}]
            }
        ]
    });
    assert_eq!(input, correct_input);

    let input_messages = result.get("input_messages").unwrap().as_str().unwrap();
    let input_messages: Vec<RequestMessage> = serde_json::from_str(input_messages).unwrap();
    let expected_input_messages = vec![RequestMessage {
        role: Role::User,
        content: vec!["What is the capital city of Japan?".to_string().into()],
    }];
    assert_eq!(input_messages, expected_input_messages);

    let system = result.get("system").unwrap().as_str().unwrap();
    assert_eq!(
        system,
        "You are a helpful and friendly assistant named Dr. Mehta"
    );

    let tool_params = result.get("tool_params");
    assert_eq!(tool_params, Some(&Value::Null));

    let inference_params = result.get("inference_params").unwrap().as_str().unwrap();
    let inference_params: Value = serde_json::from_str(inference_params).unwrap();
    let inference_params = inference_params.get("chat_completion").unwrap();
    let temperature = inference_params
        .get("temperature")
        .unwrap()
        .as_f64()
        .unwrap();
    assert_eq!(temperature, 0.9);
    let seed = inference_params.get("seed").unwrap().as_u64().unwrap();
    assert_eq!(seed, 1337);
    let max_tokens = inference_params
        .get("max_tokens")
        .unwrap()
        .as_u64()
        .unwrap();
    assert_eq!(max_tokens, 120);
    let top_p = inference_params.get("top_p").unwrap().as_f64().unwrap();
    assert_eq!(top_p, 0.9);
    let presence_penalty = inference_params
        .get("presence_penalty")
        .unwrap()
        .as_f64()
        .unwrap();
    assert_eq!(presence_penalty, 0.1);
    let frequency_penalty = inference_params
        .get("frequency_penalty")
        .unwrap()
        .as_f64()
        .unwrap();
    assert_eq!(frequency_penalty, 0.2);

    let model_name = result.get("model_name").unwrap().as_str().unwrap();
    assert_eq!(model_name, provider.model_name);

    let model_provider_name = result.get("model_provider_name").unwrap().as_str().unwrap();
    assert_eq!(model_provider_name, provider.model_provider_name);

    let output_schema = result.get("output_schema");
    assert_eq!(output_schema, Some(&Value::Null));

    let tags = result.get("tags").unwrap().as_object().unwrap();
    assert_eq!(tags.len(), 0);

    // Check if ClickHouse is ok - BatchRequest Table
    let result = select_latest_batch_request_clickhouse(&clickhouse, batch_id)
        .await
        .unwrap();

    println!("ClickHouse - BatchRequest: {result:#?}");

    let id = result.get("id").unwrap().as_str().unwrap();
    Uuid::parse_str(id).unwrap();

    let retrieved_batch_id = result.get("batch_id").unwrap().as_str().unwrap();
    let retrieved_batch_id = Uuid::parse_str(retrieved_batch_id).unwrap();
    assert_eq!(retrieved_batch_id, batch_id);
    let batch_params = result.get("batch_params").unwrap().as_str().unwrap();
    let _batch_params: Value = serde_json::from_str(batch_params).unwrap();
    // We can't check that the batch params are exactly the same because they vary per-provider
    // We will check that they are valid by using them instead.
    let model_name = result.get("model_name").unwrap().as_str().unwrap();
    assert_eq!(model_name, provider.model_name);

    let model_provider_name = result.get("model_provider_name").unwrap().as_str().unwrap();
    assert_eq!(model_provider_name, provider.model_provider_name);

    let status = result.get("status").unwrap().as_str().unwrap();
    assert_eq!(status, "pending");

    let errors = result.get("errors").unwrap().as_object().unwrap();
    assert_eq!(errors.len(), 0);
}

/// Tests that the tool use works as expected in a batch inference request.
/// Each element is a different test case from the e2e test suite for the synchronous API.
pub async fn test_tool_use_batch_inference_request_with_provider(provider: E2ETestProvider) {
    let mut episode_ids = Vec::new();
    for _ in 0..6 {
        episode_ids.push(Uuid::now_v7());
    }

    let payload = json!({
        "function_name": "weather_helper",
        "episode_ids": episode_ids,
        "inputs":
            [{
               "system": {"assistant_name": "Dr. Mehta"},
               "messages": [
                {
                    "role": "user",
                    "content": "What is the weather like in Tokyo (in Celsius)? Use the `get_temperature` tool."
                }
            ]},
            {
                "system": {"assistant_name": "Dr. Mehta"},
                "messages": [
                 {
                     "role": "user",
                     "content": "What is your name?"
                 }
             ]},
             {
                "system": { "assistant_name": "Dr. Mehta" },
                "messages": [
                    {
                        "role": "user",
                        "content": "What is your name?"
                    }
                ]
            },
            {
                "system": {"assistant_name": "Dr. Mehta"},
                "messages": [
                    {
                        "role": "user",
                        "content": "What is the weather like in Tokyo (in Celsius)? Use the `get_temperature` tool."
                    }
                ]},
                {
                    "system": {"assistant_name": "Dr. Mehta"},
                    "messages": [
                        {
                            "role": "user",
                            "content": "What is the temperature like in Tokyo (in Celsius)? Use the `get_temperature` tool."
                        }
                    ]},
                    {
                        "system": {"assistant_name": "Dr. Mehta"},
                        "messages": [
                            {
                                "role": "user",
                                "content": "What is the weather like in Tokyo? Call a function."
                            }
                        ]}
             ],
        "tool_choice": [null, null, "required", "none", {"specific": "self_destruct"}, null],
        "additional_tools": [null, null, null, null, [{
            "name": "self_destruct",
            "description": "Do not call this function under any circumstances.",
            "parameters": {
                "type": "object",
                "properties": {
                    "fast": {
                        "type": "boolean",
                        "description": "Whether to use a fast method to self-destruct."
                    },
                },
                "required": ["fast"],
                "additionalProperties": false
            },
        }], null],
        "allowed_tools": [null, null, null, null, null, ["get_humidity"]],
        "variant_name": provider.variant_name,
        "tags": [{"test": "auto_used"}, {"test": "auto_unused"}, {"test": "required"}, {"test": "none"}, {"test": "specific"}, {"test": "allowed_tools"}]
    });

    let response = Client::new()
        .post(get_gateway_endpoint("/batch_inference"))
        .json(&payload)
        .send()
        .await
        .unwrap();

    // Check if the API response is fine
    let status = response.status();
    let response_text = response.text().await.unwrap();

    println!("API response: {response_text:#?}");
    assert_eq!(status, StatusCode::OK);
    let response_json: Value = serde_json::from_str(&response_text).unwrap();

    let inference_ids = response_json
        .get("inference_ids")
        .unwrap()
        .as_array()
        .unwrap();
    assert_eq!(inference_ids.len(), 6);

    let batch_id = response_json.get("batch_id").unwrap().as_str().unwrap();
    let batch_id = Uuid::parse_str(batch_id).unwrap();

    // Parse all 6 inference IDs
    let inference_ids: Vec<Uuid> = inference_ids
        .iter()
        .map(|id| Uuid::parse_str(id.as_str().unwrap()).unwrap())
        .collect();
    assert_eq!(inference_ids.len(), 6);
    let mut inference_id_to_index: HashMap<Uuid, usize> =
        inference_ids.iter().cloned().zip(0..).collect();

    let response_episode_ids = response_json
        .get("episode_ids")
        .unwrap()
        .as_array()
        .unwrap();
    assert_eq!(response_episode_ids.len(), 6);

    // Parse and verify all 6 episode IDs match
    let response_episode_ids: Vec<Uuid> = response_episode_ids
        .iter()
        .map(|id| Uuid::parse_str(id.as_str().unwrap()).unwrap())
        .collect();

    // Verify each episode ID matches the expected episode ID
    for (episode_id_response, expected_episode_id) in response_episode_ids.iter().zip(&episode_ids)
    {
        assert_eq!(episode_id_response, expected_episode_id);
    }

    // Sleep to allow time for data to be inserted into ClickHouse (trailing writes from API)
    tokio::time::sleep(std::time::Duration::from_millis(500)).await;

    // Check if ClickHouse is ok - BatchModelInference Table
    let clickhouse = get_clickhouse().await;
    let correct_inputs = json!([
        {
            "system": {"assistant_name": "Dr. Mehta"},
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "text", "value": "What is the weather like in Tokyo (in Celsius)? Use the `get_temperature` tool."}]
                }
            ]
        },
        {
            "system": {"assistant_name": "Dr. Mehta"},
            "messages": [{"role": "user", "content": [{"type": "text", "value": "What is your name?"}]}]
        },
        {
            "system": {"assistant_name": "Dr. Mehta"},
            "messages": [{"role": "user", "content": [{"type": "text", "value": "What is your name?"}]}]
        },
        {
            "system": {"assistant_name": "Dr. Mehta"},
            "messages": [{"role": "user", "content": [{"type": "text", "value": "What is the weather like in Tokyo (in Celsius)? Use the `get_temperature` tool."}]}]
        },
        {
            "system": {"assistant_name": "Dr. Mehta"},
            "messages": [{"role": "user", "content": [{"type": "text", "value": "What is the temperature like in Tokyo (in Celsius)? Use the `get_temperature` tool."}]}]
        },
        {
            "system": {"assistant_name": "Dr. Mehta"},
            "messages": [{"role": "user", "content": [{"type": "text", "value": "What is the weather like in Tokyo? Call a function."}]}]
        }
    ]);
    let expected_input_messages = [
        [RequestMessage {
            role: Role::User,
            content: vec![
                "What is the weather like in Tokyo (in Celsius)? Use the `get_temperature` tool."
                    .to_string()
                    .into(),
            ],
        }],
        [RequestMessage {
            role: Role::User,
            content: vec!["What is your name?".to_string().into()],
        }],
        [RequestMessage {
            role: Role::User,
            content: vec!["What is your name?".to_string().into()],
        }],
        [RequestMessage {
            role: Role::User,
            content: vec![
                "What is the weather like in Tokyo (in Celsius)? Use the `get_temperature` tool."
                    .to_string()
                    .into(),
            ],
        }],
        [RequestMessage {
            role: Role::User,
            content: vec![
                "What is the temperature like in Tokyo (in Celsius)? Use the `get_temperature` tool."
                    .to_string()
                    .into(),
            ],
        }],
        [RequestMessage {
            role: Role::User,
            content: vec![
                "What is the weather like in Tokyo? Call a function."
                    .to_string()
                    .into(),
            ],
        }]
    ];

    let expected_systems = [
        "You are a helpful and friendly assistant named Dr. Mehta.\n\nPeople will ask you questions about the weather.\n\nIf asked about the weather, just respond with the tool call. Use the \"get_temperature\" tool.\n\nIf provided with a tool result, use it to respond to the user (e.g. \"The weather in New York is 55 degrees Fahrenheit.\").",
        "You are a helpful and friendly assistant named Dr. Mehta.\n\nPeople will ask you questions about the weather.\n\nIf asked about the weather, just respond with the tool call. Use the \"get_temperature\" tool.\n\nIf provided with a tool result, use it to respond to the user (e.g. \"The weather in New York is 55 degrees Fahrenheit.\").",
        "You are a helpful and friendly assistant named Dr. Mehta.\n\nPeople will ask you questions about the weather.\n\nIf asked about the weather, just respond with the tool call. Use the \"get_temperature\" tool.\n\nIf provided with a tool result, use it to respond to the user (e.g. \"The weather in New York is 55 degrees Fahrenheit.\").",
        "You are a helpful and friendly assistant named Dr. Mehta.\n\nPeople will ask you questions about the weather.\n\nIf asked about the weather, just respond with the tool call. Use the \"get_temperature\" tool.\n\nIf provided with a tool result, use it to respond to the user (e.g. \"The weather in New York is 55 degrees Fahrenheit.\").",
        "You are a helpful and friendly assistant named Dr. Mehta.\n\nPeople will ask you questions about the weather.\n\nIf asked about the weather, just respond with the tool call. Use the \"get_temperature\" tool.\n\nIf provided with a tool result, use it to respond to the user (e.g. \"The weather in New York is 55 degrees Fahrenheit.\").",
        "You are a helpful and friendly assistant named Dr. Mehta.\n\nPeople will ask you questions about the weather.\n\nIf asked about the weather, just respond with the tool call. Use the \"get_temperature\" tool.\n\nIf provided with a tool result, use it to respond to the user (e.g. \"The weather in New York is 55 degrees Fahrenheit.\")."
    ];
    let expected_tool_params = [
        json!({
            "tool_choice": "auto",
            "parallel_tool_calls": false,
            "tools_available": [{
                "name": "get_temperature",
                "description": "Get the current temperature in a given location",
                "strict": false,
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
                }
            }]
        }),
        json!({
            "tools_available": [{
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
            }],
            "tool_choice": "auto",
            "parallel_tool_calls": false
        }),
        json!({
            "tools_available": [{
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
            }],
            "tool_choice": "required",
            "parallel_tool_calls": false
        }),
        json!({
            "tools_available": [{
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
            }],
            "tool_choice": "none",
            "parallel_tool_calls": false
        }),
        json!({
            "tools_available": [{
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
            }, {
                "description": "Do not call this function under any circumstances.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "fast": {
                            "type": "boolean",
                            "description": "Whether to use a fast method to self-destruct."
                        }
                    },
                    "required": ["fast"],
                    "additionalProperties": false
                },
                "name": "self_destruct",
                "strict": false
            }],
            "tool_choice": {
                "specific": "self_destruct"
            },
            "parallel_tool_calls": false
        }),
        json!({
            "tools_available": [{
                "description": "Get the current humidity in a given location",
                "parameters": {
                    "$schema": "http://json-schema.org/draft-07/schema#",
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The location to get the humidity for (e.g. \"New York\")"
                        }
                    },
                    "required": ["location"],
                    "additionalProperties": false
                },
                "name": "get_humidity",
                "strict": false
            }],
            "tool_choice": "auto",
            "parallel_tool_calls": false
        }),
    ];

    let expected_inference_params = vec![
        json!({
            "chat_completion": {
                "max_tokens": 100,
            }
        }),
        json!({"chat_completion": {"max_tokens": 100}}),
        json!({"chat_completion": {"max_tokens": 100}}),
        json!({"chat_completion": {"max_tokens": 100}}),
        json!({"chat_completion": {"max_tokens": 100}}),
        json!({"chat_completion": {"max_tokens": 100}}),
    ];

    for inference_id in inference_ids {
        let result = select_batch_model_inference_clickhouse(&clickhouse, inference_id)
            .await
            .unwrap();
        let id_str = result.get("id").unwrap().as_str().unwrap();
        let id = Uuid::parse_str(id_str).unwrap();
        let i = inference_id_to_index.remove(&id).unwrap();
        println!("ClickHouse - BatchModelInference (#{i}): {result:#?}");

        let retrieved_batch_id = result.get("batch_id").unwrap().as_str().unwrap();
        let retrieved_batch_id = Uuid::parse_str(retrieved_batch_id).unwrap();
        assert_eq!(retrieved_batch_id, batch_id);

        let function_name = result.get("function_name").unwrap().as_str().unwrap();
        assert_eq!(function_name, payload["function_name"]);

        let variant_name = result.get("variant_name").unwrap().as_str().unwrap();
        assert_eq!(variant_name, provider.variant_name);

        let retrieved_episode_id = result.get("episode_id").unwrap().as_str().unwrap();
        let retrieved_episode_id = Uuid::parse_str(retrieved_episode_id).unwrap();
        assert_eq!(retrieved_episode_id, episode_ids[i]);

        let input: Value =
            serde_json::from_str(result.get("input").unwrap().as_str().unwrap()).unwrap();
        assert_eq!(input, correct_inputs[i]);

        let input_messages = result.get("input_messages").unwrap().as_str().unwrap();
        let input_messages: Vec<RequestMessage> = serde_json::from_str(input_messages).unwrap();
        assert_eq!(input_messages, expected_input_messages[i]);

        let system = result.get("system").unwrap().as_str().unwrap();
        assert_eq!(system, expected_systems[i]);

        let tool_params: Value =
            serde_json::from_str(result.get("tool_params").unwrap().as_str().unwrap()).unwrap();
        assert_eq!(tool_params, expected_tool_params[i]);
        let inference_params = result.get("inference_params").unwrap().as_str().unwrap();
        let inference_params: Value = serde_json::from_str(inference_params).unwrap();
        assert_eq!(inference_params, expected_inference_params[i]);

        let model_name = result.get("model_name").unwrap().as_str().unwrap();
        assert_eq!(model_name, provider.model_name);

        let model_provider_name = result.get("model_provider_name").unwrap().as_str().unwrap();
        assert_eq!(model_provider_name, provider.model_provider_name);

        let output_schema = result.get("output_schema");
        assert_eq!(output_schema, Some(&Value::Null));

        let tags = result.get("tags").unwrap().as_object().unwrap();
        assert_eq!(tags.len(), 1);
        assert_eq!(tags.get("test").unwrap(), &payload["tags"][i]["test"]);
    }

    // Check if ClickHouse is ok - BatchRequest Table
    let result = select_latest_batch_request_clickhouse(&clickhouse, batch_id)
        .await
        .unwrap();

    println!("ClickHouse - BatchRequest: {result:#?}");

    let id = result.get("id").unwrap().as_str().unwrap();
    Uuid::parse_str(id).unwrap();

    let retrieved_batch_id = result.get("batch_id").unwrap().as_str().unwrap();
    let retrieved_batch_id = Uuid::parse_str(retrieved_batch_id).unwrap();
    assert_eq!(retrieved_batch_id, batch_id);
    let batch_params = result.get("batch_params").unwrap().as_str().unwrap();
    let _batch_params: Value = serde_json::from_str(batch_params).unwrap();
    // We can't check that the batch params are exactly the same because they vary per-provider
    // We will check that they are valid by using them instead.
    let model_name = result.get("model_name").unwrap().as_str().unwrap();
    assert_eq!(model_name, provider.model_name);

    let model_provider_name = result.get("model_provider_name").unwrap().as_str().unwrap();
    assert_eq!(model_provider_name, provider.model_provider_name);

    let status = result.get("status").unwrap().as_str().unwrap();
    assert_eq!(status, "pending");

    let errors = result.get("errors").unwrap().as_object().unwrap();
    assert_eq!(errors.len(), 0);
}

pub async fn test_tool_multi_turn_batch_inference_request_with_provider(provider: E2ETestProvider) {
    let episode_id = Uuid::now_v7();

    let payload = json!({
       "function_name": "weather_helper",
        "episode_ids": [episode_id],
        "inputs":[{
            "system": {"assistant_name": "Dr. Mehta"},
            "messages": [
                {
                    "role": "user",
                    "content": "What is the weather like in Tokyo (in Celsius)? Use the `get_temperature` tool."
                },
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_call",
                            "id": "123456789",
                            "name": "get_temperature",
                            "arguments": "{\"location\": \"Tokyo\"}"
                        }
                    ]
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "id": "123456789",
                            "name": "get_temperature",
                            "result": "70"
                        }
                    ]
                }
            ]}],
        "variant_name": provider.variant_name,
        "tags": [{"test": "multi_turn"}]
    });

    let response = Client::new()
        .post(get_gateway_endpoint("/batch_inference"))
        .json(&payload)
        .send()
        .await
        .unwrap();

    // Check that the API response is ok
    assert_eq!(response.status(), StatusCode::OK);
    let response_json = response.json::<Value>().await.unwrap();

    println!("API response: {response_json:#?}");
    let batch_id = response_json.get("batch_id").unwrap().as_str().unwrap();
    let batch_id = Uuid::parse_str(batch_id).unwrap();

    let inference_ids = response_json
        .get("inference_ids")
        .unwrap()
        .as_array()
        .unwrap();
    assert_eq!(inference_ids.len(), 1);
    let inference_id = inference_ids.first().unwrap().as_str().unwrap();
    let inference_id = Uuid::parse_str(inference_id).unwrap();

    let episode_ids = response_json
        .get("episode_ids")
        .unwrap()
        .as_array()
        .unwrap();
    assert_eq!(episode_ids.len(), 1);
    let returned_episode_id = episode_ids.first().unwrap().as_str().unwrap();
    let returned_episode_id = Uuid::parse_str(returned_episode_id).unwrap();
    assert_eq!(returned_episode_id, episode_id);

    // Sleep to allow time for data to be inserted into ClickHouse (trailing writes from API)
    tokio::time::sleep(std::time::Duration::from_millis(500)).await;

    // Check if ClickHouse is ok - BatchModelInference Table
    let clickhouse = get_clickhouse().await;
    let result = select_batch_model_inference_clickhouse(&clickhouse, inference_id)
        .await
        .unwrap();

    println!("ClickHouse - BatchModelInference: {result:#?}");

    let id = result.get("id").unwrap().as_str().unwrap();
    let id = Uuid::parse_str(id).unwrap();
    assert_eq!(id, inference_id);

    let retrieved_batch_id = result.get("batch_id").unwrap().as_str().unwrap();
    let retrieved_batch_id = Uuid::parse_str(retrieved_batch_id).unwrap();
    assert_eq!(retrieved_batch_id, batch_id);

    let function_name = result.get("function_name").unwrap().as_str().unwrap();
    assert_eq!(function_name, payload["function_name"]);

    let variant_name = result.get("variant_name").unwrap().as_str().unwrap();
    assert_eq!(variant_name, provider.variant_name);

    let retrieved_episode_id = result.get("episode_id").unwrap().as_str().unwrap();
    let retrieved_episode_id = Uuid::parse_str(retrieved_episode_id).unwrap();
    assert_eq!(retrieved_episode_id, episode_id);

    let input: Value =
        serde_json::from_str(result.get("input").unwrap().as_str().unwrap()).unwrap();
    let correct_input = json!({
        "system": {"assistant_name": "Dr. Mehta"},
        "messages": [
            {
                "role": "user",
                "content": [{"type": "text", "value": "What is the weather like in Tokyo (in Celsius)? Use the `get_temperature` tool."}]
            },
            {
                "role": "assistant",
                "content": [{"type": "tool_call", "name": "get_temperature", "arguments": "{\"location\": \"Tokyo\"}", "id": "123456789"}]
            },
            {
                "role": "user",
                "content": [{"type": "tool_result", "name": "get_temperature", "result": "70", "id": "123456789"}]
            }
        ]
    });
    assert_eq!(input, correct_input);

    let input_messages = result.get("input_messages").unwrap().as_str().unwrap();
    let input_messages: Vec<RequestMessage> = serde_json::from_str(input_messages).unwrap();
    let expected_input_messages = vec![
        RequestMessage {
            role: Role::User,
            content: vec![
                "What is the weather like in Tokyo (in Celsius)? Use the `get_temperature` tool."
                    .to_string()
                    .into(),
            ],
        },
        RequestMessage {
            role: Role::Assistant,
            content: vec![ContentBlock::ToolCall(ToolCall {
                name: "get_temperature".to_string(),
                arguments: "{\"location\": \"Tokyo\"}".to_string(),
                id: "123456789".to_string(),
            })],
        },
        RequestMessage {
            role: Role::User,
            content: vec![ContentBlock::ToolResult(ToolResult {
                name: "get_temperature".to_string(),
                result: "70".to_string(),
                id: "123456789".to_string(),
            })],
        },
    ];
    assert_eq!(input_messages, expected_input_messages);

    let system = result.get("system").unwrap().as_str().unwrap();
    assert_eq!(
        system,
        "You are a helpful and friendly assistant named Dr. Mehta.\n\nPeople will ask you questions about the weather.\n\nIf asked about the weather, just respond with the tool call. Use the \"get_temperature\" tool.\n\nIf provided with a tool result, use it to respond to the user (e.g. \"The weather in New York is 55 degrees Fahrenheit.\")."
    );

    let tool_params = result.get("tool_params").unwrap().as_str().unwrap();
    let tool_params: Value = serde_json::from_str(tool_params).unwrap();
    let expected_tool_params = json!({"tools_available":[{"description":"Get the current temperature in a given location","parameters":{"$schema":"http://json-schema.org/draft-07/schema#","type":"object","properties":{"location":{"type":"string","description":"The location to get the temperature for (e.g. \"New York\")"},"units":{"type":"string","description":"The units to get the temperature in (must be \"fahrenheit\" or \"celsius\")","enum":["fahrenheit","celsius"]}},"required":["location"],"additionalProperties":false},"name":"get_temperature","strict":false}],"tool_choice":"auto","parallel_tool_calls":false});
    assert_eq!(tool_params, expected_tool_params);

    let inference_params = result.get("inference_params").unwrap().as_str().unwrap();
    let inference_params: Value = serde_json::from_str(inference_params).unwrap();
    let inference_params = inference_params.get("chat_completion").unwrap();
    assert!(inference_params.get("temperature").is_none());
    assert!(inference_params.get("seed").is_none());
    assert_eq!(
        inference_params
            .get("max_tokens")
            .unwrap()
            .as_u64()
            .unwrap(),
        100
    );

    let model_name = result.get("model_name").unwrap().as_str().unwrap();
    assert_eq!(model_name, provider.model_name);

    let model_provider_name = result.get("model_provider_name").unwrap().as_str().unwrap();
    assert_eq!(model_provider_name, provider.model_provider_name);

    let output_schema = result.get("output_schema");
    assert_eq!(output_schema, Some(&Value::Null));

    let tags = result.get("tags").unwrap().as_object().unwrap();
    assert_eq!(tags.len(), 1);
    assert_eq!(tags.get("test").unwrap().as_str().unwrap(), "multi_turn");

    // Check if ClickHouse is ok - BatchRequest Table
    let result = select_latest_batch_request_clickhouse(&clickhouse, batch_id)
        .await
        .unwrap();

    println!("ClickHouse - BatchRequest: {result:#?}");

    let id = result.get("id").unwrap().as_str().unwrap();
    Uuid::parse_str(id).unwrap();

    let retrieved_batch_id = result.get("batch_id").unwrap().as_str().unwrap();
    let retrieved_batch_id = Uuid::parse_str(retrieved_batch_id).unwrap();
    assert_eq!(retrieved_batch_id, batch_id);
    let batch_params = result.get("batch_params").unwrap().as_str().unwrap();
    let _batch_params: Value = serde_json::from_str(batch_params).unwrap();
    // We can't check that the batch params are exactly the same because they vary per-provider
    // We will check that they are valid by using them instead.
    let model_name = result.get("model_name").unwrap().as_str().unwrap();
    assert_eq!(model_name, provider.model_name);

    let model_provider_name = result.get("model_provider_name").unwrap().as_str().unwrap();
    assert_eq!(model_provider_name, provider.model_provider_name);

    let status = result.get("status").unwrap().as_str().unwrap();
    assert_eq!(status, "pending");

    let errors = result.get("errors").unwrap().as_object().unwrap();
    assert_eq!(errors.len(), 0);
}

pub async fn test_dynamic_tool_use_batch_inference_request_with_provider(
    provider: E2ETestProvider,
) {
    let episode_id = Uuid::now_v7();

    let payload = json!({
        "function_name": "basic_test",
        "episode_ids": [episode_id],
        "inputs":[{
            "system": {"assistant_name": "Dr. Mehta"},
            "messages": [
                {
                    "role": "user",
                    "content": "What is the weather like in Tokyo (in Celsius)? Use the provided `get_temperature` tool. Do not say anything else, just call the function."
                }
            ]}],
        "additional_tools": [[
            {
                "name": "get_temperature",
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
                }
            }
        ]],
        "variant_name": provider.variant_name,
    });

    let response = Client::new()
        .post(get_gateway_endpoint("/batch_inference"))
        .json(&payload)
        .send()
        .await
        .unwrap();

    // Check if the API response is fine
    assert_eq!(response.status(), StatusCode::OK);
    let response_json = response.json::<Value>().await.unwrap();

    println!("API response: {response_json:#?}");
    let batch_id = response_json.get("batch_id").unwrap().as_str().unwrap();
    let batch_id = Uuid::parse_str(batch_id).unwrap();

    let inference_ids = response_json
        .get("inference_ids")
        .unwrap()
        .as_array()
        .unwrap();
    assert_eq!(inference_ids.len(), 1);
    let inference_id = inference_ids.first().unwrap().as_str().unwrap();
    let inference_id = Uuid::parse_str(inference_id).unwrap();

    let episode_ids = response_json
        .get("episode_ids")
        .unwrap()
        .as_array()
        .unwrap();
    assert_eq!(episode_ids.len(), 1);
    let returned_episode_id = episode_ids.first().unwrap().as_str().unwrap();
    let returned_episode_id = Uuid::parse_str(returned_episode_id).unwrap();
    assert_eq!(returned_episode_id, episode_id);

    // Sleep to allow time for data to be inserted into ClickHouse (trailing writes from API)
    tokio::time::sleep(std::time::Duration::from_millis(500)).await;

    // Check if ClickHouse is ok - BatchModelInference Table
    let clickhouse = get_clickhouse().await;
    let result = select_batch_model_inference_clickhouse(&clickhouse, inference_id)
        .await
        .unwrap();

    println!("ClickHouse - BatchModelInference: {result:#?}");

    let id = result.get("id").unwrap().as_str().unwrap();
    let id = Uuid::parse_str(id).unwrap();
    assert_eq!(id, inference_id);

    let retrieved_batch_id = result.get("batch_id").unwrap().as_str().unwrap();
    let retrieved_batch_id = Uuid::parse_str(retrieved_batch_id).unwrap();
    assert_eq!(retrieved_batch_id, batch_id);

    let function_name = result.get("function_name").unwrap().as_str().unwrap();
    assert_eq!(function_name, payload["function_name"]);

    let variant_name = result.get("variant_name").unwrap().as_str().unwrap();
    assert_eq!(variant_name, provider.variant_name);

    let retrieved_episode_id = result.get("episode_id").unwrap().as_str().unwrap();
    let retrieved_episode_id = Uuid::parse_str(retrieved_episode_id).unwrap();
    assert_eq!(retrieved_episode_id, episode_id);

    let input: Value =
        serde_json::from_str(result.get("input").unwrap().as_str().unwrap()).unwrap();
    let correct_input = json!({
        "system": {"assistant_name": "Dr. Mehta"},
        "messages": [
            {
                "role": "user",
                "content": [{"type": "text", "value": "What is the weather like in Tokyo (in Celsius)? Use the provided `get_temperature` tool. Do not say anything else, just call the function."}]
            }
        ]
    });
    assert_eq!(input, correct_input);

    let input_messages = result.get("input_messages").unwrap().as_str().unwrap();
    let input_messages: Vec<RequestMessage> = serde_json::from_str(input_messages).unwrap();
    let expected_input_messages = vec![RequestMessage {
        role: Role::User,
        content: vec!["What is the weather like in Tokyo (in Celsius)? Use the provided `get_temperature` tool. Do not say anything else, just call the function.".to_string().into()],
    }];
    assert_eq!(input_messages, expected_input_messages);

    let system = result.get("system").unwrap().as_str().unwrap();
    assert_eq!(
        system,
        "You are a helpful and friendly assistant named Dr. Mehta"
    );

    let tool_params = result.get("tool_params").unwrap().as_str().unwrap();
    let tool_params: Value = serde_json::from_str(tool_params).unwrap();
    let expected_tool_params = json!({
        "tools_available": [{
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
        }],
        "tool_choice": "auto",
        "parallel_tool_calls": false
    });
    assert_eq!(tool_params, expected_tool_params);

    let inference_params = result.get("inference_params").unwrap().as_str().unwrap();
    let inference_params: Value = serde_json::from_str(inference_params).unwrap();
    let inference_params = inference_params.get("chat_completion").unwrap();
    let max_tokens = inference_params
        .get("max_tokens")
        .unwrap()
        .as_u64()
        .unwrap();
    assert_eq!(max_tokens, 100);

    let model_name = result.get("model_name").unwrap().as_str().unwrap();
    assert_eq!(model_name, provider.model_name);

    let model_provider_name = result.get("model_provider_name").unwrap().as_str().unwrap();
    assert_eq!(model_provider_name, provider.model_provider_name);

    let output_schema = result.get("output_schema");
    assert_eq!(output_schema, Some(&Value::Null));

    let tags = result.get("tags").unwrap().as_object().unwrap();
    assert_eq!(tags.len(), 0);

    // Check if ClickHouse is ok - BatchRequest Table
    let result = select_latest_batch_request_clickhouse(&clickhouse, batch_id)
        .await
        .unwrap();

    println!("ClickHouse - BatchRequest: {result:#?}");

    let id = result.get("id").unwrap().as_str().unwrap();
    Uuid::parse_str(id).unwrap();

    let retrieved_batch_id = result.get("batch_id").unwrap().as_str().unwrap();
    let retrieved_batch_id = Uuid::parse_str(retrieved_batch_id).unwrap();
    assert_eq!(retrieved_batch_id, batch_id);
    let batch_params = result.get("batch_params").unwrap().as_str().unwrap();
    let _batch_params: Value = serde_json::from_str(batch_params).unwrap();
    // We can't check that the batch params are exactly the same because they vary per-provider
    // We will check that they are valid by using them instead.
    let model_name = result.get("model_name").unwrap().as_str().unwrap();
    assert_eq!(model_name, provider.model_name);

    let model_provider_name = result.get("model_provider_name").unwrap().as_str().unwrap();
    assert_eq!(model_provider_name, provider.model_provider_name);

    let status = result.get("status").unwrap().as_str().unwrap();
    assert_eq!(status, "pending");

    let errors = result.get("errors").unwrap().as_object().unwrap();
    assert_eq!(errors.len(), 0);
}

pub async fn test_parallel_tool_use_batch_inference_request_with_provider(
    provider: E2ETestProvider,
) {
    let episode_id = Uuid::now_v7();

    let payload = json!({
        "function_name": "weather_helper_parallel",
        "episode_ids": [episode_id],
        "inputs":[{
            "system": {"assistant_name": "Dr. Mehta"},
            "messages": [
                {
                    "role": "user",
                    "content": "What is the weather like in Tokyo (in Celsius)? Use both the provided `get_temperature` and `get_humidity` tools. Do not say anything else, just call the two functions."
                }
            ]}],
        "parallel_tool_calls": [true],
        "variant_name": provider.variant_name,
    });

    let response = Client::new()
        .post(get_gateway_endpoint("/batch_inference"))
        .json(&payload)
        .send()
        .await
        .unwrap();

    // Check if the API response is fine
    assert_eq!(response.status(), StatusCode::OK);
    let response_json = response.json::<Value>().await.unwrap();

    println!("API response: {response_json:#?}");
    let batch_id = response_json.get("batch_id").unwrap().as_str().unwrap();
    let batch_id = Uuid::parse_str(batch_id).unwrap();

    let inference_ids = response_json
        .get("inference_ids")
        .unwrap()
        .as_array()
        .unwrap();
    assert_eq!(inference_ids.len(), 1);
    let inference_id = inference_ids.first().unwrap().as_str().unwrap();
    let inference_id = Uuid::parse_str(inference_id).unwrap();

    let episode_ids = response_json
        .get("episode_ids")
        .unwrap()
        .as_array()
        .unwrap();
    assert_eq!(episode_ids.len(), 1);
    let returned_episode_id = episode_ids.first().unwrap().as_str().unwrap();
    let returned_episode_id = Uuid::parse_str(returned_episode_id).unwrap();
    assert_eq!(returned_episode_id, episode_id);

    // Sleep to allow time for data to be inserted into ClickHouse (trailing writes from API)
    tokio::time::sleep(std::time::Duration::from_millis(500)).await;

    // Check if ClickHouse is ok - BatchModelInference Table
    let clickhouse = get_clickhouse().await;
    let result = select_batch_model_inference_clickhouse(&clickhouse, inference_id)
        .await
        .unwrap();

    println!("ClickHouse - BatchModelInference: {result:#?}");

    let id = result.get("id").unwrap().as_str().unwrap();
    let id = Uuid::parse_str(id).unwrap();
    assert_eq!(id, inference_id);

    let retrieved_batch_id = result.get("batch_id").unwrap().as_str().unwrap();
    let retrieved_batch_id = Uuid::parse_str(retrieved_batch_id).unwrap();
    assert_eq!(retrieved_batch_id, batch_id);

    let function_name = result.get("function_name").unwrap().as_str().unwrap();
    assert_eq!(function_name, payload["function_name"]);

    let variant_name = result.get("variant_name").unwrap().as_str().unwrap();
    assert_eq!(variant_name, provider.variant_name);

    let retrieved_episode_id = result.get("episode_id").unwrap().as_str().unwrap();
    let retrieved_episode_id = Uuid::parse_str(retrieved_episode_id).unwrap();
    assert_eq!(retrieved_episode_id, episode_id);

    let input: Value =
        serde_json::from_str(result.get("input").unwrap().as_str().unwrap()).unwrap();
    let correct_input = json!({
        "system": {"assistant_name": "Dr. Mehta"},
        "messages": [
            {
                "role": "user",
                "content": [{"type": "text", "value": "What is the weather like in Tokyo (in Celsius)? Use both the provided `get_temperature` and `get_humidity` tools. Do not say anything else, just call the two functions."}]
            }
        ]
    });
    assert_eq!(input, correct_input);

    let input_messages = result.get("input_messages").unwrap().as_str().unwrap();
    let input_messages: Vec<RequestMessage> = serde_json::from_str(input_messages).unwrap();
    let expected_input_messages = vec![RequestMessage {
        role: Role::User,
        content: vec!["What is the weather like in Tokyo (in Celsius)? Use both the provided `get_temperature` and `get_humidity` tools. Do not say anything else, just call the two functions.".to_string().into()],
    }];
    assert_eq!(input_messages, expected_input_messages);

    let system = result.get("system").unwrap().as_str().unwrap();
    assert_eq!(
        system,
        "You are a helpful and friendly assistant named Dr. Mehta.\n\nPeople will ask you questions about the weather.\n\nIf asked about the weather, just respond with two tool calls. Use BOTH the \"get_temperature\" and \"get_humidity\" tools.\n\nIf provided with a tool result, use it to respond to the user (e.g. \"The weather in New York is 55 degrees Fahrenheit with 50% humidity.\")."
    );

    let tool_params = result.get("tool_params").unwrap().as_str().unwrap();
    let tool_params: Value = serde_json::from_str(tool_params).unwrap();
    let expected_tool_params = json!({
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
            },
            {
                "description": "Get the current humidity in a given location",
                "parameters": {
                    "$schema": "http://json-schema.org/draft-07/schema#",
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The location to get the humidity for (e.g. \"New York\")"
                        }
                    },
                    "required": ["location"],
                    "additionalProperties": false
                },
                "name": "get_humidity",
                "strict": false
            }
        ],
        "tool_choice": "auto",
        "parallel_tool_calls": true
    });
    assert_eq!(tool_params, expected_tool_params);

    let inference_params = result.get("inference_params").unwrap().as_str().unwrap();
    let inference_params: Value = serde_json::from_str(inference_params).unwrap();
    let inference_params = inference_params.get("chat_completion").unwrap();
    let max_tokens = inference_params
        .get("max_tokens")
        .unwrap()
        .as_u64()
        .unwrap();
    assert_eq!(max_tokens, 100);

    let model_name = result.get("model_name").unwrap().as_str().unwrap();
    assert_eq!(model_name, provider.model_name);

    let model_provider_name = result.get("model_provider_name").unwrap().as_str().unwrap();
    assert_eq!(model_provider_name, provider.model_provider_name);

    let output_schema = result.get("output_schema");
    assert_eq!(output_schema, Some(&Value::Null));

    let tags = result.get("tags").unwrap().as_object().unwrap();
    assert_eq!(tags.len(), 0);

    // Check if ClickHouse is ok - BatchRequest Table
    let result = select_latest_batch_request_clickhouse(&clickhouse, batch_id)
        .await
        .unwrap();

    println!("ClickHouse - BatchRequest: {result:#?}");

    let id = result.get("id").unwrap().as_str().unwrap();
    Uuid::parse_str(id).unwrap();

    let retrieved_batch_id = result.get("batch_id").unwrap().as_str().unwrap();
    let retrieved_batch_id = Uuid::parse_str(retrieved_batch_id).unwrap();
    assert_eq!(retrieved_batch_id, batch_id);
    let batch_params = result.get("batch_params").unwrap().as_str().unwrap();
    let _batch_params: Value = serde_json::from_str(batch_params).unwrap();
    // We can't check that the batch params are exactly the same because they vary per-provider
    // We will check that they are valid by using them instead.
    let model_name = result.get("model_name").unwrap().as_str().unwrap();
    assert_eq!(model_name, provider.model_name);

    let model_provider_name = result.get("model_provider_name").unwrap().as_str().unwrap();
    assert_eq!(model_provider_name, provider.model_provider_name);

    let status = result.get("status").unwrap().as_str().unwrap();
    assert_eq!(status, "pending");

    let errors = result.get("errors").unwrap().as_object().unwrap();
    assert_eq!(errors.len(), 0);
}

pub async fn test_json_mode_batch_inference_request_with_provider(provider: E2ETestProvider) {
    let episode_id = Uuid::now_v7();

    let payload = json!({
        "function_name": "json_success",
        "variant_name": provider.variant_name,
        "episode_ids": [episode_id],
        "inputs": [{
            "system": {"assistant_name": "Dr. Mehta"},
               "messages": [
                {
                    "role": "user",
                    "content": {"country": "Japan"}
                }
            ]}],
    });

    let response = Client::new()
        .post(get_gateway_endpoint("/batch_inference"))
        .json(&payload)
        .send()
        .await
        .unwrap();
    // Check if the API response is fine
    assert_eq!(response.status(), StatusCode::OK);
    let response_json = response.json::<Value>().await.unwrap();

    println!("API response: {response_json:#?}");
    let batch_id = response_json.get("batch_id").unwrap().as_str().unwrap();
    let batch_id = Uuid::parse_str(batch_id).unwrap();

    let inference_ids = response_json
        .get("inference_ids")
        .unwrap()
        .as_array()
        .unwrap();
    assert_eq!(inference_ids.len(), 1);
    let inference_id = inference_ids.first().unwrap().as_str().unwrap();
    let inference_id = Uuid::parse_str(inference_id).unwrap();

    let episode_ids = response_json
        .get("episode_ids")
        .unwrap()
        .as_array()
        .unwrap();
    assert_eq!(episode_ids.len(), 1);
    let returned_episode_id = episode_ids.first().unwrap().as_str().unwrap();
    let returned_episode_id = Uuid::parse_str(returned_episode_id).unwrap();
    assert_eq!(returned_episode_id, episode_id);

    // Sleep to allow time for data to be inserted into ClickHouse (trailing writes from API)
    tokio::time::sleep(std::time::Duration::from_millis(500)).await;

    // Check if ClickHouse is ok - BatchModelInference Table
    let clickhouse = get_clickhouse().await;
    let result = select_batch_model_inference_clickhouse(&clickhouse, inference_id)
        .await
        .unwrap();

    println!("ClickHouse - BatchModelInference: {result:#?}");

    let id = result.get("id").unwrap().as_str().unwrap();
    let id = Uuid::parse_str(id).unwrap();
    assert_eq!(id, inference_id);

    let retrieved_batch_id = result.get("batch_id").unwrap().as_str().unwrap();
    let retrieved_batch_id = Uuid::parse_str(retrieved_batch_id).unwrap();
    assert_eq!(retrieved_batch_id, batch_id);

    let function_name = result.get("function_name").unwrap().as_str().unwrap();
    assert_eq!(function_name, payload["function_name"]);

    let variant_name = result.get("variant_name").unwrap().as_str().unwrap();
    assert_eq!(variant_name, provider.variant_name);

    let retrieved_episode_id = result.get("episode_id").unwrap().as_str().unwrap();
    let retrieved_episode_id = Uuid::parse_str(retrieved_episode_id).unwrap();
    assert_eq!(retrieved_episode_id, episode_id);

    let input: Value =
        serde_json::from_str(result.get("input").unwrap().as_str().unwrap()).unwrap();
    let correct_input = json!({
        "system": {"assistant_name": "Dr. Mehta"},
        "messages": [
            {
                "role": "user",
                "content": [{"type": "text", "value": {"country": "Japan"}}]
            }
        ]
    });
    assert_eq!(input, correct_input);

    let input_messages = result.get("input_messages").unwrap().as_str().unwrap();
    let input_messages: Vec<RequestMessage> = serde_json::from_str(input_messages).unwrap();
    let expected_input_messages = vec![RequestMessage {
        role: Role::User,
        content: vec!["What is the capital city of Japan?".to_string().into()],
    }];
    assert_eq!(input_messages, expected_input_messages);

    let system = result.get("system").unwrap().as_str().unwrap();
    assert_eq!(
        system,
        "You are a helpful and friendly assistant named Dr. Mehta.\n\nPlease answer the questions in a JSON with key \"answer\".\n\nDo not include any other text than the JSON object. Do not include \"```json\" or \"```\" or anything else.\n\nExample Response:\n\n{\n    \"answer\": \"42\"\n}"
    );

    assert!(result.get("tool_params").unwrap().is_null());
    let inference_params = result.get("inference_params").unwrap().as_str().unwrap();
    let inference_params: Value = serde_json::from_str(inference_params).unwrap();
    let inference_params = inference_params.get("chat_completion").unwrap();
    let max_tokens = inference_params
        .get("max_tokens")
        .unwrap()
        .as_u64()
        .unwrap();
    assert_eq!(max_tokens, 100);

    let model_name = result.get("model_name").unwrap().as_str().unwrap();
    assert_eq!(model_name, provider.model_name);

    let model_provider_name = result.get("model_provider_name").unwrap().as_str().unwrap();
    assert_eq!(model_provider_name, provider.model_provider_name);

    let output_schema = result.get("output_schema").unwrap().as_str().unwrap();
    let output_schema: Value = serde_json::from_str(output_schema).unwrap();
    let expected_output_schema = json!({
        "type": "object",
        "properties": {
            "answer": {"type": "string"}
        },
        "required": ["answer"],
        "additionalProperties": false
    });
    assert_eq!(output_schema, expected_output_schema);

    let tags = result.get("tags").unwrap().as_object().unwrap();
    assert_eq!(tags.len(), 0);

    // Check if ClickHouse is ok - BatchRequest Table
    let result = select_latest_batch_request_clickhouse(&clickhouse, batch_id)
        .await
        .unwrap();

    println!("ClickHouse - BatchRequest: {result:#?}");

    let id = result.get("id").unwrap().as_str().unwrap();
    Uuid::parse_str(id).unwrap();

    let retrieved_batch_id = result.get("batch_id").unwrap().as_str().unwrap();
    let retrieved_batch_id = Uuid::parse_str(retrieved_batch_id).unwrap();
    assert_eq!(retrieved_batch_id, batch_id);
    let batch_params = result.get("batch_params").unwrap().as_str().unwrap();
    let _batch_params: Value = serde_json::from_str(batch_params).unwrap();
    // We can't check that the batch params are exactly the same because they vary per-provider
    // We will check that they are valid by using them instead.
    let model_name = result.get("model_name").unwrap().as_str().unwrap();
    assert_eq!(model_name, provider.model_name);

    let model_provider_name = result.get("model_provider_name").unwrap().as_str().unwrap();
    assert_eq!(model_provider_name, provider.model_provider_name);

    let status = result.get("status").unwrap().as_str().unwrap();
    assert_eq!(status, "pending");

    let errors = result.get("errors").unwrap().as_object().unwrap();
    assert_eq!(errors.len(), 0);
}

pub async fn test_dynamic_json_mode_batch_inference_request_with_provider(
    provider: E2ETestProvider,
) {
    let episode_id = Uuid::now_v7();
    let output_schema = json!({
      "type": "object",
      "properties": {
        "response": {
          "type": "string"
        }
      },
      "required": ["response"],
      "additionalProperties": false
    });
    let serialized_output_schema = serde_json::to_string(&output_schema).unwrap();

    let payload = json!({
        "function_name": "dynamic_json",
        "variant_name": provider.variant_name,
        "episode_ids": [episode_id],
        "inputs": [
            {
               "system": {"assistant_name": "Dr. Mehta", "schema": serialized_output_schema},
               "messages": [
                {
                    "role": "user",
                    "content": {"country": "Japan"}
                }
            ]}],
        "output_schemas": [output_schema.clone()],
    });

    let response = Client::new()
        .post(get_gateway_endpoint("/batch_inference"))
        .json(&payload)
        .send()
        .await
        .unwrap();
    // Check if the API response is fine
    assert_eq!(response.status(), StatusCode::OK);
    let response_json = response.json::<Value>().await.unwrap();

    println!("API response: {response_json:#?}");
    let batch_id = response_json.get("batch_id").unwrap().as_str().unwrap();
    let batch_id = Uuid::parse_str(batch_id).unwrap();

    let inference_ids = response_json
        .get("inference_ids")
        .unwrap()
        .as_array()
        .unwrap();
    assert_eq!(inference_ids.len(), 1);
    let inference_id = inference_ids.first().unwrap().as_str().unwrap();
    let inference_id = Uuid::parse_str(inference_id).unwrap();

    let episode_ids = response_json
        .get("episode_ids")
        .unwrap()
        .as_array()
        .unwrap();
    assert_eq!(episode_ids.len(), 1);
    let returned_episode_id = episode_ids.first().unwrap().as_str().unwrap();
    let returned_episode_id = Uuid::parse_str(returned_episode_id).unwrap();
    assert_eq!(returned_episode_id, episode_id);

    // Sleep to allow time for data to be inserted into ClickHouse (trailing writes from API)
    tokio::time::sleep(std::time::Duration::from_millis(500)).await;

    // Check if ClickHouse is ok - BatchModelInference Table
    let clickhouse = get_clickhouse().await;
    let result = select_batch_model_inference_clickhouse(&clickhouse, inference_id)
        .await
        .unwrap();

    println!("ClickHouse - BatchModelInference: {result:#?}");

    let id = result.get("id").unwrap().as_str().unwrap();
    let id = Uuid::parse_str(id).unwrap();
    assert_eq!(id, inference_id);

    let retrieved_batch_id = result.get("batch_id").unwrap().as_str().unwrap();
    let retrieved_batch_id = Uuid::parse_str(retrieved_batch_id).unwrap();
    assert_eq!(retrieved_batch_id, batch_id);

    let function_name = result.get("function_name").unwrap().as_str().unwrap();
    assert_eq!(function_name, payload["function_name"]);

    let variant_name = result.get("variant_name").unwrap().as_str().unwrap();
    assert_eq!(variant_name, provider.variant_name);

    let retrieved_episode_id = result.get("episode_id").unwrap().as_str().unwrap();
    let retrieved_episode_id = Uuid::parse_str(retrieved_episode_id).unwrap();
    assert_eq!(retrieved_episode_id, episode_id);

    let input: Value =
        serde_json::from_str(result.get("input").unwrap().as_str().unwrap()).unwrap();
    let correct_input = json!({
        "system": {"assistant_name": "Dr. Mehta", "schema": serialized_output_schema},
        "messages": [
            {
                "role": "user",
                "content": [{"type": "text", "value": {"country": "Japan"}}]
            }
        ]
    });
    assert_eq!(input, correct_input);

    let input_messages = result.get("input_messages").unwrap().as_str().unwrap();
    let input_messages: Vec<RequestMessage> = serde_json::from_str(input_messages).unwrap();
    let expected_input_messages = vec![RequestMessage {
        role: Role::User,
        content: vec!["What is the capital city of Japan?".to_string().into()],
    }];
    assert_eq!(input_messages, expected_input_messages);

    let system = result.get("system").unwrap().as_str().unwrap();
    assert_eq!(
        system,
        "You are a helpful and friendly assistant named Dr. Mehta.\n\nDo not include any other text than the JSON object.  Do not include \"```json\" or \"```\" or anything else.\n\nPlease answer the questions in a JSON with the following schema:\n\n{\"type\":\"object\",\"properties\":{\"response\":{\"type\":\"string\"}},\"required\":[\"response\"],\"additionalProperties\":false}"
    );

    assert!(result.get("tool_params").unwrap().is_null());
    let inference_params = result.get("inference_params").unwrap().as_str().unwrap();
    let inference_params: Value = serde_json::from_str(inference_params).unwrap();
    let inference_params = inference_params.get("chat_completion").unwrap();
    let max_tokens = inference_params
        .get("max_tokens")
        .unwrap()
        .as_u64()
        .unwrap();
    assert_eq!(max_tokens, 100);

    let model_name = result.get("model_name").unwrap().as_str().unwrap();
    assert_eq!(model_name, provider.model_name);

    let model_provider_name = result.get("model_provider_name").unwrap().as_str().unwrap();
    assert_eq!(model_provider_name, provider.model_provider_name);

    let output_schema = result.get("output_schema").unwrap().as_str().unwrap();
    let output_schema: Value = serde_json::from_str(output_schema).unwrap();
    let expected_output_schema = json!({
        "type": "object",
        "properties": {
            "response": {"type": "string"}
        },
        "required": ["response"],
        "additionalProperties": false
    });
    assert_eq!(output_schema, expected_output_schema);

    let tags = result.get("tags").unwrap().as_object().unwrap();
    assert_eq!(tags.len(), 0);

    // Check if ClickHouse is ok - BatchRequest Table
    let result = select_latest_batch_request_clickhouse(&clickhouse, batch_id)
        .await
        .unwrap();

    println!("ClickHouse - BatchRequest: {result:#?}");

    let id = result.get("id").unwrap().as_str().unwrap();
    Uuid::parse_str(id).unwrap();

    let retrieved_batch_id = result.get("batch_id").unwrap().as_str().unwrap();
    let retrieved_batch_id = Uuid::parse_str(retrieved_batch_id).unwrap();
    assert_eq!(retrieved_batch_id, batch_id);
    let batch_params = result.get("batch_params").unwrap().as_str().unwrap();
    let _batch_params: Value = serde_json::from_str(batch_params).unwrap();
    // We can't check that the batch params are exactly the same because they vary per-provider
    // We will check that they are valid by using them instead.
    let model_name = result.get("model_name").unwrap().as_str().unwrap();
    assert_eq!(model_name, provider.model_name);

    let model_provider_name = result.get("model_provider_name").unwrap().as_str().unwrap();
    assert_eq!(model_provider_name, provider.model_provider_name);

    let status = result.get("status").unwrap().as_str().unwrap();
    assert_eq!(status, "pending");

    let errors = result.get("errors").unwrap().as_object().unwrap();
    assert_eq!(errors.len(), 0);
}
