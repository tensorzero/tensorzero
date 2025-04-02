#![allow(clippy::print_stdout)]
use std::collections::HashMap;

use futures::StreamExt;
use reqwest::{Client, StatusCode};
use reqwest_eventsource::{Event, RequestBuilderExt};
use serde_json::{json, Value};
use uuid::Uuid;

use crate::{
    common::get_gateway_endpoint,
    providers::common::{E2ETestProvider, E2ETestProviders},
};
use tensorzero_internal::clickhouse::test_helpers::{
    get_clickhouse, select_chat_inference_clickhouse, select_model_inference_clickhouse,
};

crate::generate_provider_tests!(get_providers);
crate::generate_batch_inference_tests!(get_providers);

async fn get_providers() -> E2ETestProviders {
    let credentials = match std::env::var("ANTHROPIC_API_KEY") {
        Ok(key) => HashMap::from([("anthropic_api_key".to_string(), key)]),
        Err(_) => HashMap::new(),
    };

    let standard_providers = vec![E2ETestProvider {
        variant_name: "anthropic".to_string(),
        model_name: "claude-3-haiku-20240307-anthropic".into(),
        model_provider_name: "anthropic".into(),
        credentials: HashMap::new(),
    }];

    let image_providers = vec![E2ETestProvider {
        variant_name: "anthropic".to_string(),
        model_name: "anthropic::claude-3-haiku-20240307".into(),
        model_provider_name: "anthropic".into(),
        credentials: HashMap::new(),
    }];

    let extra_body_providers = vec![E2ETestProvider {
        variant_name: "anthropic-extra-body".to_string(),
        model_name: "claude-3-haiku-20240307-anthropic".into(),
        model_provider_name: "anthropic".into(),
        credentials: HashMap::new(),
    }];

    let bad_auth_extra_headers = vec![E2ETestProvider {
        variant_name: "anthropic-extra-headers".to_string(),
        model_name: "claude-3-haiku-20240307-anthropic".into(),
        model_provider_name: "anthropic".into(),
        credentials: HashMap::new(),
    }];

    let inference_params_dynamic_providers = vec![E2ETestProvider {
        variant_name: "anthropic-dynamic".to_string(),
        model_name: "claude-3-haiku-20240307-anthropic-dynamic".into(),
        model_provider_name: "anthropic".into(),
        credentials,
    }];

    let json_providers = vec![
        E2ETestProvider {
            variant_name: "anthropic".to_string(),
            model_name: "claude-3-haiku-20240307-anthropic".into(),
            model_provider_name: "anthropic".into(),
            credentials: HashMap::new(),
        },
        E2ETestProvider {
            variant_name: "anthropic-implicit".to_string(),
            model_name: "claude-3-haiku-20240307-anthropic".into(),
            model_provider_name: "anthropic".into(),
            credentials: HashMap::new(),
        },
        E2ETestProvider {
            variant_name: "anthropic-default".to_string(),
            model_name: "claude-3-haiku-20240307-anthropic".into(),
            model_provider_name: "anthropic".into(),
            credentials: HashMap::new(),
        },
    ];

    let shorthand_providers = vec![E2ETestProvider {
        variant_name: "anthropic-shorthand".to_string(),
        model_name: "anthropic::claude-3-haiku-20240307".into(),
        model_provider_name: "anthropic".into(),
        credentials: HashMap::new(),
    }];

    E2ETestProviders {
        simple_inference: standard_providers.clone(),
        bad_auth_extra_headers,
        extra_body_inference: extra_body_providers,
        reasoning_inference: vec![],
        inference_params_inference: standard_providers.clone(),
        inference_params_dynamic_credentials: inference_params_dynamic_providers,
        tool_use_inference: standard_providers.clone(),
        tool_multi_turn_inference: standard_providers.clone(),
        dynamic_tool_use_inference: standard_providers.clone(),
        parallel_tool_use_inference: standard_providers.clone(),
        json_mode_inference: json_providers.clone(),
        image_inference: image_providers,

        shorthand_inference: shorthand_providers.clone(),
        supports_batch_inference: false,
    }
}

#[tokio::test]
async fn test_thinking_128k() {
    let client = Client::new();

    // This model uses a custom stop sequence, as we want to make sure that
    // we don't actually generate 128k tokens of output. This test just verifies
    // that we can pass through the necessary 'anthropic-beta' header to support
    // a large 'max_tokens'
    let payload = json!({
        "model_name": "claude-3-7-sonnet-20250219-thinking-128k",
        "input":{
            "messages": [
                {
                    "role": "user",
                    "content": "Output a haiku that ends in the word 'my_custom_stop'"
                }
            ]},
        "params": {
            "chat_completion": {
                "max_tokens": 128000,
            }
        },
        "stream": false,
    });

    let response = client
        .post(get_gateway_endpoint("/inference"))
        .json(&payload)
        .send()
        .await
        .unwrap();

    let status = response.status();
    let resp_json = response.json::<Value>().await.unwrap();
    assert_eq!(
        status,
        StatusCode::OK,
        "Unexpected status code with response: {resp_json}"
    );
    let content = resp_json.get("content").unwrap().as_array().unwrap();
    assert_eq!(content.len(), 2, "Unexpected content length: {content:?}");
    assert_eq!(content[0]["type"], "thought", "Unexpected content type");
    assert_eq!(content[1]["type"], "text", "Unexpected content type");
    assert!(
        !content[1]["text"]
            .as_str()
            .unwrap()
            .contains("my_custom_stop"),
        "Found my_custom_stop in content: {}",
        content[1]["text"].as_str().unwrap()
    );

    // We don't check the database, as we already do that in lots of places.
}

#[tokio::test]
async fn test_thinking_signature() {
    let client = Client::new();
    let episode_id = Uuid::now_v7();

    let payload = json!({
        "function_name": "weather_helper",
        "variant_name": "anthropic-thinking",
        "episode_id": episode_id,
        "input":{
            "system": {"assistant_name": "AskJeeves"},
            "messages": [
                {
                    "role": "user",
                    "content": "Hi I'm visiting Brooklyn from Brazil. What's the weather?"
                }
            ]},
        "stream": false,
    });

    let response = client
        .post(get_gateway_endpoint("/inference"))
        .json(&payload)
        .send()
        .await
        .unwrap();
    // Check Response is OK, then fields in order
    assert_eq!(response.status(), StatusCode::OK);
    let response_json = response.json::<Value>().await.unwrap();
    let content_blocks = response_json.get("content").unwrap().as_array().unwrap();
    let mut tensorzero_content_blocks = content_blocks.clone();
    assert!(
        content_blocks.len() == 3,
        "Unexpected content blocks: {:?}",
        content_blocks
    );
    let first_block = &content_blocks[0];
    let first_block_type = first_block.get("type").unwrap().as_str().unwrap();
    assert_eq!(first_block_type, "thought");
    assert!(
        first_block["text"]
            .as_str()
            .unwrap()
            .to_lowercase()
            .contains("weather"),
        "Thinking block should mention 'weather': {first_block}"
    );

    let second_block = &content_blocks[1];
    assert_eq!(second_block["type"], "text");
    let content = second_block.get("text").unwrap().as_str().unwrap();
    // Assert that weather is in the content
    assert!(
        content.contains("weather") || content.contains("temperature"),
        "Content should mention 'weather' or 'temperature': {second_block}"
    );
    // Check that inference_id is here
    let inference_id = response_json.get("inference_id").unwrap().as_str().unwrap();
    let inference_id = Uuid::parse_str(inference_id).unwrap();

    let third_block = &content_blocks[2];
    assert_eq!(third_block["type"], "tool_call");
    assert_eq!(third_block["name"], "get_temperature");
    println!("Third block: {third_block}");
    let tool_id = third_block.get("id").unwrap().as_str().unwrap();

    // Sleep for 1 second to allow time for data to be inserted into ClickHouse (trailing writes from API)
    tokio::time::sleep(std::time::Duration::from_secs(1)).await;

    // Check ClickHouse
    let clickhouse = get_clickhouse().await;

    // First, check Inference table
    let result = select_chat_inference_clickhouse(&clickhouse, inference_id)
        .await
        .unwrap();
    let id = result.get("id").unwrap().as_str().unwrap();
    let id_uuid = Uuid::parse_str(id).unwrap();
    assert_eq!(id_uuid, inference_id);
    let function_name = result.get("function_name").unwrap().as_str().unwrap();
    assert_eq!(function_name, "weather_helper");
    let input: Value =
        serde_json::from_str(result.get("input").unwrap().as_str().unwrap()).unwrap();
    let correct_input = json!({
        "system": {"assistant_name": "AskJeeves"},
        "messages": [
            {
                "role": "user",
                "content": [{"type": "text", "value": "Hi I'm visiting Brooklyn from Brazil. What's the weather?"}]
            }
        ]
    });
    assert_eq!(input, correct_input);
    let content_blocks = result.get("output").unwrap().as_str().unwrap();
    // Check that content_blocks is a list of blocks length 1
    let content_blocks: Vec<Value> = serde_json::from_str(content_blocks).unwrap();
    assert_eq!(content_blocks.len(), 3);
    let first_block = &content_blocks[0];
    // Check the type and content in the block
    assert_eq!(first_block["type"], "thought");
    let second_block = &content_blocks[1];
    assert_eq!(second_block["type"], "text");
    let clickhouse_content = second_block.get("text").unwrap().as_str().unwrap();
    assert_eq!(clickhouse_content, content);
    // Check that episode_id is here and correct
    let retrieved_episode_id = result.get("episode_id").unwrap().as_str().unwrap();
    let retrieved_episode_id = Uuid::parse_str(retrieved_episode_id).unwrap();
    assert_eq!(retrieved_episode_id, episode_id);
    // Check the variant name
    let variant_name = result.get("variant_name").unwrap().as_str().unwrap();
    assert_eq!(variant_name, "anthropic-thinking");
    // Check the processing time
    let processing_time_ms = result.get("processing_time_ms").unwrap().as_u64().unwrap();
    assert!(processing_time_ms > 0);

    // Check the ModelInference Table
    let result = select_model_inference_clickhouse(&clickhouse, inference_id)
        .await
        .unwrap();
    let inference_id_result = result.get("inference_id").unwrap().as_str().unwrap();
    let inference_id_result = Uuid::parse_str(inference_id_result).unwrap();
    assert_eq!(inference_id_result, inference_id);
    let model_name = result.get("model_name").unwrap().as_str().unwrap();
    assert_eq!(model_name, "claude-3-7-sonnet-20250219-thinking");
    let model_provider_name = result.get("model_provider_name").unwrap().as_str().unwrap();
    assert_eq!(model_provider_name, "anthropic-extra-body");
    let raw_request = result.get("raw_request").unwrap().as_str().unwrap();
    assert!(raw_request.to_lowercase().contains("weather"));
    // Check that raw_request is valid JSON
    let _: Value = serde_json::from_str(raw_request).expect("raw_request should be valid JSON");
    let input_tokens = result.get("input_tokens").unwrap().as_u64().unwrap();
    assert!(input_tokens > 5);
    let output_tokens = result.get("output_tokens").unwrap().as_u64().unwrap();
    assert!(output_tokens > 5);
    let response_time_ms = result.get("response_time_ms").unwrap().as_u64().unwrap();
    assert!(response_time_ms > 0);
    assert!(result.get("ttft_ms").unwrap().is_null());
    let raw_response = result.get("raw_response").unwrap().as_str().unwrap();
    let _raw_response_json: Value = serde_json::from_str(raw_response).unwrap();

    println!("Original tensorzero_content_blocks: {tensorzero_content_blocks:?}");

    // TODO - support passing tool call in directly (https://github.com/tensorzero/tensorzero/issues/1366)
    tensorzero_content_blocks[2]["arguments"] =
        Value::String(serde_json::to_string(&tensorzero_content_blocks[2]["arguments"]).unwrap());
    tensorzero_content_blocks[2]
        .as_object_mut()
        .unwrap()
        .remove("raw_arguments");
    tensorzero_content_blocks[2]
        .as_object_mut()
        .unwrap()
        .remove("raw_name");

    // Feed content blocks back in
    let mut new_messages = vec![
        serde_json::json!({
            "role": "user",
            "content": "Hi I'm visiting Brooklyn from Brazil. What's the weather?"
        }),
        serde_json::json!({
            "role": "assistant",
            "content": tensorzero_content_blocks,
        }),
    ];
    new_messages.push(serde_json::json!({
        "role": "user",
        "content": [{"type": "tool_result", "name": "My result", "result": "100", "id": tool_id}],
    }));

    println!("New messages: {new_messages:?}");

    let payload = json!({
        "model_name": "claude-3-7-sonnet-20250219-thinking",
        "episode_id": episode_id,
        "input": {
            "messages": new_messages
        },
        "stream": false,
    });

    let response = client
        .post(get_gateway_endpoint("/inference"))
        .json(&payload)
        .send()
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    let response_json = response.json::<Value>().await.unwrap();

    let content_blocks = response_json.get("content").unwrap().as_array().unwrap();
    assert!(
        content_blocks.len() == 1,
        "Unexpected content blocks: {:?}",
        content_blocks
    );
    assert_eq!(content_blocks[0]["type"], "text");
    assert!(
        content_blocks[0]["text"].as_str().unwrap().contains("100°"),
        "Content should mention '100°': {}",
        content_blocks[0]
    );
}

#[tokio::test]
async fn test_redacted_thinking() {
    let client = Client::new();
    let episode_id = Uuid::now_v7();

    let payload = json!({
        "model_name": "claude-3-7-sonnet-20250219-thinking",
        "episode_id": episode_id,
        "input": {
            "messages": [
                {
                    "role": "user",
                    // This forces a 'redacted_thinking' response - see https://docs.anthropic.com/en/docs/build-with-claude/extended-thinking
                    "content": "ANTHROPIC_MAGIC_STRING_TRIGGER_REDACTED_THINKING_46C9A13E193C177646C7398A98432ECCCE4C1253D5E2D82641AC0E52CC2876CB"
                },
                {
                    "role": "user",
                    "content": "What is the capital of Japan?"
                }
            ]},
        "stream": false,
    });

    let response = client
        .post(get_gateway_endpoint("/inference"))
        .json(&payload)
        .send()
        .await
        .unwrap();
    // Check Response is OK, then fields in order
    assert_eq!(response.status(), StatusCode::OK);
    let response_json = response.json::<Value>().await.unwrap();
    let content_blocks = response_json.get("content").unwrap().as_array().unwrap();
    let tensorzero_content_blocks = content_blocks.clone();
    assert!(
        content_blocks.len() == 2,
        "Unexpected content blocks: {:?}",
        content_blocks
    );
    let first_block = &content_blocks[0];
    let first_block_type = first_block.get("type").unwrap().as_str().unwrap();
    assert_eq!(first_block_type, "unknown");
    assert_eq!(
        first_block["model_provider_name"],
        "tensorzero::model_name::claude-3-7-sonnet-20250219-thinking::provider_name::anthropic-extra-body"
    );
    assert_eq!(first_block["data"]["type"], "redacted_thinking");

    let second_block = &content_blocks[1];
    assert_eq!(second_block["type"], "text");
    let content = second_block.get("text").unwrap().as_str().unwrap();
    // Assert that Tokyo is in the content
    assert!(content.contains("Tokyo"), "Content should mention Tokyo");
    // Check that inference_id is here
    let inference_id = response_json.get("inference_id").unwrap().as_str().unwrap();
    let inference_id = Uuid::parse_str(inference_id).unwrap();

    // Sleep for 1 second to allow time for data to be inserted into ClickHouse (trailing writes from API)
    tokio::time::sleep(std::time::Duration::from_secs(1)).await;

    // Check ClickHouse
    let clickhouse = get_clickhouse().await;

    // First, check Inference table
    let result = select_chat_inference_clickhouse(&clickhouse, inference_id)
        .await
        .unwrap();
    let id = result.get("id").unwrap().as_str().unwrap();
    let id_uuid = Uuid::parse_str(id).unwrap();
    assert_eq!(id_uuid, inference_id);
    let function_name = result.get("function_name").unwrap().as_str().unwrap();
    assert_eq!(function_name, "tensorzero::default");
    let input: Value =
        serde_json::from_str(result.get("input").unwrap().as_str().unwrap()).unwrap();
    let correct_input = json!({
        "messages": [
            {
                "role": "user",
                "content": [{"type": "text", "value": "ANTHROPIC_MAGIC_STRING_TRIGGER_REDACTED_THINKING_46C9A13E193C177646C7398A98432ECCCE4C1253D5E2D82641AC0E52CC2876CB"}]
            },
            {
                "role": "user",
                "content": [{"type": "text", "value": "What is the capital of Japan?"}]
            }
        ]
    });
    assert_eq!(input, correct_input);
    let content_blocks = result.get("output").unwrap().as_str().unwrap();
    // Check that content_blocks is a list of blocks length 1
    let content_blocks: Vec<Value> = serde_json::from_str(content_blocks).unwrap();
    assert_eq!(content_blocks.len(), 2);
    let first_block = &content_blocks[0];
    // Check the type and content in the block
    assert_eq!(first_block["type"], "unknown");
    assert_eq!(first_block["data"]["type"], "redacted_thinking");
    assert_eq!(first_block["model_provider_name"], "tensorzero::model_name::claude-3-7-sonnet-20250219-thinking::provider_name::anthropic-extra-body");
    let second_block = &content_blocks[1];
    assert_eq!(second_block["type"], "text");
    let clickhouse_content = second_block.get("text").unwrap().as_str().unwrap();
    assert_eq!(clickhouse_content, content);
    // Check that episode_id is here and correct
    let retrieved_episode_id = result.get("episode_id").unwrap().as_str().unwrap();
    let retrieved_episode_id = Uuid::parse_str(retrieved_episode_id).unwrap();
    assert_eq!(retrieved_episode_id, episode_id);
    // Check the variant name
    let variant_name = result.get("variant_name").unwrap().as_str().unwrap();
    assert_eq!(variant_name, "claude-3-7-sonnet-20250219-thinking");
    // Check the processing time
    let processing_time_ms = result.get("processing_time_ms").unwrap().as_u64().unwrap();
    assert!(processing_time_ms > 0);

    // Check the ModelInference Table
    let result = select_model_inference_clickhouse(&clickhouse, inference_id)
        .await
        .unwrap();
    let inference_id_result = result.get("inference_id").unwrap().as_str().unwrap();
    let inference_id_result = Uuid::parse_str(inference_id_result).unwrap();
    assert_eq!(inference_id_result, inference_id);
    let model_name = result.get("model_name").unwrap().as_str().unwrap();
    assert_eq!(model_name, "claude-3-7-sonnet-20250219-thinking");
    let model_provider_name = result.get("model_provider_name").unwrap().as_str().unwrap();
    assert_eq!(model_provider_name, "anthropic-extra-body");
    let raw_request = result.get("raw_request").unwrap().as_str().unwrap();
    assert!(raw_request.to_lowercase().contains("japan"));
    // Check that raw_request is valid JSON
    let _: Value = serde_json::from_str(raw_request).expect("raw_request should be valid JSON");
    let input_tokens = result.get("input_tokens").unwrap().as_u64().unwrap();
    assert!(input_tokens > 5);
    let output_tokens = result.get("output_tokens").unwrap().as_u64().unwrap();
    assert!(output_tokens > 5);
    let response_time_ms = result.get("response_time_ms").unwrap().as_u64().unwrap();
    assert!(response_time_ms > 0);
    assert!(result.get("ttft_ms").unwrap().is_null());
    let raw_response = result.get("raw_response").unwrap().as_str().unwrap();
    let _raw_response_json: Value = serde_json::from_str(raw_response).unwrap();

    // Feed content blocks back in

    let mut new_messages = json!([
        {
            "role": "user",
            // This forces a 'redacted_thinking' response - see https://docs.anthropic.com/en/docs/build-with-claude/extended-thinking
            "content": "ANTHROPIC_MAGIC_STRING_TRIGGER_REDACTED_THINKING_46C9A13E193C177646C7398A98432ECCCE4C1253D5E2D82641AC0E52CC2876CB"
        },
        {
            "role": "user",
            "content": "What is the capital of Japan?"
        }
    ]);
    let array = new_messages.as_array_mut().unwrap();
    array.push(serde_json::json!({
        "role": "assistant",
        "content": tensorzero_content_blocks,
    }));

    let payload = json!({
        "model_name": "claude-3-7-sonnet-20250219-thinking",
        "episode_id": episode_id,
        "input": {
            "messages": new_messages
        },
        "stream": false,
    });

    let response = client
        .post(get_gateway_endpoint("/inference"))
        .json(&payload)
        .send()
        .await
        .unwrap();
    // Check Response is OK, then fields in order
    assert_eq!(response.status(), StatusCode::OK);
    let response_json = response.json::<Value>().await.unwrap();

    let content_blocks = response_json.get("content").unwrap().as_array().unwrap();
    assert!(
        content_blocks.len() == 2,
        "Unexpected content blocks: {:?}",
        content_blocks
    );
    assert_eq!(content_blocks[1]["type"], "text");
}

/// This test checks that streaming inference works as expected.
#[tokio::test]
async fn test_streaming_thinking() {
    let episode_id = Uuid::now_v7();

    let payload = json!({
        "model_name": "claude-3-7-sonnet-20250219-thinking",
        "episode_id": episode_id,
        "input": {
            "system": "Always thinking before responding",
            "messages": [
                {
                    "role": "user",
                    "content": "What is the capital of Japan?"
                }
            ]
        },
        "tool_choice": "auto",
        "additional_tools": [
            {
                "description": "Gets the capital of the provided country",
                "name": "get_capital",
                "parameters": {
                    "$schema": "http://json-schema.org/draft-07/schema#",
                    "type": "object",
                    "properties": {
                        "country": {
                            "type": "string",
                            "description": "The country to lookup",
                        }
                    },
                    "required": ["country"],
                    "additionalProperties": false
                },
                "strict": true,
            }
        ],
        "stream": true,
    });

    let client = Client::new();

    let mut event_source = client
        .post(get_gateway_endpoint("/inference"))
        .json(&payload)
        .eventsource()
        .unwrap();
    let mut chunks = vec![];
    while let Some(event) = event_source.next().await {
        let event = event.unwrap();
        match event {
            Event::Open => continue,
            Event::Message(message) => {
                if message.data == "[DONE]" {
                    break;
                }
                chunks.push(message.data);
            }
        }
    }
    let mut inference_id = None;
    let mut content_blocks: HashMap<String, String> = HashMap::new();
    let mut content_block_signatures: HashMap<String, String> = HashMap::new();
    for chunk in chunks {
        let chunk_json: Value = serde_json::from_str(&chunk).unwrap();
        println!("Chunk: {chunk_json}");
        inference_id = Some(
            chunk_json
                .get("inference_id")
                .unwrap()
                .as_str()
                .unwrap()
                .to_string(),
        );
        for block in chunk_json.get("content").unwrap().as_array().unwrap() {
            let block_id = block.get("id").unwrap().as_str().unwrap();
            let block_type = block.get("type").unwrap().as_str().unwrap();
            let target = content_blocks.entry(block_id.to_string()).or_default();
            if block_type == "text" {
                *target += block.get("text").unwrap().as_str().unwrap();
            } else if block_type == "thought" {
                if let Some(text) = block.get("text").and_then(|t| t.as_str()) {
                    *target += text;
                } else if let Some(signature) = block.get("signature").and_then(|s| s.as_str()) {
                    *content_block_signatures
                        .entry(block_id.to_string())
                        .or_default() += signature;
                }
            } else if block_type == "tool_call" {
                *target += block.get("raw_arguments").unwrap().as_str().unwrap();
            } else {
                panic!("Unexpected block type: {block_type}");
            }
        }
    }
    assert_eq!(content_blocks.len(), 3);
    assert_eq!(content_block_signatures.len(), 1);
    let inference_id = Uuid::parse_str(&inference_id.unwrap()).unwrap();
    // Sleep for 1 second to allow time for data to be inserted into ClickHouse (trailing writes from API)
    tokio::time::sleep(std::time::Duration::from_secs(1)).await;

    // Check ClickHouse
    let clickhouse = get_clickhouse().await;
    let result = select_chat_inference_clickhouse(&clickhouse, inference_id)
        .await
        .unwrap();
    let id = result.get("id").unwrap().as_str().unwrap();
    let id_uuid = Uuid::parse_str(id).unwrap();
    assert_eq!(id_uuid, inference_id);
    let input: Value =
        serde_json::from_str(result.get("input").unwrap().as_str().unwrap()).unwrap();
    let correct_input: Value = json!(
        {
            "system": "Always thinking before responding",
            "messages": [
            {
                "role": "user",
                "content": [{"type": "text", "value": "What is the capital of Japan?"}]
            }
        ]}
    );
    assert_eq!(input, correct_input);
    // Check content blocks
    let clickhouse_content_blocks = result.get("output").unwrap().as_str().unwrap();
    let clickhouse_content_blocks: Vec<Value> =
        serde_json::from_str(clickhouse_content_blocks).unwrap();
    println!("Got content blocks: {clickhouse_content_blocks:?}");
    // We should reconstruct thee blocks - a thought block, tool call, and text block
    // Note that we currently always store the tool call first in ClickHouse
    assert_eq!(clickhouse_content_blocks.len(), 3);
    assert_eq!(clickhouse_content_blocks[0]["type"], "tool_call");
    assert_eq!(clickhouse_content_blocks[1]["type"], "thought");
    assert_eq!(clickhouse_content_blocks[2]["type"], "text");

    assert_eq!(clickhouse_content_blocks[1]["text"], content_blocks["0"]);
    assert_eq!(
        clickhouse_content_blocks[1]["signature"],
        content_block_signatures["0"]
    );
    assert_eq!(clickhouse_content_blocks[2]["text"], content_blocks["1"]);

    let tool_call_id = clickhouse_content_blocks[0]["id"].as_str().unwrap();

    assert_eq!(
        clickhouse_content_blocks[0]["raw_arguments"],
        content_blocks[tool_call_id]
    );

    // We already check ModelInference in lots of tests, so we don't check it here

    // Call Anthropic again with our reconstructed blocks, and make sure that it accepts the signed thought block

    let good_input = json!({
        "model_name": "claude-3-7-sonnet-20250219-thinking",
        "episode_id": episode_id,
        "input": {
            "system": "Always thinking before responding",
            "messages": [
                {
                    "role": "user",
                    "content": "What is the capital of Japan?"
                },
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "thought",
                            "text": content_blocks["0"],
                            "signature": content_block_signatures["0"],
                        },
                        {
                            "type": "text",
                            "text": content_blocks["1"]
                        },
                        {
                            "type": "tool_call",
                            "name": "get_capital",
                            "id": tool_call_id,
                            "arguments": serde_json::Value::String(content_blocks[tool_call_id].to_string()),
                        },
                    ]
                },
                {
                    "role": "user",
                    "content":[
                        {
                            "type": "tool_result",
                            "name": "get_capital",
                            "id": tool_call_id,
                            "result": "FakeCapital",
                        },
                    ]
                }
            ]
        },
        "tool_choice": "auto",
        "additional_tools": [
            {
                "description": "Gets the capital of the provided country",
                "name": "get_capital",
                "parameters": {
                    "$schema": "http://json-schema.org/draft-07/schema#",
                    "type": "object",
                    "properties": {
                        "country": {
                            "type": "string",
                            "description": "The country to lookup",
                        }
                    },
                    "required": ["country"],
                    "additionalProperties": false
                },
                "strict": true,
            }
        ],
        "stream": false,
    });

    let good_response = client
        .post(get_gateway_endpoint("/inference"))
        .json(&good_input)
        .send()
        .await
        .unwrap();
    assert_eq!(good_response.status(), StatusCode::OK);
    let good_json = good_response.json::<Value>().await.unwrap();
    println!("Good response: {good_json}");

    // Break the signature and check that it fails
    let bad_signature = "A".repeat(content_block_signatures["0"].len());
    let mut bad_input = good_input.clone();
    bad_input["input"]["messages"][1]["content"][0]["signature"] =
        Value::String(bad_signature.clone());

    let bad_response = client
        .post(get_gateway_endpoint("/inference"))
        .json(&bad_input)
        .send()
        .await
        .unwrap();

    let status = bad_response.status();
    let resp: Value = bad_response.json().await.unwrap();
    assert_eq!(
        status,
        StatusCode::BAD_GATEWAY,
        "Request should have failed: {resp}"
    );
    assert!(
        resp["error"]
            .as_str()
            .unwrap()
            .contains("Invalid `signature`"),
        "Unexpected error: {resp}"
    );
}
