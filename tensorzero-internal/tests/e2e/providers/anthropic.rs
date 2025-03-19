#![allow(clippy::print_stdout)]
use std::collections::HashMap;

use reqwest::{Client, StatusCode};
use serde_json::{json, Value};
use uuid::Uuid;

use crate::{
    common::get_gateway_endpoint,
    providers::common::{E2ETestProvider, E2ETestProviders},
};
use tensorzero_internal::clickhouse::test_helpers::{
    get_clickhouse, select_chat_inference_clickhouse, select_model_inference_clickhouse,
};

#[cfg(feature = "e2e_tests")]
crate::generate_provider_tests!(get_providers);
#[cfg(feature = "batch_tests")]
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

    #[cfg(feature = "e2e_tests")]
    let shorthand_providers = vec![E2ETestProvider {
        variant_name: "anthropic-shorthand".to_string(),
        model_name: "anthropic::claude-3-haiku-20240307".into(),
        model_provider_name: "anthropic".into(),
        credentials: HashMap::new(),
    }];

    E2ETestProviders {
        simple_inference: standard_providers.clone(),
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
        #[cfg(feature = "e2e_tests")]
        shorthand_inference: shorthand_providers.clone(),
        #[cfg(feature = "batch_tests")]
        supports_batch_inference: false,
    }
}

#[cfg_attr(not(feature = "e2e_tests"), ignore)]
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

#[cfg_attr(not(feature = "e2e_tests"), ignore)]
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
