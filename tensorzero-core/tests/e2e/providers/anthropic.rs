#![expect(clippy::print_stdout)]
use std::collections::HashMap;

use futures::StreamExt;
use indexmap::IndexMap;
use reqwest::{Client, StatusCode};
use reqwest_eventsource::{Event, RequestBuilderExt};
use serde_json::{json, Value};
use tensorzero::{
    test_helpers::make_embedded_gateway_with_config, ClientInferenceParams, ClientInput,
    ClientInputMessage, ClientInputMessageContent, File, InferenceOutput, InferenceResponse, Role,
    UrlFile,
};
use url::Url;
use uuid::Uuid;

use crate::{
    common::get_gateway_endpoint,
    providers::common::{E2ETestProvider, E2ETestProviders, DEEPSEEK_PAPER_PDF, FERRIS_PNG},
};
use tensorzero_core::{
    db::clickhouse::test_helpers::{
        get_clickhouse, select_chat_inference_clickhouse, select_model_inference_clickhouse,
    },
    inference::types::{ContentBlockChatOutput, TextKind},
};

use super::common::ModelTestProvider;

crate::generate_provider_tests!(get_providers);
crate::generate_batch_inference_tests!(get_providers);

async fn get_providers() -> E2ETestProviders {
    let credentials = match std::env::var("ANTHROPIC_API_KEY") {
        Ok(key) => HashMap::from([("anthropic_api_key".to_string(), key)]),
        Err(_) => HashMap::new(),
    };

    let standard_providers = vec![E2ETestProvider {
        supports_batch_inference: false,
        variant_name: "anthropic".to_string(),
        model_name: "claude-3-haiku-20240307-anthropic".into(),
        model_provider_name: "anthropic".into(),
        credentials: HashMap::new(),
    }];

    let image_providers = vec![E2ETestProvider {
        supports_batch_inference: false,
        variant_name: "anthropic".to_string(),
        model_name: "anthropic::claude-3-haiku-20240307".into(),
        model_provider_name: "anthropic".into(),
        credentials: HashMap::new(),
    }];

    let pdf_providers = vec![E2ETestProvider {
        supports_batch_inference: false,
        variant_name: "anthropic".to_string(),
        model_name: "anthropic::claude-sonnet-4-5-20250929".into(),
        model_provider_name: "anthropic".into(),
        credentials: HashMap::new(),
    }];

    let extra_body_providers = vec![E2ETestProvider {
        supports_batch_inference: false,
        variant_name: "anthropic-extra-body".to_string(),
        model_name: "claude-3-haiku-20240307-anthropic".into(),
        model_provider_name: "anthropic".into(),
        credentials: HashMap::new(),
    }];

    let bad_auth_extra_headers = vec![E2ETestProvider {
        supports_batch_inference: false,
        variant_name: "anthropic-extra-headers".to_string(),
        model_name: "claude-3-haiku-20240307-anthropic".into(),
        model_provider_name: "anthropic".into(),
        credentials: HashMap::new(),
    }];

    let inference_params_dynamic_providers = vec![E2ETestProvider {
        supports_batch_inference: false,
        variant_name: "anthropic-dynamic".to_string(),
        model_name: "claude-3-haiku-20240307-anthropic-dynamic".into(),
        model_provider_name: "anthropic".into(),
        credentials,
    }];

    let json_providers = vec![
        E2ETestProvider {
            supports_batch_inference: false,
            variant_name: "anthropic".to_string(),
            model_name: "claude-3-haiku-20240307-anthropic".into(),
            model_provider_name: "anthropic".into(),
            credentials: HashMap::new(),
        },
        E2ETestProvider {
            supports_batch_inference: false,
            variant_name: "anthropic-implicit".to_string(),
            model_name: "claude-3-haiku-20240307-anthropic".into(),
            model_provider_name: "anthropic".into(),
            credentials: HashMap::new(),
        },
        E2ETestProvider {
            supports_batch_inference: false,
            variant_name: "anthropic-strict".to_string(),
            model_name: "claude-3-haiku-20240307-anthropic".into(),
            model_provider_name: "anthropic".into(),
            credentials: HashMap::new(),
        },
    ];

    let json_mode_off_providers = vec![E2ETestProvider {
        supports_batch_inference: false,
        variant_name: "anthropic_json_mode_off".to_string(),
        model_name: "claude-3-haiku-20240307-anthropic".into(),
        model_provider_name: "anthropic".into(),
        credentials: HashMap::new(),
    }];

    let shorthand_providers = vec![E2ETestProvider {
        supports_batch_inference: false,
        variant_name: "anthropic-shorthand".to_string(),
        model_name: "anthropic::claude-3-haiku-20240307".into(),
        model_provider_name: "anthropic".into(),
        credentials: HashMap::new(),
    }];

    let provider_type_default_credentials_providers = vec![E2ETestProvider {
        supports_batch_inference: false,
        variant_name: "anthropic".to_string(),
        model_name: "claude-3-haiku-20240307".into(),
        model_provider_name: "anthropic".into(),
        credentials: HashMap::new(),
    }];

    let provider_type_default_credentials_shorthand_providers = vec![E2ETestProvider {
        supports_batch_inference: false,
        variant_name: "anthropic-shorthand".to_string(),
        model_name: "anthropic::claude-3-haiku-20240307".into(),
        model_provider_name: "anthropic".into(),
        credentials: HashMap::new(),
    }];

    let credential_fallbacks = vec![ModelTestProvider {
        provider_type: "anthropic".into(),
        model_info: HashMap::from([(
            "model_name".to_string(),
            "claude-3-haiku-20240307".to_string(),
        )]),
        use_modal_headers: false,
    }];

    E2ETestProviders {
        simple_inference: standard_providers.clone(),
        bad_auth_extra_headers,
        extra_body_inference: extra_body_providers,
        reasoning_inference: vec![],
        embeddings: vec![],
        inference_params_inference: standard_providers.clone(),
        inference_params_dynamic_credentials: inference_params_dynamic_providers,
        provider_type_default_credentials: provider_type_default_credentials_providers,
        provider_type_default_credentials_shorthand:
            provider_type_default_credentials_shorthand_providers,
        tool_use_inference: standard_providers.clone(),
        tool_multi_turn_inference: standard_providers.clone(),
        dynamic_tool_use_inference: standard_providers.clone(),
        parallel_tool_use_inference: standard_providers.clone(),
        json_mode_inference: json_providers.clone(),
        json_mode_off_inference: json_mode_off_providers.clone(),
        image_inference: image_providers,
        pdf_inference: pdf_providers,
        input_audio: vec![],
        shorthand_inference: shorthand_providers.clone(),
        credential_fallbacks,
    }
}

#[tokio::test]
async fn test_thinking_rejected_128k() {
    let client = Client::new();

    // Test that we get an error when we don't pass the 'anthropic-beta' header
    // This ensures that our remaining extra-headers tests actually test something
    // that has an effect

    // We inject randomness to ensure that we don't get a cache hit in provider-proxy-cache,
    // since we want to test the current Anthropic behavior.
    let random = Uuid::now_v7();
    let payload = json!({
        "model_name": "anthropic::claude-3-7-sonnet-20250219",
        "input":{
            "messages": [
                {
                    "role": "user",
                    "content": format!("Output a haiku that ends in the word 'my_custom_stop': {random}"),
                }
            ]},
        "params": {
            "chat_completion": {
                "max_tokens": 128000,
                "thinking_budget_tokens": 1024,
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
        StatusCode::BAD_GATEWAY,
        "Unexpected status code with response: {resp_json}"
    );
    assert!(
        resp_json["error"]
            .as_str()
            .unwrap()
            .contains("the maximum allowed number of output tokens"),
        "Unexpected error: {resp_json}"
    );
}

async fn test_thinking_helper(client: &Client, payload: &serde_json::Value) -> serde_json::Value {
    let response = client
        .post(get_gateway_endpoint("/inference"))
        .json(payload)
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
    resp_json
}

#[tokio::test]
async fn test_thinking_inference_extra_header_128k() {
    let client = Client::new();

    // This model uses a custom stop sequence, as we want to make sure that
    // we don't actually generate 128k tokens of output. This test just verifies
    // that we can pass through the necessary 'anthropic-beta' header to support
    // a large 'max_tokens'
    let payload = json!({
        "model_name": "anthropic::claude-3-7-sonnet-20250219",
        "input":{
            "messages": [
                {
                    "role": "user",
                    "content": "Output a haiku that ends in the word 'my_custom_stop'"
                }
            ]},
        "extra_headers": [
            {
                "model_name": "anthropic::claude-3-7-sonnet-20250219",
                "provider_name": "anthropic",
                "name": "anthropic-beta",
                "value": "output-128k-2025-02-19"
            }
        ],
        "extra_body": [
            {
                "model_name": "anthropic::claude-3-7-sonnet-20250219",
                "provider_name": "anthropic",
                "pointer": "/stop_sequences",
                "value": [
                    "my_custom_stop",
                ]
            }
        ],
        // We use a budget tokens of 1024 to make sure that it doesn't think for too long,
        // since 'stop_sequences' does not seem to apply to thinking. We set 'max_tokens'
        // to 128k in 'test_thinking_128k'
        "params": {
            "chat_completion": {
                "max_tokens": 128000,
                "thinking_budget_tokens": 1024,
            }
        },
    });

    let response = test_thinking_helper(&client, &payload).await;
    let inference_id = response.get("inference_id").unwrap().as_str().unwrap();

    let inference_id = Uuid::parse_str(inference_id).unwrap();

    // Sleep for 200ms to allow time for data to be inserted into ClickHouse (trailing writes from API)
    tokio::time::sleep(std::time::Duration::from_millis(200)).await;

    // Check ClickHouse
    let clickhouse = get_clickhouse().await;

    let result = select_chat_inference_clickhouse(&clickhouse, inference_id)
        .await
        .unwrap();

    // Check that thinking_budget_tokens was stored correctly
    let inference_params: Value =
        serde_json::from_str(result["inference_params"].as_str().unwrap()).unwrap();
    println!("inference_params: {inference_params:?}");
    assert_eq!(
        inference_params["chat_completion"]["thinking_budget_tokens"],
        1024
    );
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
                "thinking_budget_tokens": 1024,
                "max_tokens": 128000,
            }
        },
        "stream": false,
    });

    test_thinking_helper(&client, &payload).await;

    // We don't check the database, as we already do that in lots of places.
}

#[tokio::test]
pub async fn test_thinking_signature() {
    test_thinking_signature_helper(
        "anthropic-thinking",
        "anthropic::claude-3-7-sonnet-20250219",
        "anthropic",
    )
    .await;
}

pub async fn test_thinking_signature_helper(
    variant_name: &str,
    model_name: &str,
    model_provider_name: &str,
) {
    let client = Client::new();
    let episode_id = Uuid::now_v7();

    let payload = json!({
        "function_name": "weather_helper",
        "variant_name": variant_name,
        "episode_id": episode_id,
        "input":{
            "system": {"assistant_name": "AskJeeves"},
            "messages": [
                {
                    "role": "user",
                    "content": "Hi I'm visiting Brooklyn from Brazil. What's the weather (use degrees Celsius)?"
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
        content_blocks.len() >= 2,
        "Unexpected content blocks: {content_blocks:?}"
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

    // There should be a tool call block in the output. We don't check for a 'text' output block, since not
    // all models emit one here.
    let inference_id = response_json.get("inference_id").unwrap().as_str().unwrap();
    let inference_id = Uuid::parse_str(inference_id).unwrap();
    let tool_call_block = content_blocks
        .iter()
        .find(|block| block["type"] == "tool_call")
        .unwrap();
    assert_eq!(tool_call_block["type"], "tool_call");
    assert_eq!(tool_call_block["name"], "get_temperature");
    let tool_id = tool_call_block.get("id").unwrap().as_str().unwrap();

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
                "content": [{"type": "text", "text": "Hi I'm visiting Brooklyn from Brazil. What's the weather (use degrees Celsius)?"}]
            }
        ]
    });
    assert_eq!(input, correct_input);
    // Check that episode_id is here and correct
    let retrieved_episode_id = result.get("episode_id").unwrap().as_str().unwrap();
    let retrieved_episode_id = Uuid::parse_str(retrieved_episode_id).unwrap();
    assert_eq!(retrieved_episode_id, episode_id);
    // Check the variant name
    let retrieved_variant_name = result.get("variant_name").unwrap().as_str().unwrap();
    assert_eq!(retrieved_variant_name, variant_name);
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
    let retrieved_model_name = result.get("model_name").unwrap().as_str().unwrap();
    assert_eq!(retrieved_model_name, model_name);
    let retrieved_model_provider_name =
        result.get("model_provider_name").unwrap().as_str().unwrap();
    assert_eq!(retrieved_model_provider_name, model_provider_name);
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
        "model_name": model_name,
        "episode_id": episode_id,
        "input": {
            "messages": new_messages
        },
        "params": {
            "chat_completion": {
                "thinking_budget_tokens": 1024,
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

    assert_eq!(response.status(), StatusCode::OK);
    let response_json = response.json::<Value>().await.unwrap();

    let content_blocks = response_json.get("content").unwrap().as_array().unwrap();
    assert!(
        content_blocks.len() == 1,
        "Unexpected content blocks: {content_blocks:?}"
    );
    assert_eq!(content_blocks[0]["type"], "text");
    assert!(
        content_blocks[0]["text"].as_str().unwrap().contains("100°"),
        "Content should mention '100°': {}",
        content_blocks[0]
    );
}

#[tokio::test]
pub async fn test_redacted_thinking() {
    test_redacted_thinking_helper(
        "anthropic::claude-3-7-sonnet-20250219",
        "anthropic",
        "anthropic",
    )
    .await;
}

pub async fn test_redacted_thinking_helper(
    model_name: &str,
    model_provider_name: &str,
    provider_type: &str,
) {
    let client = Client::new();
    let episode_id = Uuid::now_v7();

    let payload = json!({
        "model_name": model_name,
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
        "params": {
            "chat_completion": {
                "thinking_budget_tokens": 1024,
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
    // Check Response is OK, then fields in order
    assert_eq!(response.status(), StatusCode::OK);
    let response_json = response.json::<Value>().await.unwrap();
    let content_blocks = response_json.get("content").unwrap().as_array().unwrap();
    let tensorzero_content_blocks = content_blocks.clone();
    assert!(
        content_blocks.len() == 2,
        "Unexpected content blocks: {content_blocks:?}"
    );
    let first_block = &content_blocks[0];
    let first_block_type = first_block.get("type").unwrap().as_str().unwrap();
    assert_eq!(first_block_type, "thought");
    assert_eq!(first_block["_internal_provider_type"], provider_type);
    assert!(first_block["signature"].as_str().is_some());

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
                "content": [{"type": "text", "text": "ANTHROPIC_MAGIC_STRING_TRIGGER_REDACTED_THINKING_46C9A13E193C177646C7398A98432ECCCE4C1253D5E2D82641AC0E52CC2876CB"}]
            },
            {
                "role": "user",
                "content": [{"type": "text", "text": "What is the capital of Japan?"}]
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
    assert_eq!(first_block["type"], "thought");
    assert_eq!(first_block["_internal_provider_type"], provider_type);
    assert!(first_block["signature"].as_str().is_some());
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
    assert_eq!(variant_name, model_name);
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
    let retrieved_model_name = result.get("model_name").unwrap().as_str().unwrap();
    assert_eq!(retrieved_model_name, model_name);
    let retrieved_model_provider_name =
        result.get("model_provider_name").unwrap().as_str().unwrap();
    assert_eq!(retrieved_model_provider_name, model_provider_name);
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
        "model_name": model_name,
        "episode_id": episode_id,
        "input": {
            "messages": new_messages
        },
        "params": {
            "chat_completion": {
                "thinking_budget_tokens": 1024,
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
    // Check Response is OK, then fields in order
    assert_eq!(response.status(), StatusCode::OK);
    let response_json = response.json::<Value>().await.unwrap();

    let content_blocks = response_json.get("content").unwrap().as_array().unwrap();
    assert!(
        content_blocks.len() == 2,
        "Unexpected content blocks: {content_blocks:?}"
    );
    assert_eq!(content_blocks[1]["type"], "text");
}

#[tokio::test]
async fn test_beta_structured_outputs_json_streaming() {
    test_beta_structured_outputs_json_helper(true).await;
}

#[tokio::test]
async fn test_beta_structured_outputs_json_non_streaming() {
    test_beta_structured_outputs_json_helper(false).await;
}

async fn test_beta_structured_outputs_json_helper(stream: bool) {
    let client = Client::new();
    let payload = json!({
        "function_name": "anthropic_beta_structured_outputs_json",
        "input": {
            "messages": [
                {
                    "role": "user",
                    "content": "What is the capital of Japan?"
                }
            ]
        },
        "stream": stream,
    });

    let inference_id = if stream {
        let mut event_source = client
            .post(get_gateway_endpoint("/inference"))
            .json(&payload)
            .eventsource()
            .unwrap();
        let mut first_inference_id = None;
        while let Some(event) = event_source.next().await {
            let event = event.unwrap();
            match event {
                Event::Open => continue,
                Event::Message(message) => {
                    if message.data == "[DONE]" {
                        break;
                    }
                    let chunk_json: Value = serde_json::from_str(&message.data).unwrap();
                    if let Some(inference_id) = chunk_json
                        .get("inference_id")
                        .and_then(|id| id.as_str().map(|id| Uuid::parse_str(id).unwrap()))
                    {
                        first_inference_id = Some(inference_id);
                    }
                }
            }
        }
        first_inference_id.unwrap()
    } else {
        let response = client
            .post(get_gateway_endpoint("/inference"))
            .json(&payload)
            .send()
            .await
            .unwrap();
        let status = response.status();
        let response_text = response.text().await.unwrap();
        println!("Response text: {response_text}");
        let response_json: Value = serde_json::from_str(&response_text).unwrap();
        assert_eq!(status, StatusCode::OK);
        let inference_id = response_json
            .get("inference_id")
            .unwrap()
            .as_str()
            .unwrap()
            .to_string();
        Uuid::parse_str(&inference_id).unwrap()
    };

    // Wait one second to allow time for data to be inserted into ClickHouse (trailing writes from API)
    tokio::time::sleep(std::time::Duration::from_secs(1)).await;

    // Check ClickHouse
    let clickhouse = get_clickhouse().await;
    let result_json = select_model_inference_clickhouse(&clickhouse, inference_id)
        .await
        .unwrap();
    println!("Result: {result_json}");

    // Check that the result is valid JSON
    println!("Result JSON: {result_json}");
    let output = result_json.get("output");
    println!("Output: {output:?}");

    let raw_request = result_json.get("raw_request").unwrap().as_str().unwrap();
    if stream {
        assert_eq!(raw_request, "{\"model\":\"claude-sonnet-4-5-20250929\",\"messages\":[{\"role\":\"user\",\"content\":[{\"type\":\"text\",\"text\":\"What is the capital of Japan?\"}]}],\"max_tokens\":100,\"stream\":true,\"output_format\":{\"type\":\"json_schema\",\"schema\":{\"type\":\"object\",\"properties\":{\"answer\":{\"type\":\"string\"}},\"required\":[\"answer\"],\"additionalProperties\":false}}}");
    } else {
        assert_eq!(raw_request, "{\"model\":\"claude-sonnet-4-5-20250929\",\"messages\":[{\"role\":\"user\",\"content\":[{\"type\":\"text\",\"text\":\"What is the capital of Japan?\"}]}],\"max_tokens\":100,\"stream\":false,\"output_format\":{\"type\":\"json_schema\",\"schema\":{\"type\":\"object\",\"properties\":{\"answer\":{\"type\":\"string\"}},\"required\":[\"answer\"],\"additionalProperties\":false}}}");
    }
}

#[tokio::test]
async fn test_beta_structured_outputs_strict_tool_streaming() {
    test_beta_structured_outputs_strict_tool_helper(true).await;
}

#[tokio::test]
async fn test_beta_structured_outputs_strict_tool_non_streaming() {
    test_beta_structured_outputs_strict_tool_helper(false).await;
}

async fn test_beta_structured_outputs_strict_tool_helper(stream: bool) {
    let client = Client::new();
    let payload = json!({
        "function_name": "anthropic_beta_structured_outputs_chat",
        "input": {
            "messages": [
                {
                    "role": "user",
                    "content": "What is the capital of Japan?"
                }
            ]
        },
        "stream": stream,
    });

    let inference_id = if stream {
        let mut event_source = client
            .post(get_gateway_endpoint("/inference"))
            .json(&payload)
            .eventsource()
            .unwrap();
        let mut first_inference_id = None;
        while let Some(event) = event_source.next().await {
            let event = event.unwrap();
            match event {
                Event::Open => continue,
                Event::Message(message) => {
                    if message.data == "[DONE]" {
                        break;
                    }
                    let chunk_json: Value = serde_json::from_str(&message.data).unwrap();
                    if let Some(inference_id) = chunk_json
                        .get("inference_id")
                        .and_then(|id| id.as_str().map(|id| Uuid::parse_str(id).unwrap()))
                    {
                        first_inference_id = Some(inference_id);
                    }
                }
            }
        }
        first_inference_id.unwrap()
    } else {
        let response = client
            .post(get_gateway_endpoint("/inference"))
            .json(&payload)
            .send()
            .await
            .unwrap();
        let status = response.status();
        let response_text = response.text().await.unwrap();
        println!("Response text: {response_text}");
        let response_json: Value = serde_json::from_str(&response_text).unwrap();
        assert_eq!(status, StatusCode::OK);
        let inference_id = response_json
            .get("inference_id")
            .unwrap()
            .as_str()
            .unwrap()
            .to_string();
        Uuid::parse_str(&inference_id).unwrap()
    };

    // Wait one second to allow time for data to be inserted into ClickHouse (trailing writes from API)
    tokio::time::sleep(std::time::Duration::from_secs(1)).await;

    // Check ClickHouse
    let clickhouse = get_clickhouse().await;
    let result_json = select_model_inference_clickhouse(&clickhouse, inference_id)
        .await
        .unwrap();
    println!("Result: {result_json}");

    // Check that the result is valid JSON
    println!("Result JSON: {result_json}");
    let output = result_json.get("output");
    println!("Output: {output:?}");

    let raw_request = result_json.get("raw_request").unwrap().as_str().unwrap();
    if stream {
        assert_eq!(raw_request, "{\"model\":\"claude-sonnet-4-5-20250929\",\"messages\":[{\"role\":\"user\",\"content\":[{\"type\":\"text\",\"text\":\"What is the capital of Japan?\"}]}],\"max_tokens\":100,\"stream\":true,\"tool_choice\":{\"type\":\"auto\",\"disable_parallel_tool_use\":false},\"tools\":[{\"name\":\"answer_question\",\"description\":\"End the search process and answer a question. Returns the answer to the question.\",\"input_schema\":{\"$schema\":\"http://json-schema.org/draft-07/schema#\",\"type\":\"object\",\"description\":\"End the search process and answer a question. Returns the answer to the question.\",\"properties\":{\"answer\":{\"type\":\"string\",\"description\":\"The answer to the question.\"}},\"required\":[\"answer\"],\"additionalProperties\":false},\"strict\":true}]}");
    } else {
        assert_eq!(raw_request, "{\"model\":\"claude-sonnet-4-5-20250929\",\"messages\":[{\"role\":\"user\",\"content\":[{\"type\":\"text\",\"text\":\"What is the capital of Japan?\"}]}],\"max_tokens\":100,\"stream\":false,\"tool_choice\":{\"type\":\"auto\",\"disable_parallel_tool_use\":false},\"tools\":[{\"name\":\"answer_question\",\"description\":\"End the search process and answer a question. Returns the answer to the question.\",\"input_schema\":{\"$schema\":\"http://json-schema.org/draft-07/schema#\",\"type\":\"object\",\"description\":\"End the search process and answer a question. Returns the answer to the question.\",\"properties\":{\"answer\":{\"type\":\"string\",\"description\":\"The answer to the question.\"}},\"required\":[\"answer\"],\"additionalProperties\":false},\"strict\":true}]}");
    }
}

/// This test checks that streaming inference works as expected.
#[tokio::test]
async fn test_streaming_thinking() {
    test_streaming_thinking_helper("anthropic::claude-3-7-sonnet-20250219", "anthropic").await;
}

pub async fn test_streaming_thinking_helper(model_name: &str, provider_type: &str) {
    let episode_id = Uuid::now_v7();

    let payload = json!({
        "model_name": model_name,
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
        "params": {
            "chat_completion": {
                "thinking_budget_tokens": 1024,
            }
        },
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
    let mut content_blocks: IndexMap<(String, String), String> = IndexMap::new();
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
            let target = content_blocks
                .entry((block_type.to_string(), block_id.to_string()))
                .or_default();
            if block_type == "text" {
                *target += block.get("text").unwrap().as_str().unwrap();
            } else if block_type == "thought" {
                if let Some(text) = block.get("text").and_then(|t| t.as_str()) {
                    *target += text;
                }
                if let Some(signature) = block.get("signature").and_then(|s| s.as_str()) {
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
    assert!(
        content_blocks.len() == 2 || content_blocks.len() == 3,
        "Expected 2 or 3 content blocks, got {}",
        content_blocks.len()
    );
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
                "content": [{"type": "text", "text": "What is the capital of Japan?"}]
            }
        ]}
    );
    assert_eq!(input, correct_input);
    // Check content blocks
    let clickhouse_content_blocks = result.get("output").unwrap().as_str().unwrap();
    let clickhouse_content_blocks: Vec<Value> =
        serde_json::from_str(clickhouse_content_blocks).unwrap();
    println!("Got content blocks: {clickhouse_content_blocks:?}");
    assert!(
        clickhouse_content_blocks.len() == 2 || clickhouse_content_blocks.len() == 3,
        "Expected 2 or 3 content blocks in ClickHouse, got {}",
        clickhouse_content_blocks.len()
    );
    assert_eq!(clickhouse_content_blocks[0]["type"], "thought");

    let tool_call_index = clickhouse_content_blocks.len() - 1;
    let has_text_block = clickhouse_content_blocks.len() == 3;

    if has_text_block {
        assert_eq!(clickhouse_content_blocks[1]["type"], "text");
    }
    assert_eq!(
        clickhouse_content_blocks[tool_call_index]["type"],
        "tool_call"
    );

    assert_eq!(
        clickhouse_content_blocks[0],
        serde_json::json!({
            "type": "thought",
            "text": content_blocks[&("thought".to_string(), "0".to_string())],
            "signature": content_block_signatures["0"],
            "_internal_provider_type": provider_type,
        })
    );

    if has_text_block {
        assert_eq!(
            clickhouse_content_blocks[1]["text"],
            content_blocks[&("text".to_string(), "1".to_string())]
        );
    }

    let tool_call_id = clickhouse_content_blocks[tool_call_index]["id"]
        .as_str()
        .unwrap();

    assert_eq!(
        clickhouse_content_blocks[tool_call_index]["raw_arguments"],
        content_blocks[&("tool_call".to_string(), tool_call_id.to_string())]
    );

    // We already check ModelInference in lots of tests, so we don't check it here

    // Call Anthropic again with our reconstructed blocks, and make sure that it accepts the signed thought block

    let good_input = json!({
        "model_name": model_name,
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
                    "content": clickhouse_content_blocks
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
        "params": {
            "chat_completion": {
                "thinking_budget_tokens": 1024,
            }
        },
        "stream": false,
    });

    println!("Good input: {good_input}");

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

#[tokio::test]
async fn test_forward_image_url() {
    let temp_dir = tempfile::tempdir().unwrap();
    let config = format!(
        r#"
    [object_storage]
    type = "filesystem"
    path = "{}"

    [gateway]
    fetch_and_encode_input_files_before_inference = false
    "#,
        temp_dir.path().to_string_lossy()
    );

    let client = make_embedded_gateway_with_config(&config).await;

    let response = client.inference(ClientInferenceParams {
        model_name: Some("anthropic::claude-3-haiku-20240307".to_string()),
        input: ClientInput {
            messages: vec![ClientInputMessage {
                role: Role::User,
                content: vec![ClientInputMessageContent::Text(TextKind::Text { text: "Describe the contents of the image".to_string() }),
                ClientInputMessageContent::File(File::Url(UrlFile {
                    url: Url::parse("https://raw.githubusercontent.com/tensorzero/tensorzero/ff3e17bbd3e32f483b027cf81b54404788c90dc1/tensorzero-internal/tests/e2e/providers/ferris.png").unwrap(),
                    mime_type: Some(mime::IMAGE_PNG),
                    detail: None,
                    filename: None,
                })),
                ],
            }],
            ..Default::default()
        },
        ..Default::default()
    }).await.unwrap();

    let InferenceOutput::NonStreaming(response) = response else {
        panic!("Expected non-streaming inference response");
    };

    // Sleep for 1 second to allow writing to ClickHouse
    tokio::time::sleep(std::time::Duration::from_secs(1)).await;
    let clickhouse = get_clickhouse().await;

    let model_inference = select_model_inference_clickhouse(&clickhouse, response.inference_id())
        .await
        .unwrap();

    let raw_request = model_inference
        .get("raw_request")
        .unwrap()
        .as_str()
        .unwrap();
    assert_eq!(raw_request, "{\"model\":\"claude-3-haiku-20240307\",\"messages\":[{\"role\":\"user\",\"content\":[{\"type\":\"text\",\"text\":\"Describe the contents of the image\"},{\"type\":\"image\",\"source\":{\"type\":\"url\",\"url\":\"https://raw.githubusercontent.com/tensorzero/tensorzero/ff3e17bbd3e32f483b027cf81b54404788c90dc1/tensorzero-internal/tests/e2e/providers/ferris.png\"}}]}],\"max_tokens\":4096,\"stream\":false}");

    let file_path =
        "observability/files/08bfa764c6dc25e658bab2b8039ddb494546c3bc5523296804efc4cab604df5d.png";

    // Check that the file exists on the filesystem
    let result = std::fs::read(temp_dir.path().join(file_path)).unwrap();
    assert_eq!(result, FERRIS_PNG);

    println!("Got response: {response:#?}");

    let InferenceResponse::Chat(response) = response else {
        panic!("Expected chat inference response");
    };
    let text_block = &response.content[0];
    let ContentBlockChatOutput::Text(text) = text_block else {
        panic!("Expected text content block");
    };
    assert!(
        text.text.to_lowercase().contains("cartoon")
            || text.text.to_lowercase().contains("crab")
            || text.text.to_lowercase().contains("animal"),
        "Content should contain 'cartoon' or 'crab' or 'animal': {text:?}"
    );
}

#[tokio::test]
async fn test_forward_file_url() {
    let temp_dir = tempfile::tempdir().unwrap();
    let config = format!(
        r#"
    [object_storage]
    type = "filesystem"
    path = "{}"

    [gateway]
    fetch_and_encode_input_files_before_inference = false
    "#,
        temp_dir.path().to_string_lossy()
    );

    let client = make_embedded_gateway_with_config(&config).await;

    let response = client.inference(ClientInferenceParams {
        model_name: Some("anthropic::claude-sonnet-4-5-20250929".to_string()),
        input: ClientInput {
            messages: vec![ClientInputMessage {
                role: Role::User,
                content: vec![ClientInputMessageContent::Text(TextKind::Text { text: "Describe the contents of the PDF".to_string() }),
                ClientInputMessageContent::File(File::Url(UrlFile {
                    url: Url::parse("https://raw.githubusercontent.com/tensorzero/tensorzero/ac37477d56deaf6e0585a394eda68fd4f9390cab/tensorzero-core/tests/e2e/providers/deepseek_paper.pdf").unwrap(),
                    mime_type: Some(mime::APPLICATION_PDF),
                    detail: None,
                    filename: None,
                })),
                ],
            }],
            ..Default::default()
        },
        ..Default::default()
    }).await.unwrap();

    let InferenceOutput::NonStreaming(response) = response else {
        panic!("Expected non-streaming inference response");
    };

    // Sleep for 1 second to allow writing to ClickHouse
    tokio::time::sleep(std::time::Duration::from_secs(1)).await;
    let clickhouse = get_clickhouse().await;

    let model_inference = select_model_inference_clickhouse(&clickhouse, response.inference_id())
        .await
        .unwrap();

    let raw_request = model_inference
        .get("raw_request")
        .unwrap()
        .as_str()
        .unwrap();
    assert_eq!(raw_request, "{\"model\":\"claude-sonnet-4-5-20250929\",\"messages\":[{\"role\":\"user\",\"content\":[{\"type\":\"text\",\"text\":\"Describe the contents of the PDF\"},{\"type\":\"document\",\"source\":{\"type\":\"url\",\"url\":\"https://raw.githubusercontent.com/tensorzero/tensorzero/ac37477d56deaf6e0585a394eda68fd4f9390cab/tensorzero-core/tests/e2e/providers/deepseek_paper.pdf\"}}]}],\"max_tokens\":64000,\"stream\":false}");

    let file_path =
        "observability/files/3e127d9a726f6be0fd81d73ccea97d96ec99419f59650e01d49183cd3be999ef.pdf";

    // Check that the file exists on the filesystem
    let result = std::fs::read(temp_dir.path().join(file_path)).unwrap();
    assert_eq!(result, DEEPSEEK_PAPER_PDF);

    println!("Got response: {response:#?}");

    let InferenceResponse::Chat(response) = response else {
        panic!("Expected chat inference response");
    };
    let text_block = &response.content[0];
    let ContentBlockChatOutput::Text(text) = text_block else {
        panic!("Expected text content block");
    };
    assert!(
        text.text.to_lowercase().contains("deepseek"),
        "Content should contain 'deepseek': {text:?}"
    );
}
