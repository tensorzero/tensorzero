#![expect(clippy::print_stdout)]
use std::collections::HashMap;
use std::sync::Arc;

use futures::StreamExt;
use reqwest::{Client, StatusCode};
use serde_json::{json, Value};
use tensorzero::test_helpers::make_embedded_gateway_with_config;
use tensorzero::{
    ClientExt, ClientInferenceParams, ClientInput, ClientInputMessage, ClientInputMessageContent,
    ContentBlockChunk, File, InferenceOutput, InferenceResponse, InferenceResponseChunk, Input,
    InputMessage, InputMessageContent, Role, UnknownChunk, UrlFile,
};
use tensorzero_core::cache::{CacheEnabledMode, CacheOptions};
use tensorzero_core::config::provider_types::ProviderTypesConfig;
use tensorzero_core::db::postgres::PostgresConnectionInfo;
use tensorzero_core::embeddings::{
    Embedding, EmbeddingEncodingFormat, EmbeddingModelConfig, EmbeddingProviderConfig,
    EmbeddingRequest, UninitializedEmbeddingProviderConfig,
};
use tensorzero_core::endpoints::batch_inference::StartBatchInferenceParams;
use tensorzero_core::endpoints::inference::{InferenceClients, InferenceCredentials};
use tensorzero_core::http::TensorzeroHttpClient;
use tensorzero_core::inference::types::{
    ContentBlockChatOutput, Latency, ModelInferenceRequestJsonMode, Text, TextKind,
};
use tensorzero_core::model_table::ProviderTypeDefaultCredentials;
use tensorzero_core::rate_limiting::ScopeInfo;
use tensorzero_core::tool::{
    ProviderTool, ProviderToolScope, ProviderToolScopeModelProvider, ToolCallWrapper,
};
use url::Url;
use uuid::Uuid;

use crate::common::get_gateway_endpoint;
use crate::providers::common::{
    E2ETestProvider, E2ETestProviders, EmbeddingTestProvider, ModelTestProvider,
    DEEPSEEK_PAPER_PDF, FERRIS_PNG,
};
use tensorzero_core::db::clickhouse::test_helpers::{
    get_clickhouse, select_batch_model_inference_clickhouse, select_chat_inference_clickhouse,
    select_model_inference_clickhouse,
};

mod custom_tools;

crate::generate_provider_tests!(get_providers);
crate::generate_batch_inference_tests!(get_providers);
crate::generate_unified_mock_batch_tests!(get_providers);

async fn get_providers() -> E2ETestProviders {
    let credentials = match std::env::var("OPENAI_API_KEY") {
        Ok(key) => HashMap::from([("openai_api_key".to_string(), key)]),
        Err(_) => HashMap::new(),
    };

    let standard_without_o1 = vec![E2ETestProvider {
        supports_batch_inference: true,
        variant_name: "openai".to_string(),
        model_name: "gpt-4o-mini-2024-07-18".into(),
        model_provider_name: "openai".into(),
        credentials: HashMap::new(),
    }];

    let extra_body_providers = vec![E2ETestProvider {
        supports_batch_inference: true,
        variant_name: "openai-extra-body".to_string(),
        model_name: "gpt-4o-mini-2024-07-18".into(),
        model_provider_name: "openai".into(),
        credentials: HashMap::new(),
    }];

    let bad_auth_extra_headers = vec![E2ETestProvider {
        supports_batch_inference: true,
        variant_name: "openai-extra-headers".to_string(),
        model_name: "gpt-4o-mini-2024-07-18".into(),
        model_provider_name: "openai".into(),
        credentials: HashMap::new(),
    }];

    let standard_providers = vec![
        E2ETestProvider {
            supports_batch_inference: true,
            variant_name: "openai".to_string(),
            model_name: "gpt-4o-mini-2024-07-18".into(),
            model_provider_name: "openai".into(),
            credentials: HashMap::new(),
        },
        E2ETestProvider {
            supports_batch_inference: true,
            variant_name: "openai-o1".to_string(),
            model_name: "o1-2024-12-17".into(),
            model_provider_name: "openai".into(),
            credentials: HashMap::new(),
        },
        E2ETestProvider {
            supports_batch_inference: false,
            variant_name: "openai-responses".to_string(),
            model_name: "responses-gpt-4o-mini-2024-07-18".into(),
            model_provider_name: "openai".into(),
            credentials: HashMap::new(),
        },
    ];

    let inference_params_providers = vec![
        E2ETestProvider {
            supports_batch_inference: true,
            variant_name: "openai".to_string(),
            model_name: "gpt-4o-mini-2024-07-18".into(),
            model_provider_name: "openai".into(),
            credentials: credentials.clone(),
        },
        E2ETestProvider {
            supports_batch_inference: false,
            variant_name: "openai-responses".to_string(),
            model_name: "responses-gpt-4o-mini-2024-07-18".into(),
            model_provider_name: "openai".into(),
            credentials: HashMap::new(),
        },
    ];

    let inference_params_dynamic_providers = vec![E2ETestProvider {
        supports_batch_inference: true,
        variant_name: "openai-dynamic".to_string(),
        model_name: "gpt-4o-mini-2024-07-18-dynamic".into(),
        model_provider_name: "openai".into(),
        credentials,
    }];

    let image_providers = vec![
        E2ETestProvider {
            supports_batch_inference: true,
            variant_name: "openai".to_string(),
            model_name: "openai::gpt-4o-mini-2024-07-18".into(),
            model_provider_name: "openai".into(),
            credentials: HashMap::new(),
        },
        E2ETestProvider {
            supports_batch_inference: false,
            variant_name: "openai-responses".to_string(),
            model_name: "responses-gpt-4o-mini-2024-07-18".into(),
            model_provider_name: "openai".into(),
            credentials: HashMap::new(),
        },
    ];

    let input_audio_providers = vec![E2ETestProvider {
        supports_batch_inference: true,
        variant_name: "openai".to_string(),
        model_name: "gpt-4o-audio-preview".into(),
        model_provider_name: "openai".into(),
        credentials: HashMap::new(),
    }];

    let json_providers = vec![
        E2ETestProvider {
            supports_batch_inference: true,
            variant_name: "openai".to_string(),
            model_name: "gpt-4o-mini-2024-07-18".into(),
            model_provider_name: "openai".into(),
            credentials: HashMap::new(),
        },
        E2ETestProvider {
            supports_batch_inference: true,
            variant_name: "openai-implicit".to_string(),
            model_name: "gpt-4o-mini-2024-07-18".into(),
            model_provider_name: "openai".into(),
            credentials: HashMap::new(),
        },
        E2ETestProvider {
            supports_batch_inference: true,
            variant_name: "openai-strict".to_string(),
            model_name: "gpt-4o-mini-2024-07-18".into(),
            model_provider_name: "openai".into(),
            credentials: HashMap::new(),
        },
        E2ETestProvider {
            supports_batch_inference: true,
            variant_name: "openai-o1".to_string(),
            model_name: "o1-2024-12-17".into(),
            model_provider_name: "openai".into(),
            credentials: HashMap::new(),
        },
        E2ETestProvider {
            supports_batch_inference: true,
            variant_name: "openai-cot".to_string(),
            model_name: "openai::gpt-4.1-nano-2025-04-14".into(),
            model_provider_name: "openai".into(),
            credentials: HashMap::new(),
        },
        E2ETestProvider {
            supports_batch_inference: false,
            variant_name: "openai-responses".to_string(),
            model_name: "responses-gpt-4o-mini-2024-07-18".into(),
            model_provider_name: "openai".into(),
            credentials: HashMap::new(),
        },
        E2ETestProvider {
            supports_batch_inference: false,
            variant_name: "openai-responses-strict".to_string(),
            model_name: "responses-gpt-4o-mini-2024-07-18".into(),
            model_provider_name: "openai".into(),
            credentials: HashMap::new(),
        },
    ];

    let json_mode_off_providers = vec![
        E2ETestProvider {
            supports_batch_inference: true,
            variant_name: "openai_json_mode_off".to_string(),
            model_name: "gpt-4o-mini-2024-07-18".into(),
            model_provider_name: "openai".into(),
            credentials: HashMap::new(),
        },
        E2ETestProvider {
            supports_batch_inference: true,
            variant_name: "openai_o1_json_mode_off".to_string(),
            model_name: "o1-2024-12-17".into(),
            model_provider_name: "openai".into(),
            credentials: HashMap::new(),
        },
        E2ETestProvider {
            supports_batch_inference: true,
            variant_name: "openai-responses_json_mode_off".to_string(),
            model_name: "responses-gpt-4o-mini-2024-07-18".into(),
            model_provider_name: "openai".into(),
            credentials: HashMap::new(),
        },
    ];

    let shorthand_providers = vec![E2ETestProvider {
        supports_batch_inference: true,
        variant_name: "openai-shorthand".to_string(),
        model_name: "openai::gpt-4o-mini-2024-07-18".into(),
        model_provider_name: "openai".into(),
        credentials: HashMap::new(),
    }];

    let embedding_providers = vec![EmbeddingTestProvider {
        model_name: "text-embedding-3-small".into(),
        dimensions: 1536,
    }];

    let provider_type_default_credentials_providers = vec![E2ETestProvider {
        supports_batch_inference: true,
        variant_name: "openai".to_string(),
        model_name: "gpt-4o-mini-2024-07-18".into(),
        model_provider_name: "openai".into(),
        credentials: HashMap::new(),
    }];

    let provider_type_default_credentials_shorthand_providers = vec![E2ETestProvider {
        supports_batch_inference: true,
        variant_name: "openai-shorthand".to_string(),
        model_name: "openai::gpt-4o-mini-2024-07-18".into(),
        model_provider_name: "openai".into(),
        credentials: HashMap::new(),
    }];

    let credential_fallbacks = vec![ModelTestProvider {
        provider_type: "openai".to_string(),
        model_info: HashMap::from([(
            "model_name".to_string(),
            "gpt-4o-mini-2024-07-18".to_string(),
        )]),
        use_modal_headers: false,
    }];

    E2ETestProviders {
        simple_inference: standard_providers.clone(),
        extra_body_inference: extra_body_providers,
        bad_auth_extra_headers,
        reasoning_inference: vec![],
        embeddings: embedding_providers,
        inference_params_inference: inference_params_providers,
        inference_params_dynamic_credentials: inference_params_dynamic_providers,
        provider_type_default_credentials: provider_type_default_credentials_providers,
        provider_type_default_credentials_shorthand:
            provider_type_default_credentials_shorthand_providers,
        tool_use_inference: standard_providers.clone(),
        tool_multi_turn_inference: standard_providers.clone(),
        dynamic_tool_use_inference: standard_providers.clone(),
        parallel_tool_use_inference: standard_without_o1.clone(),
        json_mode_inference: json_providers.clone(),
        json_mode_off_inference: json_mode_off_providers.clone(),
        image_inference: image_providers.clone(),
        pdf_inference: image_providers.clone(),
        input_audio: input_audio_providers,
        shorthand_inference: shorthand_providers.clone(),
        credential_fallbacks,
    }
}

#[tokio::test]
pub async fn test_provider_config_extra_body() {
    let episode_id = Uuid::now_v7();

    let payload = json!({
        "function_name": "basic_test",
        "variant_name": "openai-extra-body-provider-config",
        "episode_id": episode_id,
        "params": {
            "chat_completion": {
                "temperature": 9000
            }
        },
        "input":
            {
               "system": {"assistant_name": "Dr. Mehta"},
               "messages": [
                {
                    "role": "user",
                    "content": "What is the name of the capital city of Japan?"
                }
            ]},
        "stream": false,
        "tags": {"foo": "bar"},
    });

    let response = Client::new()
        .post(get_gateway_endpoint("/inference"))
        .json(&payload)
        .send()
        .await
        .unwrap();

    // Check that the API response is ok
    assert_eq!(response.status(), StatusCode::OK);
    let response_json = response.json::<Value>().await.unwrap();

    println!("API response: {response_json:#?}");

    let inference_id = response_json.get("inference_id").unwrap().as_str().unwrap();
    let inference_id = Uuid::parse_str(inference_id).unwrap();

    // Sleep to allow time for data to be inserted into ClickHouse (trailing writes from API)
    tokio::time::sleep(std::time::Duration::from_millis(200)).await;

    // Check if ClickHouse is ok - ChatInference Table
    let clickhouse = get_clickhouse().await;

    // Check the ModelInference Table
    let result = select_model_inference_clickhouse(&clickhouse, inference_id)
        .await
        .unwrap();

    println!("ClickHouse - ModelInference: {result:#?}");

    let model_inference_id = result.get("id").unwrap().as_str().unwrap();
    assert!(Uuid::parse_str(model_inference_id).is_ok());

    let inference_id_result = result.get("inference_id").unwrap().as_str().unwrap();
    let inference_id_result = Uuid::parse_str(inference_id_result).unwrap();
    assert_eq!(inference_id_result, inference_id);

    let raw_request = result.get("raw_request").unwrap().as_str().unwrap();
    let raw_request_val: serde_json::Value = serde_json::from_str::<Value>(raw_request).unwrap();

    // This is set in both the variant and model provider extra_body, and
    // so the model provider should win
    assert_eq!(
        raw_request_val
            .get("temperature")
            .unwrap()
            .as_f64()
            .expect("Temperature is not a number"),
        0.456
    );

    // This is only set in the variant extra_body
    assert_eq!(
        raw_request_val
            .get("max_completion_tokens")
            .unwrap()
            .as_u64()
            .expect("max_completion_tokens is not a number"),
        123
    );

    // This is only set in the model provider extra_body
    assert_eq!(
        raw_request_val
            .get("frequency_penalty")
            .unwrap()
            .as_f64()
            .expect("frequency_penalty is not a number"),
        1.42
    );
}

#[tokio::test]
async fn test_default_function_default_tool_choice() {
    let client = Client::new();
    let episode_id = Uuid::now_v7();

    let payload = json!({
        "model_name": "openai::gpt-4o-mini",
        "episode_id" : episode_id,
        "input": {
            "messages": [{"role": "user", "content": "What is the weather in NYC?"}],
        },
        "additional_tools": [
            {
                "name": "temperature_api",
                "description": "Get the current temperature",
                "parameters": {
                  "$schema": "http://json-schema.org/draft-07/schema#",
                  "type": "object",
                  "description": "Get the current temperature in Celsius for a given location.",
                  "properties": {
                    "location": {
                      "type": "string",
                      "description": "The location to get the temperature for (e.g. \"New York\")"
                    }
                  },
                  "required": ["location"],
                  "additionalProperties": false
                }
                ,
                "strict": true
            }
        ],
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
    println!("Response: {response_json}");

    // Check that inference_id is here
    let inference_id = response_json.get("inference_id").unwrap().as_str().unwrap();
    let inference_id = Uuid::parse_str(inference_id).unwrap();

    let content_blocks = response_json.get("content").unwrap().as_array().unwrap();
    assert_eq!(content_blocks.len(), 1);

    assert_eq!(
        content_blocks[0].get("type").unwrap().as_str().unwrap(),
        "tool_call"
    );

    // Sleep for 200ms second to allow time for data to be inserted into ClickHouse (trailing writes from API)
    tokio::time::sleep(std::time::Duration::from_millis(200)).await;

    // Just check 'tool_choice' in the raw request, since we already have lots of tests
    // that check the full ChatInference/ModelInference rows
    let clickhouse = get_clickhouse().await;

    // Check the ModelInference Table
    let result = select_model_inference_clickhouse(&clickhouse, inference_id)
        .await
        .unwrap();
    let inference_id_result = result.get("inference_id").unwrap().as_str().unwrap();
    let inference_id_result = Uuid::parse_str(inference_id_result).unwrap();
    assert_eq!(inference_id_result, inference_id);
    let raw_request = result.get("raw_request").unwrap().as_str().unwrap();
    let raw_request_json: Value = serde_json::from_str(raw_request).unwrap();
    assert_eq!(raw_request_json["tool_choice"], "auto");
}

// Tests using 'model_name' with a shorthand model

#[tokio::test]
async fn test_default_function_model_name_shorthand() {
    let client = Client::new();
    let episode_id = Uuid::now_v7();

    let payload = json!({
        "model_name": "openai::o4-mini",
        "episode_id": episode_id,
        "input": {
            "messages": [
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
    assert!(content_blocks.len() == 1);
    let content_block = content_blocks.first().unwrap();
    let content_block_type = content_block.get("type").unwrap().as_str().unwrap();
    assert_eq!(content_block_type, "text");
    let content = content_block.get("text").unwrap().as_str().unwrap();
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
                "content": [{"type": "text", "text": "What is the capital of Japan?"}]
            }
        ]
    });
    assert_eq!(input, correct_input);
    let content_blocks = result.get("output").unwrap().as_str().unwrap();
    // Check that content_blocks is a list of blocks length 1
    let content_blocks: Vec<Value> = serde_json::from_str(content_blocks).unwrap();
    assert_eq!(content_blocks.len(), 1);
    let content_block = content_blocks.first().unwrap();
    // Check the type and content in the block
    let content_block_type = content_block.get("type").unwrap().as_str().unwrap();
    assert_eq!(content_block_type, "text");
    let clickhouse_content = content_block.get("text").unwrap().as_str().unwrap();
    assert_eq!(clickhouse_content, content);
    // Check that episode_id is here and correct
    let retrieved_episode_id = result.get("episode_id").unwrap().as_str().unwrap();
    let retrieved_episode_id = Uuid::parse_str(retrieved_episode_id).unwrap();
    assert_eq!(retrieved_episode_id, episode_id);
    // Check the variant name
    let variant_name = result.get("variant_name").unwrap().as_str().unwrap();
    assert_eq!(variant_name, "openai::o4-mini");
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
    assert_eq!(model_name, "openai::o4-mini");
    let model_provider_name = result.get("model_provider_name").unwrap().as_str().unwrap();
    assert_eq!(model_provider_name, "openai");
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
}

// Tests using 'model_name' with a non-shorthand model

#[tokio::test]
async fn test_default_function_model_name_non_shorthand() {
    let client = Client::new();
    let episode_id = Uuid::now_v7();

    let payload = json!({
        "model_name": "o4-mini",
        "episode_id": episode_id,
        "input": {
            "messages": [
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
    assert!(content_blocks.len() == 1);
    let content_block = content_blocks.first().unwrap();
    let content_block_type = content_block.get("type").unwrap().as_str().unwrap();
    assert_eq!(content_block_type, "text");
    let content = content_block.get("text").unwrap().as_str().unwrap();
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
                "content": [{"type": "text", "text": "What is the capital of Japan?"}]
            }
        ]
    });
    assert_eq!(input, correct_input);
    let content_blocks = result.get("output").unwrap().as_str().unwrap();
    // Check that content_blocks is a list of blocks length 1
    let content_blocks: Vec<Value> = serde_json::from_str(content_blocks).unwrap();
    assert_eq!(content_blocks.len(), 1);
    let content_block = content_blocks.first().unwrap();
    // Check the type and content in the block
    let content_block_type = content_block.get("type").unwrap().as_str().unwrap();
    assert_eq!(content_block_type, "text");
    let clickhouse_content = content_block.get("text").unwrap().as_str().unwrap();
    assert_eq!(clickhouse_content, content);
    // Check that episode_id is here and correct
    let retrieved_episode_id = result.get("episode_id").unwrap().as_str().unwrap();
    let retrieved_episode_id = Uuid::parse_str(retrieved_episode_id).unwrap();
    assert_eq!(retrieved_episode_id, episode_id);
    // Check the variant name
    let variant_name = result.get("variant_name").unwrap().as_str().unwrap();
    assert_eq!(variant_name, "o4-mini");
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
    assert_eq!(model_name, "o4-mini");
    let model_provider_name = result.get("model_provider_name").unwrap().as_str().unwrap();
    assert_eq!(model_provider_name, "openai");
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
}

// Tests using 'model_name' with a non-shorthand model

#[tokio::test]
async fn test_default_function_invalid_model_name() {
    use reqwest::StatusCode;

    let client = Client::new();
    let episode_id = Uuid::now_v7();

    let payload = json!({
        "model_name": "openai::my-bad-model-name",
        "episode_id": episode_id,
        "input": {
            "messages": [
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
    let status = response.status();
    let text = response.text().await.unwrap();
    assert!(
        text.contains("`my-bad-model-name` does not exist"),
        "Unexpected error: {text}"
    );
    assert_eq!(status, StatusCode::BAD_GATEWAY);
}

#[tokio::test]
async fn test_chat_function_json_override_with_mode_on() {
    test_chat_function_json_override_with_mode(ModelInferenceRequestJsonMode::On).await;
}

#[tokio::test]
async fn test_chat_function_json_override_with_mode_off() {
    test_chat_function_json_override_with_mode(ModelInferenceRequestJsonMode::Off).await;
}

#[tokio::test]
async fn test_chat_function_json_override_with_mode_strict() {
    test_chat_function_json_override_with_mode(ModelInferenceRequestJsonMode::Strict).await;
}

async fn test_chat_function_json_override_with_mode(json_mode: ModelInferenceRequestJsonMode) {
    let client = Client::new();
    let episode_id = Uuid::now_v7();
    let mode = serde_json::to_value(json_mode).unwrap();

    // Note that we need to include 'json' somewhere in the messages, to stop OpenAI from complaining
    let payload = json!({
        "function_name": "basic_test",
        "variant_name": "openai",
        "episode_id": episode_id,
        "input":
            {"system": {"assistant_name": "AskJeeves"},
            "messages": [
                {
                    "role": "user",
                    "content": "What is the capital of Japan (possibly as JSON)?"
                }
            ]},
        "params": {
            "chat_completion": {
                "json_mode": mode,
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
    let response_status = response.status();
    let response_json = response.json::<Value>().await.unwrap();
    // Check Response is OK, then fields in order
    assert_eq!(
        response_status,
        StatusCode::OK,
        "Unexpected response status, body: {response_json:?})"
    );
    let content_blocks = response_json.get("content").unwrap().as_array().unwrap();
    assert!(content_blocks.len() == 1);
    let content_block = content_blocks.first().unwrap();
    let content_block_type = content_block.get("type").unwrap().as_str().unwrap();
    // We should have forwarded the JSON mode to the provider, so OpenAI should give us back json
    // Since we're using a text function, tensorzero should not handle the JSON parsing for us.
    assert_eq!(content_block_type, "text");
    let content = content_block.get("text").unwrap().as_str().unwrap();
    // Assert that Tokyo is in the content
    assert!(content.contains("Tokyo"), "Content should mention Tokyo");
    if let ModelInferenceRequestJsonMode::On | ModelInferenceRequestJsonMode::Strict = json_mode {
        let _context_as_json: Value =
            serde_json::from_str(content).expect("Content should be valid JSON");
    }
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
    let input: Value =
        serde_json::from_str(result.get("input").unwrap().as_str().unwrap()).unwrap();
    let correct_input = json!({
        "system": {"assistant_name": "AskJeeves"},
        "messages": [
            {
                "role": "user",
                "content": [{"type": "text", "text": "What is the capital of Japan (possibly as JSON)?"}]
            }
        ]
    });
    assert_eq!(input, correct_input);
    let content_blocks = result.get("output").unwrap().as_str().unwrap();
    // Check that content_blocks is a list of blocks length 1
    let content_blocks: Vec<Value> = serde_json::from_str(content_blocks).unwrap();
    assert_eq!(content_blocks.len(), 1);
    let content_block = content_blocks.first().unwrap();
    // Check the type and content in the block
    let content_block_type = content_block.get("type").unwrap().as_str().unwrap();
    assert_eq!(content_block_type, "text");
    let clickhouse_content = content_block.get("text").unwrap().as_str().unwrap();
    assert_eq!(clickhouse_content, content);
    // Check that episode_id is here and correct
    let retrieved_episode_id = result.get("episode_id").unwrap().as_str().unwrap();
    let retrieved_episode_id = Uuid::parse_str(retrieved_episode_id).unwrap();
    assert_eq!(retrieved_episode_id, episode_id);
    // Check the variant name
    let variant_name = result.get("variant_name").unwrap().as_str().unwrap();
    assert_eq!(variant_name, "openai");
    // Check the processing time
    let processing_time_ms = result.get("processing_time_ms").unwrap().as_u64().unwrap();
    assert!(processing_time_ms > 0);

    // Check that we saved the correct json mode to ClickHouse
    let inference_params = result.get("inference_params").unwrap().as_str().unwrap();
    let inference_params: Value = serde_json::from_str(inference_params).unwrap();
    let expected_json_mode = serde_json::to_value(json_mode).unwrap();
    let clickhouse_json_mode = inference_params
        .get("chat_completion")
        .unwrap()
        .get("json_mode")
        .unwrap()
        .as_str()
        .unwrap();
    assert_eq!(expected_json_mode, clickhouse_json_mode);

    // Check the ModelInference Table
    let result = select_model_inference_clickhouse(&clickhouse, inference_id)
        .await
        .unwrap();
    let inference_id_result = result.get("inference_id").unwrap().as_str().unwrap();
    let inference_id_result = Uuid::parse_str(inference_id_result).unwrap();
    assert_eq!(inference_id_result, inference_id);
    let model_name = result.get("model_name").unwrap().as_str().unwrap();
    assert_eq!(model_name, "gpt-4o-mini-2024-07-18");
    let model_provider_name = result.get("model_provider_name").unwrap().as_str().unwrap();
    assert_eq!(model_provider_name, "openai");
    let raw_request = result.get("raw_request").unwrap().as_str().unwrap();
    assert!(raw_request.to_lowercase().contains("japan"));
    // Check that raw_request is valid JSON
    let raw_request_val: Value =
        serde_json::from_str(raw_request).expect("raw_request should be valid JSON");
    let expected_request = match json_mode {
        ModelInferenceRequestJsonMode::Off => {
            json!({"messages":[{"role":"system","content":"You are a helpful and friendly assistant named AskJeeves"},{"role":"user","content":"What is the capital of Japan (possibly as JSON)?"}],"model":"gpt-4o-mini-2024-07-18","max_completion_tokens":100,"stream":false})
        }
        ModelInferenceRequestJsonMode::On => {
            json!({"messages":[{"role":"system","content":"You are a helpful and friendly assistant named AskJeeves"},{"role":"user","content":"What is the capital of Japan (possibly as JSON)?"}],"model":"gpt-4o-mini-2024-07-18","max_completion_tokens":100,"stream":false,"response_format":{"type":"json_object"}})
        }
        ModelInferenceRequestJsonMode::Strict => {
            json!({"messages":[{"role":"system","content":"You are a helpful and friendly assistant named AskJeeves"},{"role":"user","content":"What is the capital of Japan (possibly as JSON)?"}],"model":"gpt-4o-mini-2024-07-18","max_completion_tokens":100,"stream":false,"response_format":{"type":"json_object"}})
        }
    };
    assert_eq!(raw_request_val, expected_request);
    let input_tokens = result.get("input_tokens").unwrap().as_u64().unwrap();
    assert!(input_tokens > 5);
    let output_tokens = result.get("output_tokens").unwrap().as_u64().unwrap();
    assert!(output_tokens > 5);
    let response_time_ms = result.get("response_time_ms").unwrap().as_u64().unwrap();
    assert!(response_time_ms > 0);
    assert!(result.get("ttft_ms").unwrap().is_null());
    let raw_response = result.get("raw_response").unwrap().as_str().unwrap();
    let _raw_response_json: Value = serde_json::from_str(raw_response).unwrap();
}

#[tokio::test]
async fn test_o4_mini_inference() {
    let client = Client::new();
    let episode_id = Uuid::now_v7();

    let payload = json!({
        "function_name": "basic_test",
        "variant_name": "o4-mini",
        "episode_id": episode_id,
        "input":
            {"system": {"assistant_name": "AskJeeves"},
            "messages": [
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
    assert!(content_blocks.len() == 1);
    let content_block = content_blocks.first().unwrap();
    let content_block_type = content_block.get("type").unwrap().as_str().unwrap();
    assert_eq!(content_block_type, "text");
    let content = content_block.get("text").unwrap().as_str().unwrap();
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
    let input: Value =
        serde_json::from_str(result.get("input").unwrap().as_str().unwrap()).unwrap();
    let correct_input = json!({
        "system": {"assistant_name": "AskJeeves"},
        "messages": [
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
    assert_eq!(content_blocks.len(), 1);
    let content_block = content_blocks.first().unwrap();
    // Check the type and content in the block
    let content_block_type = content_block.get("type").unwrap().as_str().unwrap();
    assert_eq!(content_block_type, "text");
    let clickhouse_content = content_block.get("text").unwrap().as_str().unwrap();
    assert_eq!(clickhouse_content, content);
    // Check that episode_id is here and correct
    let retrieved_episode_id = result.get("episode_id").unwrap().as_str().unwrap();
    let retrieved_episode_id = Uuid::parse_str(retrieved_episode_id).unwrap();
    assert_eq!(retrieved_episode_id, episode_id);
    // Check the variant name
    let variant_name = result.get("variant_name").unwrap().as_str().unwrap();
    assert_eq!(variant_name, "o4-mini");
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
    assert_eq!(model_name, "o4-mini");
    let model_provider_name = result.get("model_provider_name").unwrap().as_str().unwrap();
    assert_eq!(model_provider_name, "openai");
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
}

#[tokio::test]
async fn test_o3_mini_inference_with_reasoning_effort() {
    let client = Client::new();
    let episode_id = Uuid::now_v7();

    let payload = json!({
        "function_name": "basic_test",
        "variant_name": "o3-mini",
        "episode_id": episode_id,
        "input":
            {"system": {"assistant_name": "AskJeeves"},
            "messages": [
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
    // assert_eq!(response.status(), StatusCode::OK);
    let response_json = response.json::<Value>().await.unwrap();
    println!("Response JSON: {response_json:?}");

    let content_blocks = response_json.get("content").unwrap().as_array().unwrap();
    assert!(content_blocks.len() == 1);
    let content_block = content_blocks.first().unwrap();
    let content_block_type = content_block.get("type").unwrap().as_str().unwrap();
    assert_eq!(content_block_type, "text");
    let content = content_block.get("text").unwrap().as_str().unwrap();

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
    let input: Value =
        serde_json::from_str(result.get("input").unwrap().as_str().unwrap()).unwrap();
    let correct_input = json!({
        "system": {"assistant_name": "AskJeeves"},
        "messages": [
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
    assert_eq!(content_blocks.len(), 1);
    let content_block = content_blocks.first().unwrap();

    // Check the type and content in the block
    let content_block_type = content_block.get("type").unwrap().as_str().unwrap();
    assert_eq!(content_block_type, "text");
    let clickhouse_content = content_block.get("text").unwrap().as_str().unwrap();
    assert_eq!(clickhouse_content, content);

    // Check that episode_id is here and correct
    let retrieved_episode_id = result.get("episode_id").unwrap().as_str().unwrap();
    let retrieved_episode_id = Uuid::parse_str(retrieved_episode_id).unwrap();
    assert_eq!(retrieved_episode_id, episode_id);

    // Check the variant name
    let variant_name = result.get("variant_name").unwrap().as_str().unwrap();
    assert_eq!(variant_name, "o3-mini");

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
    assert_eq!(model_name, "o3-mini");
    let model_provider_name = result.get("model_provider_name").unwrap().as_str().unwrap();
    assert_eq!(model_provider_name, "openai");
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
}

#[tokio::test]
async fn test_embedding_request() {
    let clickhouse = get_clickhouse().await;
    let provider_config_serialized = r#"
    type = "openai"
    model_name = "text-embedding-3-small"
    "#;
    let provider_config =
        toml::from_str::<UninitializedEmbeddingProviderConfig>(provider_config_serialized)
            .expect("Failed to deserialize EmbeddingProviderConfig")
            .load(
                &ProviderTypesConfig::default(),
                Arc::from("good".to_string()),
                &ProviderTypeDefaultCredentials::default(),
                TensorzeroHttpClient::new_testing().unwrap(),
            )
            .await
            .unwrap();
    assert!(matches!(
        provider_config.inner,
        EmbeddingProviderConfig::OpenAI(_)
    ));

    // Inject randomness into the model request to ensure that the first request
    // is a cache miss
    let model_name = "my_embedding".to_string();

    let model_config = EmbeddingModelConfig {
        routing: vec![model_name.as_str().into()],
        providers: [(model_name.as_str().into(), provider_config)]
            .into_iter()
            .collect(),
        timeout_ms: None,
    };

    let request = EmbeddingRequest {
        input: format!("This is a test input: {}", Uuid::now_v7()).into(),
        dimensions: None,
        encoding_format: EmbeddingEncodingFormat::Float,
    };
    let api_keys = InferenceCredentials::default();
    let response = model_config
        .embed(
            &request,
            &model_name,
            &InferenceClients {
                http_client: TensorzeroHttpClient::new_testing().unwrap(),
                clickhouse_connection_info: clickhouse.clone(),
                postgres_connection_info: PostgresConnectionInfo::Disabled,
                credentials: Arc::new(api_keys.clone()),
                cache_options: CacheOptions {
                    max_age_s: None,
                    enabled: CacheEnabledMode::On,
                },
                tags: Arc::new(Default::default()),
                rate_limiting_config: Arc::new(Default::default()),
                otlp_config: Default::default(),
                deferred_tasks: tokio_util::task::TaskTracker::new(),
                scope_info: ScopeInfo {
                    tags: Arc::new(HashMap::new()),
                    api_key_public_id: None,
                },
            },
        )
        .await
        .unwrap();
    assert!(
        !response.cached,
        "Response was incorrectly cached: {response:?}"
    );
    let [first_embedding] = response.embeddings.as_slice() else {
        panic!("Expected exactly one embedding");
    };
    assert_eq!(first_embedding.ndims(), 1536);
    assert!(!response.cached);
    // Calculate the L2 norm of the embedding
    let norm: f32 = first_embedding
        .clone()
        .as_float()
        .unwrap()
        .iter()
        .map(|&x| x.powi(2))
        .sum::<f32>()
        .sqrt();

    // Assert that the norm is approximately 1 (allowing for small floating-point errors)
    assert!(
        (norm - 1.0).abs() < 1e-6,
        "The L2 norm of the embedding should be 1, but it is {norm}"
    );
    // Check that the timestamp in created is within 1 second of the current time
    let created = response.created;
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .expect("Time went backwards")
        .as_secs() as i64;
    assert!(
        (created as i64 - now).abs() <= 1,
        "The created timestamp should be within 1 second of the current time, but it is {created}"
    );
    let parsed_raw_response: Value = serde_json::from_str(&response.raw_response).unwrap();
    assert!(
        !parsed_raw_response.is_null(),
        "Parsed raw response should not be null"
    );
    let parsed_raw_request: Value = serde_json::from_str(&response.raw_request).unwrap();
    assert!(
        !parsed_raw_request.is_null(),
        "Parsed raw request should not be null"
    );
    // The randomness affects the exact number of tokens, so we just check that it's at least 20
    assert!(
        response.usage.input_tokens.unwrap() >= 20,
        "Unexpected input tokens: {:?}",
        response.usage.input_tokens
    );
    assert_eq!(response.usage.output_tokens, Some(0));
    match response.latency {
        Latency::NonStreaming { response_time } => {
            assert!(
                response_time.as_millis() > 100,
                "Response time should be greater than 100ms: {}",
                response_time.as_millis()
            );
        }
        _ => panic!("Latency should be non-streaming"),
    }

    // Wait for ClickHouse write
    tokio::time::sleep(std::time::Duration::from_secs(1)).await;
    let cached_response = model_config
        .embed(
            &request,
            &model_name,
            &InferenceClients {
                http_client: TensorzeroHttpClient::new_testing().unwrap(),
                clickhouse_connection_info: clickhouse.clone(),
                postgres_connection_info: PostgresConnectionInfo::Disabled,
                credentials: Arc::new(api_keys.clone()),
                cache_options: CacheOptions {
                    max_age_s: None,
                    enabled: CacheEnabledMode::On,
                },
                tags: Arc::new(Default::default()),
                rate_limiting_config: Arc::new(Default::default()),
                otlp_config: Default::default(),
                deferred_tasks: tokio_util::task::TaskTracker::new(),
                scope_info: ScopeInfo {
                    tags: Arc::new(HashMap::new()),
                    api_key_public_id: None,
                },
            },
        )
        .await
        .unwrap();
    assert!(cached_response.cached);
    assert_eq!(response.embeddings, cached_response.embeddings);
    assert!(
        cached_response.usage.input_tokens.unwrap() >= 20,
        "Unexpected input tokens: {:?}",
        cached_response.usage.input_tokens
    );
    assert_eq!(cached_response.usage.output_tokens, Some(0));
}

#[tokio::test]
async fn test_embedding_sanity_check() {
    let clickhouse = get_clickhouse().await;
    let provider_config_serialized = r#"
    type = "openai"
    model_name = "text-embedding-3-small"
    "#;
    let provider_config =
        toml::from_str::<UninitializedEmbeddingProviderConfig>(provider_config_serialized)
            .expect("Failed to deserialize EmbeddingProviderConfig")
            .load(
                &ProviderTypesConfig::default(),
                Arc::from("good".to_string()),
                &ProviderTypeDefaultCredentials::default(),
                TensorzeroHttpClient::new_testing().unwrap(),
            )
            .await
            .unwrap();
    let client = TensorzeroHttpClient::new_testing().unwrap();
    let embedding_request_a = EmbeddingRequest {
        input: "Joe Biden is the president of the United States"
            .to_string()
            .into(),
        dimensions: None,
        encoding_format: EmbeddingEncodingFormat::Float,
    };

    let embedding_request_b = EmbeddingRequest {
        input: "Kamala Harris is the vice president of the United States"
            .to_string()
            .into(),
        dimensions: None,
        encoding_format: EmbeddingEncodingFormat::Float,
    };

    let embedding_request_c = EmbeddingRequest {
        input: "My favorite systems programming language is Rust"
            .to_string()
            .into(),
        dimensions: None,
        encoding_format: EmbeddingEncodingFormat::Float,
    };
    let request_info = (&provider_config).into();
    let api_keys = InferenceCredentials::default();
    let clients = InferenceClients {
        http_client: client.clone(),
        clickhouse_connection_info: clickhouse.clone(),
        postgres_connection_info: PostgresConnectionInfo::Disabled,
        credentials: Arc::new(api_keys),
        cache_options: CacheOptions {
            max_age_s: None,
            enabled: CacheEnabledMode::On,
        },
        tags: Arc::new(Default::default()),
        rate_limiting_config: Arc::new(Default::default()),
        otlp_config: Default::default(),
        deferred_tasks: tokio_util::task::TaskTracker::new(),
        scope_info: ScopeInfo {
            tags: Arc::new(HashMap::new()),
            api_key_public_id: None,
        },
    };

    // Compute all 3 embeddings concurrently
    let (response_a, response_b, response_c) = tokio::join!(
        provider_config.embed(&embedding_request_a, &clients, &request_info),
        provider_config.embed(&embedding_request_b, &clients, &request_info),
        provider_config.embed(&embedding_request_c, &clients, &request_info)
    );

    // Unwrap the results
    let response_a = response_a.expect("Failed to get embedding for request A");
    let response_b = response_b.expect("Failed to get embedding for request B");
    let response_c = response_c.expect("Failed to get embedding for request C");
    let [embedding_a] = response_a.embeddings.as_slice() else {
        panic!("Failed to get embedding for request A");
    };
    let [embedding_b] = response_b.embeddings.as_slice() else {
        panic!("Failed to get embedding for request b");
    };
    let [embedding_c] = response_c.embeddings.as_slice() else {
        panic!("Failed to get embedding for request C");
    };

    // Calculate cosine similarities
    let similarity_ab = cosine_similarity(embedding_a, embedding_b);
    let similarity_ac = cosine_similarity(embedding_a, embedding_c);
    let similarity_bc = cosine_similarity(embedding_b, embedding_c);

    // Assert that semantically similar sentences have higher similarity (with a margin of 0.3)
    // We empirically determined this by staring at it (no science to it)
    assert!(
        similarity_ab - similarity_ac > 0.3,
        "Similarity between A and B should be higher than between A and C"
    );
    assert!(
        similarity_ab - similarity_bc > 0.3,
        "Similarity between A and B should be higher than between B and C"
    );
}

fn cosine_similarity(a: &Embedding, b: &Embedding) -> f32 {
    let a = a.clone();
    let b = b.clone();
    let a_float = a.as_float().unwrap();
    let b_float = b.as_float().unwrap();
    let dot_product: f32 = a_float.iter().zip(b_float.iter()).map(|(x, y)| x * y).sum();
    let magnitude_a: f32 = a_float.iter().map(|x| x * x).sum::<f32>().sqrt();
    let magnitude_b: f32 = b_float.iter().map(|x| x * x).sum::<f32>().sqrt();
    dot_product / (magnitude_a * magnitude_b)
}

// We already test Amazon S3 with all image providers, so let's test Cloudflare R2
// (which is S3-compatible) with just OpenAI to save time and money.

#[tokio::test]
pub async fn test_image_inference_with_provider_cloudflare_r2() {
    use crate::providers::common::test_image_inference_with_provider_s3_compatible;
    use object_store::{aws::AmazonS3Builder, ObjectStore};
    use rand::distr::Alphanumeric;
    use rand::distr::SampleString;
    use std::sync::Arc;
    use tensorzero_core::inference::types::storage::StorageKind;

    // We expect CI to provide our credentials in 'R2_' variables
    // (to avoid conflicting with the normal AWS credentials for bedrock)
    let r2_access_key_id = std::env::var("R2_ACCESS_KEY_ID").unwrap();
    let r2_secret_access_key = std::env::var("R2_SECRET_ACCESS_KEY").unwrap();

    // Our S3-compatible object store checks for these variables, giving them
    // higher priority than the normal 'AWS_ACCESS_KEY_ID'/'AWS_SECRET_ACCESS_KEY' vars
    std::env::set_var("S3_ACCESS_KEY_ID", &r2_access_key_id);
    std::env::set_var("S3_SECRET_ACCESS_KEY", &r2_secret_access_key);

    let provider = E2ETestProvider {
        supports_batch_inference: true,
        variant_name: "openai".to_string(),
        model_name: "openai::gpt-4o-mini-2024-07-18".into(),
        model_provider_name: "openai".into(),
        credentials: HashMap::new(),
    };

    let endpoint = "https://19918a216783f0ac9e052233569aef60.r2.cloudflarestorage.com/tensorzero-e2e-test-images".to_string();

    let test_bucket = "tensorzero-e2e-test-images";

    let client: Arc<dyn ObjectStore> = Arc::new(
        AmazonS3Builder::new()
            .with_bucket_name(test_bucket)
            .with_access_key_id(r2_access_key_id)
            .with_secret_access_key(r2_secret_access_key)
            .with_endpoint(&endpoint)
            .build()
            .unwrap(),
    );

    let mut prefix = Alphanumeric.sample_string(&mut rand::rng(), 6);
    prefix += "-";

    test_image_inference_with_provider_s3_compatible(
        provider,
        &StorageKind::S3Compatible {
            bucket_name: Some(test_bucket.to_string()),
            region: None,
            prefix: prefix.clone(),
            endpoint: Some(endpoint.clone()),
            allow_http: None,
        },
        &client,
        &format!(
            r#"
    [object_storage]
    type = "s3_compatible"
    endpoint = "{endpoint}"
    bucket_name = "{test_bucket}"
    prefix = "{prefix}"

    [functions.image_test]
    type = "chat"

    [functions.image_test.variants.openai]
    type = "chat_completion"
    model = "openai::gpt-4o-mini-2024-07-18"
    "#
        ),
        &prefix,
    )
    .await;
}

// Tests using `{"type": "text", "text": "Some string"}` as input
#[tokio::test]
async fn test_content_block_text_field() {
    let client = Client::new();
    let episode_id = Uuid::now_v7();

    let payload = json!({
        "model_name": "openai::o4-mini",
        "episode_id": episode_id,
        "input": {
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": "What is the capital of Japan?"}]
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
    assert!(content_blocks.len() == 1);
    let content_block = content_blocks.first().unwrap();
    let content_block_type = content_block.get("type").unwrap().as_str().unwrap();
    assert_eq!(content_block_type, "text");
    let content = content_block.get("text").unwrap().as_str().unwrap();
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
                "content": [{"type": "text", "text": "What is the capital of Japan?"}]
            }
        ]
    });
    assert_eq!(input, correct_input);
    let content_blocks = result.get("output").unwrap().as_str().unwrap();
    // Check that content_blocks is a list of blocks length 1
    let content_blocks: Vec<Value> = serde_json::from_str(content_blocks).unwrap();
    assert_eq!(content_blocks.len(), 1);
    let content_block = content_blocks.first().unwrap();
    // Check the type and content in the block
    let content_block_type = content_block.get("type").unwrap().as_str().unwrap();
    assert_eq!(content_block_type, "text");
    let clickhouse_content = content_block.get("text").unwrap().as_str().unwrap();
    assert_eq!(clickhouse_content, content);
    // Check that episode_id is here and correct
    let retrieved_episode_id = result.get("episode_id").unwrap().as_str().unwrap();
    let retrieved_episode_id = Uuid::parse_str(retrieved_episode_id).unwrap();
    assert_eq!(retrieved_episode_id, episode_id);
    // Check the variant name
    let variant_name = result.get("variant_name").unwrap().as_str().unwrap();
    assert_eq!(variant_name, "openai::o4-mini");
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
    assert_eq!(model_name, "openai::o4-mini");
    let model_provider_name = result.get("model_provider_name").unwrap().as_str().unwrap();
    assert_eq!(model_provider_name, "openai");
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
}

// We already test Amazon S3 with all image providers, so let's test Google Cloud Storage
// (which is S3-compatible) with just OpenAI to save time and money.

#[tokio::test(flavor = "multi_thread")]
pub async fn test_image_inference_with_provider_gcp_storage() {
    use crate::providers::common::test_image_inference_with_provider_s3_compatible;
    use crate::providers::common::IMAGE_FUNCTION_CONFIG;
    use object_store::{aws::AmazonS3Builder, ObjectStore};
    use rand::distr::Alphanumeric;
    use rand::distr::SampleString;
    use std::sync::Arc;
    use tensorzero_core::inference::types::storage::StorageKind;

    // We expect CI to provide our credentials in 'GCP_STORAGE_' variables
    // (to avoid conflicting with the normal AWS credentials for bedrock)
    let gcloud_access_key_id = std::env::var("GCP_STORAGE_ACCESS_KEY_ID").unwrap();
    let gcloud_secret_access_key = std::env::var("GCP_STORAGE_SECRET_ACCESS_KEY").unwrap();

    // Our S3-compatible object store checks for these variables, giving them
    // higher priority than the normal 'AWS_ACCESS_KEY_ID'/'AWS_SECRET_ACCESS_KEY' vars
    std::env::set_var("S3_ACCESS_KEY_ID", &gcloud_access_key_id);
    std::env::set_var("S3_SECRET_ACCESS_KEY", &gcloud_secret_access_key);

    let provider = E2ETestProvider {
        supports_batch_inference: true,
        variant_name: "openai".to_string(),
        model_name: "openai::gpt-4o-mini-2024-07-18".into(),
        model_provider_name: "openai".into(),
        credentials: HashMap::new(),
    };

    let endpoint = "https://storage.googleapis.com".to_string();

    let test_bucket = "tensorzero-e2e-tests";

    let client: Arc<dyn ObjectStore> = Arc::new(
        AmazonS3Builder::new()
            .with_bucket_name(test_bucket)
            .with_access_key_id(gcloud_access_key_id)
            .with_secret_access_key(gcloud_secret_access_key)
            .with_endpoint(&endpoint)
            .build()
            .unwrap(),
    );

    let mut prefix = Alphanumeric.sample_string(&mut rand::rng(), 6);
    prefix += "-";

    test_image_inference_with_provider_s3_compatible(
        provider,
        &StorageKind::S3Compatible {
            bucket_name: Some(test_bucket.to_string()),
            region: None,
            prefix: prefix.clone(),
            endpoint: Some(endpoint.clone()),
            allow_http: None,
        },
        &client,
        &format!(
            r#"
    [object_storage]
    type = "s3_compatible"
    endpoint = "{endpoint}"
    bucket_name = "{test_bucket}"
    prefix = "{prefix}"

    {IMAGE_FUNCTION_CONFIG}
    "#
        ),
        &prefix,
    )
    .await;
}

// We already test Amazon S3 with all image providers, so let's test minio
// (which is S3-compatible) with just OpenAI to save time and money.

#[tokio::test]
pub async fn test_image_inference_with_provider_docker_minio() {
    use crate::providers::common::test_image_inference_with_provider_s3_compatible;
    use object_store::{aws::AmazonS3Builder, ObjectStore};
    use rand::distr::Alphanumeric;
    use rand::distr::SampleString;
    use std::sync::Arc;
    use tensorzero_core::inference::types::storage::StorageKind;

    // These are set in `ci/minio-docker-compose.yml`
    let minio_access_key_id = "tensorzero-root".to_string();
    let minio_secret_access_key = "tensorzero-root".to_string();

    // Our S3-compatible  store checks for these variables, giving them
    // higher priority than the normal 'AWS_ACCESS_KEY_ID'/'AWS_SECRET_ACCESS_KEY' vars
    std::env::set_var("S3_ACCESS_KEY_ID", &minio_access_key_id);
    std::env::set_var("S3_SECRET_ACCESS_KEY", &minio_secret_access_key);

    let provider = E2ETestProvider {
        supports_batch_inference: true,
        variant_name: "openai".to_string(),
        model_name: "openai::gpt-4o-mini-2024-07-18".into(),
        model_provider_name: "openai".into(),
        credentials: HashMap::new(),
    };

    let endpoint = std::env::var("TENSORZERO_MINIO_URL")
        .unwrap_or_else(|_| "http://127.0.0.1:8000/".to_string());

    let test_bucket = "tensorzero-e2e-tests";

    let client: Arc<dyn ObjectStore> = Arc::new(
        AmazonS3Builder::new()
            .with_bucket_name(test_bucket)
            .with_access_key_id(minio_access_key_id)
            .with_secret_access_key(minio_secret_access_key)
            .with_endpoint(&endpoint)
            .with_allow_http(true)
            .build()
            .unwrap(),
    );

    let mut prefix = Alphanumeric.sample_string(&mut rand::rng(), 6);
    prefix += "-";

    test_image_inference_with_provider_s3_compatible(
        provider,
        &StorageKind::S3Compatible {
            bucket_name: Some(test_bucket.to_string()),
            region: None,
            prefix: prefix.clone(),
            endpoint: Some(endpoint.clone()),
            allow_http: Some(true),
        },
        &client,
        &format!(
            r#"
    [object_storage]
    type = "s3_compatible"
    endpoint = "{endpoint}"
    bucket_name = "{test_bucket}"
    prefix = "{prefix}"
    allow_http = true

    [functions.image_test]
    type = "chat"

    [functions.image_test.variants.openai]
    type = "chat_completion"
    model = "openai::gpt-4o-mini-2024-07-18"
    "#
        ),
        &prefix,
    )
    .await;
}

#[tokio::test]
pub async fn test_parallel_tool_use_default_true_inference_request() {
    use crate::providers::common::check_parallel_tool_use_inference_response;

    let episode_id = Uuid::now_v7();

    let provider = E2ETestProvider {
        supports_batch_inference: true,
        variant_name: "openai".to_string(),
        model_name: "gpt-4o-mini-2024-07-18".into(),
        model_provider_name: "openai".into(),
        credentials: HashMap::new(),
    };

    // We don't specify `parallel_tool_use` in the request, so it shouldn't get passed to OpenAI,
    // resulting in their default value (`true`)
    let payload = json!({
        "function_name": "weather_helper_parallel",
        "episode_id": episode_id,
        "input":{
            "system": {"assistant_name": "Dr. Mehta"},
            "messages": [
                {
                    "role": "user",
                    "content": "What is the weather like in Tokyo (in Celsius)? Use both the provided `get_temperature` and `get_humidity` tools. Do not say anything else, just call the two functions."
                }
            ]},
        "stream": false,
        "variant_name": provider.variant_name,
    });

    let response = Client::new()
        .post(get_gateway_endpoint("/inference"))
        .json(&payload)
        .send()
        .await
        .unwrap();

    // Check if the API response is fine
    assert_eq!(response.status(), StatusCode::OK);
    let response_json = response.json::<Value>().await.unwrap();

    println!("API response: {response_json:#?}");
    check_parallel_tool_use_inference_response(
        response_json,
        &provider,
        Some(episode_id),
        false,
        Value::Null,
    )
    .await;
}

#[tokio::test]
pub async fn test_shorthand_embedding() {
    let shorthand_model = "openai::text-embedding-3-small";
    let payload = json!({
        "input": "Hello, world!",
        "model": format!("tensorzero::embedding_model_name::{}", shorthand_model),
    });
    let response = Client::new()
        .post(get_gateway_endpoint("/openai/v1/embeddings"))
        .json(&payload)
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let response_json = response.json::<Value>().await.unwrap();
    println!("Shorthand API response: {response_json:?}");
    assert_eq!(response_json["object"].as_str().unwrap(), "list");
    assert_eq!(
        response_json["model"].as_str().unwrap(),
        format!("tensorzero::embedding_model_name::{shorthand_model}")
    );
    assert_eq!(response_json["data"].as_array().unwrap().len(), 1);
    assert_eq!(response_json["data"][0]["index"].as_u64().unwrap(), 0);
    assert_eq!(
        response_json["data"][0]["object"].as_str().unwrap(),
        "embedding"
    );
    assert!(!response_json["data"][0]["embedding"]
        .as_array()
        .unwrap()
        .is_empty());
    assert!(response_json["usage"]["prompt_tokens"].as_u64().unwrap() > 0);
    assert!(response_json["usage"]["total_tokens"].as_u64().unwrap() > 0);
}

#[tokio::test]
pub async fn test_embedding_extra_body() {
    let payload = json!({
        "input": "Hello, world!",
        "model": "tensorzero::embedding_model_name::voyage_3_5_lite_256",
    });
    let response = Client::new()
        .post(get_gateway_endpoint("/openai/v1/embeddings"))
        .json(&payload)
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let response_json = response.json::<Value>().await.unwrap();
    println!("API response: {response_json:?}");
    assert_eq!(response_json["object"].as_str().unwrap(), "list");
    // voyage-3.5-lite outputs 1024 dimensions by default, but we use extra_body to tell it to output 256.
    assert_eq!(
        response_json["data"][0]["embedding"]
            .as_array()
            .unwrap()
            .len(),
        256
    );
}

// Tests that starting a batch inference with file input writes the file to the object store
// We don't attempt to poll this batch inference, as we already have lots of tests that do that
// (and we never read things back from the object in batch inference handling)
#[tokio::test]
pub async fn test_start_batch_inference_write_file() {
    let temp_dir = tempfile::tempdir().unwrap();
    let config = format!(
        r#"
    [object_storage]
    type = "filesystem"
    path = "{}"

    [functions.batch_image]
    type = "chat"

    [functions.batch_image.variants.openai]
    type = "chat_completion"
    model = "openai::gpt-4o-mini-2024-07-18"
    "#,
        temp_dir.path().to_string_lossy()
    );

    let client = tensorzero::test_helpers::make_embedded_gateway_with_config(&config).await;

    let episode_id = Uuid::now_v7();

    let response = client
        .start_batch_inference(StartBatchInferenceParams {
            function_name: "batch_image".to_string(),
            variant_name: Some("openai".to_string()),
            episode_ids: Some(vec![Some(episode_id)]),
            inputs: vec![Input {
                system: None,
                messages: vec![InputMessage {
                    role: Role::User,
                    content: vec![InputMessageContent::Text(Text { text: "Tell me about this image".to_string() }),
                    InputMessageContent::File(File::Url(UrlFile {
                        url: "https://raw.githubusercontent.com/tensorzero/tensorzero/ff3e17bbd3e32f483b027cf81b54404788c90dc1/tensorzero-internal/tests/e2e/providers/ferris.png".parse().unwrap(),
                        mime_type: None,
                        detail: None,
                        filename: None,
                    }))],
                }],
            }],
            tags: Some(vec![Some([("foo".to_string(), "bar".to_string()), ("test_type".to_string(), "batch_image_object_store".to_string())].into_iter().collect() )]),
            ..Default::default()
        })
        .await
        .unwrap();

    let batch_id = response.batch_id;
    let inference_ids = response.inference_ids;
    assert_eq!(inference_ids.len(), 1);

    let inference_id = inference_ids[0];
    let episode_ids = response.episode_ids;
    assert_eq!(episode_ids.len(), 1);
    let returned_episode_id = episode_ids[0];
    assert_eq!(returned_episode_id, episode_id);

    // Sleep to allow time for data to be inserted into ClickHouse (trailing writes from API)
    tokio::time::sleep(std::time::Duration::from_millis(500)).await;

    // Check if ClickHouse is ok - BatchModelInference Table
    let clickhouse = get_clickhouse().await;
    let result = select_batch_model_inference_clickhouse(&clickhouse, inference_id)
        .await
        .unwrap();

    println!("ClickHouse - BatchModelInference: {result:#?}");

    let id = result.get("inference_id").unwrap().as_str().unwrap();
    let id = Uuid::parse_str(id).unwrap();
    assert_eq!(id, inference_id);

    let retrieved_batch_id = result.get("batch_id").unwrap().as_str().unwrap();
    let retrieved_batch_id = Uuid::parse_str(retrieved_batch_id).unwrap();
    assert_eq!(retrieved_batch_id, batch_id);

    let function_name = result.get("function_name").unwrap().as_str().unwrap();
    assert_eq!(function_name, "batch_image");

    let variant_name = result.get("variant_name").unwrap().as_str().unwrap();
    assert_eq!(variant_name, "openai");

    let retrieved_episode_id = result.get("episode_id").unwrap().as_str().unwrap();
    let retrieved_episode_id = Uuid::parse_str(retrieved_episode_id).unwrap();
    assert_eq!(retrieved_episode_id, episode_id);

    let input: Value =
        serde_json::from_str(result.get("input").unwrap().as_str().unwrap()).unwrap();

    let file_path =
        "observability/files/08bfa764c6dc25e658bab2b8039ddb494546c3bc5523296804efc4cab604df5d.png";
    let correct_input = json!({
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Tell me about this image"},
                    {
                        "type": "file",
                        "source_url": "https://raw.githubusercontent.com/tensorzero/tensorzero/ff3e17bbd3e32f483b027cf81b54404788c90dc1/tensorzero-internal/tests/e2e/providers/ferris.png",
                        "mime_type": "image/png",
                        "storage_path": {
                            "kind": {"type": "filesystem", "path": temp_dir.path().to_string_lossy()},
                            "path": file_path
                        }
                    }
                ]
            }
        ]
    });
    assert_eq!(input, correct_input);

    // Check that the file exists on the filesystem
    let result = std::fs::read(temp_dir.path().join(file_path)).unwrap();
    assert_eq!(result, FERRIS_PNG);
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
        model_name: Some("openai::gpt-4o-mini".to_string()),
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
    assert_eq!(raw_request, "{\"messages\":[{\"role\":\"user\",\"content\":[{\"type\":\"text\",\"text\":\"Describe the contents of the image\"},{\"type\":\"image_url\",\"image_url\":{\"url\":\"https://raw.githubusercontent.com/tensorzero/tensorzero/ff3e17bbd3e32f483b027cf81b54404788c90dc1/tensorzero-internal/tests/e2e/providers/ferris.png\"}}]}],\"model\":\"gpt-4o-mini\",\"stream\":false}");

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
        model_name: Some("openai::gpt-4o-mini".to_string()),
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
    // OpenAI currently doesn't support forwarding file urls, so we should base64 encode the file data
    assert_eq!(raw_request, "{\"messages\":[{\"role\":\"user\",\"content\":[{\"type\":\"text\",\"text\":\"Describe the contents of the PDF\"},{\"type\":\"file\",\"file\":{\"file_data\":\"data:application/pdf;base64,<TENSORZERO_FILE_0>\",\"filename\":\"input.pdf\"}}]}],\"model\":\"gpt-4o-mini\",\"stream\":false}");

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

#[tokio::test]
async fn test_responses_api_reasoning() {
    let payload = json!({
        "function_name": "openai_responses_gpt5",
        "variant_name": "openai",
        "input":
            {
               "messages": [
                {
                    "role": "user",
                    "content": "How many letters are in the word potato?"
                }
            ]},
        "extra_body": [
            {
                "variant_name": "openai",
                "pointer": "/reasoning",
                "value": {
                    "effort": "low",
                    "summary": "auto"
                }
            }
        ]
    });

    let response = Client::new()
        .post(get_gateway_endpoint("/inference"))
        .json(&payload)
        .send()
        .await
        .unwrap();

    let response_json = response.json::<Value>().await.unwrap();
    println!("API response: {response_json}");

    let content_blocks = response_json.get("content").unwrap().as_array().unwrap();
    let has_thought = content_blocks
        .iter()
        .any(|block| block.get("type").unwrap().as_str().unwrap() == "thought");
    assert!(
        has_thought,
        "Missing thought block in output: {content_blocks:?}"
    );

    let encrypted_thought = content_blocks.iter().find(|block| {
        block.get("type").unwrap().as_str().unwrap() == "thought"
            && block.get("signature").unwrap().as_str().is_some()
    });
    assert!(
        encrypted_thought.is_some(),
        "Missing encrypted thought block in output: {content_blocks:?}"
    );
    let encrypted_thought = encrypted_thought.unwrap();

    assert_eq!(
        encrypted_thought.get("text").unwrap(),
        &Value::Null,
        "Text should be null in encrypted thought: {encrypted_thought:?}"
    );
    let summary = encrypted_thought
        .get("summary")
        .unwrap()
        .as_array()
        .unwrap();
    assert!(
        !summary.is_empty(),
        "Missing summary in encrypted thought: {encrypted_thought:?}"
    );
    for item in summary {
        assert_eq!(item.get("type").unwrap().as_str().unwrap(), "summary_text");
        let summary_text = item.get("text").unwrap().as_str().unwrap();
        assert!(
            !summary_text.is_empty(),
            "Missing summary text in item: {item:?}"
        );
    }

    let payload = json!({
        "function_name": "openai_responses_gpt5",
        "variant_name": "openai",
        "input":
            {
               "messages": [
                {
                    "role": "user",
                    "content": "How many letters are in the word potato?"
                },
                {
                    "role": "assistant",
                    "content": content_blocks,
                },
                {
                    "role": "user",
                    "content": "What were you thinking about during your last response?"
                }
            ]},
        "extra_body": [
            {
                "variant_name": "openai",
                "pointer": "/reasoning",
                "value": {
                    "effort": "low",
                    "summary": "auto"
                }
            }
        ]
    });

    let response = Client::new()
        .post(get_gateway_endpoint("/inference"))
        .json(&payload)
        .send()
        .await
        .unwrap();

    let status = response.status();
    let response_json = response.json::<Value>().await.unwrap();
    println!("New API response: {response_json}");
    assert_eq!(status, StatusCode::OK);
}

#[tokio::test]
async fn test_responses_api_invalid_thought() {
    let payload = json!({
        "function_name": "openai_responses_gpt5",
        "variant_name": "openai",
        "input":
            {
               "messages": [
                {
                    "role": "user",
                    "content": "How many letters are in the word potato?"
                },
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "thought",
                            "signature": "My fake signature",
                            "summary": [
                                {
                                    "type": "summary_text",
                                    "text": "I thought about responding to the next message with the word 'potato'"
                                }
                            ]
                        },
                    ]
                },
                {
                    "role": "user",
                    "content": "What time is it?"
                }
            ]},
        "extra_body": [
            {
                "variant_name": "openai",
                "pointer": "/reasoning",
                "value": {
                    "effort": "low",
                    "summary": "auto"
                }
            }
        ]
    });

    let response = Client::new()
        .post(get_gateway_endpoint("/inference"))
        .json(&payload)
        .send()
        .await
        .unwrap();

    let status = response.status();
    let response_text = response.text().await.unwrap();
    println!("API response: {response_text}");
    assert!(response_text.contains("could not be verified"));
    assert_eq!(status, StatusCode::INTERNAL_SERVER_ERROR);
}

const WEB_SEARCH_PROMPT: &str = "Tell me some good news that happened today from around the world. Don't ask me any questions, and provide markdown citations in the form [text](url)";

#[tokio::test(flavor = "multi_thread")]
pub async fn test_openai_built_in_websearch() {
    // Create a config with the custom credential location
    let config = r#"

gateway.debug = true
[models."test-model"]
routing = ["test-provider"]

[models."test-model".providers.test-provider]
type = "openai"
model_name = "gpt-5-nano"
provider_tools = [{type = "web_search"}]
api_type = "responses"

[functions.basic_test]
type = "chat"

[functions.basic_test.variants.default]
type = "chat_completion"
model = "test-model"
"#;

    // Create an embedded gateway with this config
    let client = tensorzero::test_helpers::make_embedded_gateway_with_config(config).await;

    // Make a simple inference request to verify it works
    let episode_id = Uuid::now_v7();
    let result = client
        .inference(ClientInferenceParams {
            function_name: Some("basic_test".to_string()),
            variant_name: Some("default".to_string()),
            episode_id: Some(episode_id),
            input: ClientInput {
                system: None,
                messages: vec![ClientInputMessage {
                    role: Role::User,
                    content: vec![ClientInputMessageContent::Text(TextKind::Text {
                        text: WEB_SEARCH_PROMPT.to_string(),
                    })],
                }],
            },
            stream: Some(false),
            ..Default::default()
        })
        .await;

    // Assert the inference succeeded
    let response = result.unwrap();
    println!("response: {response:?}");

    // Extract the chat response
    let InferenceOutput::NonStreaming(response) = response else {
        panic!("Expected non-streaming inference response");
    };

    let InferenceResponse::Chat(chat_response) = response else {
        panic!("Expected chat inference response");
    };

    // Assert that we have at least one Unknown content block with type "web_search_call"
    let web_search_blocks: Vec<_> = chat_response
        .content
        .iter()
        .filter(|block| {
            if let ContentBlockChatOutput::Unknown(unknown) = block {
                unknown
                    .data
                    .get("type")
                    .and_then(|t| t.as_str())
                    .map(|t| t == "web_search_call")
                    .unwrap_or(false)
            } else {
                false
            }
        })
        .collect();

    assert!(
        !web_search_blocks.is_empty(),
        "Expected at least one Unknown content block with type 'web_search_call', but found none. Content blocks: {:#?}",
        chat_response.content
    );

    // Assert that we have exactly one Text content block
    let text_blocks: Vec<_> = chat_response
        .content
        .iter()
        .filter_map(|block| {
            if let ContentBlockChatOutput::Text(text) = block {
                Some(text)
            } else {
                None
            }
        })
        .collect();

    assert_eq!(
        text_blocks.len(),
        1,
        "Expected exactly one Text content block, but found {}. Content blocks: {:#?}",
        text_blocks.len(),
        chat_response.content
    );

    // Assert that the text block contains citations (markdown links)
    let text_content = &text_blocks[0].text;
    assert!(
        text_content.contains("]("),
        "Expected text content to contain citations in markdown format [text](url), but found none. Text: {text_content}",
    );

    // Round-trip test: Convert output content blocks back to input and make another inference
    let assistant_content: Vec<ClientInputMessageContent> = chat_response
        .content
        .iter()
        .map(|block| match block {
            ContentBlockChatOutput::Text(text) => ClientInputMessageContent::Text(TextKind::Text {
                text: text.text.clone(),
            }),
            ContentBlockChatOutput::ToolCall(tool_call) => ClientInputMessageContent::ToolCall(
                ToolCallWrapper::InferenceResponseToolCall(tool_call.clone()),
            ),
            ContentBlockChatOutput::Thought(thought) => {
                ClientInputMessageContent::Thought(thought.clone())
            }
            ContentBlockChatOutput::Unknown(unknown) => {
                ClientInputMessageContent::Unknown(unknown.clone())
            }
        })
        .collect();

    // Make a second inference with the assistant's response and a new user question
    let result2 = client
        .inference(ClientInferenceParams {
            function_name: Some("basic_test".to_string()),
            variant_name: Some("default".to_string()),
            episode_id: Some(episode_id),
            input: ClientInput {
                system: None,
                messages: vec![
                    ClientInputMessage {
                        role: Role::User,
                        content: vec![ClientInputMessageContent::Text(TextKind::Text {
                            text: WEB_SEARCH_PROMPT.to_string(),
                        })],
                    },
                    ClientInputMessage {
                        role: Role::Assistant,
                        content: assistant_content,
                    },
                    ClientInputMessage {
                        role: Role::User,
                        content: vec![ClientInputMessageContent::Text(TextKind::Text {
                            text: "Can you summarize what you just told me in one sentence?"
                                .to_string(),
                        })],
                    },
                ],
            },
            stream: Some(false),
            ..Default::default()
        })
        .await;

    // Assert the round-trip inference succeeded
    let response2 = result2.unwrap();
    println!("Round-trip response: {response2:?}");

    let InferenceOutput::NonStreaming(response2) = response2 else {
        panic!("Expected non-streaming inference response for round-trip");
    };

    let InferenceResponse::Chat(chat_response2) = response2 else {
        panic!("Expected chat inference response for round-trip");
    };

    // Assert that the second response has at least one text block
    let has_text = chat_response2
        .content
        .iter()
        .any(|block| matches!(block, ContentBlockChatOutput::Text(_)));
    assert!(
        has_text,
        "Expected at least one text content block in round-trip response. Content blocks: {:#?}",
        chat_response2.content
    );
}

#[tokio::test(flavor = "multi_thread")]
pub async fn test_openai_built_in_websearch_streaming() {
    // Create a config with the custom credential location
    let config = r#"

gateway.debug = true
[models."test-model"]
routing = ["test-provider"]

[models."test-model".providers.test-provider]
type = "openai"
model_name = "gpt-5-nano"
provider_tools = [{type = "web_search"}]
api_type = "responses"

[functions.basic_test]
type = "chat"

[functions.basic_test.variants.default]
type = "chat_completion"
model = "test-model"
"#;

    // Create an embedded gateway with this config
    let client = tensorzero::test_helpers::make_embedded_gateway_with_config(config).await;

    // Make a streaming inference request
    let episode_id = Uuid::now_v7();
    let result = client
        .inference(ClientInferenceParams {
            function_name: Some("basic_test".to_string()),
            variant_name: Some("default".to_string()),
            episode_id: Some(episode_id),
            input: ClientInput {
                system: None,
                messages: vec![ClientInputMessage {
                    role: Role::User,
                    content: vec![ClientInputMessageContent::Text(TextKind::Text {
                        text: WEB_SEARCH_PROMPT.to_string(),
                    })],
                }],
            },
            stream: Some(true),
            ..Default::default()
        })
        .await;

    // Assert the inference succeeded
    let response = result.unwrap();
    println!("response: {response:?}");

    // Extract the streaming response
    let InferenceOutput::Streaming(mut stream) = response else {
        panic!("Expected streaming inference response");
    };

    // Collect all chunks
    let mut chunks = vec![];
    let mut inference_id: Option<Uuid> = None;
    let mut full_text = String::new();
    let mut unknown_chunks = vec![];

    while let Some(chunk_result) = stream.next().await {
        let chunk = chunk_result.unwrap();

        // Extract inference_id from the first chunk
        if inference_id.is_none() {
            if let InferenceResponseChunk::Chat(chat_chunk) = &chunk {
                inference_id = Some(chat_chunk.inference_id);
            }
        }

        // Collect text and unknown chunks
        if let InferenceResponseChunk::Chat(chat_chunk) = &chunk {
            for content_block in &chat_chunk.content {
                match content_block {
                    ContentBlockChunk::Text(text_chunk) => {
                        full_text.push_str(&text_chunk.text);
                    }
                    ContentBlockChunk::Unknown(UnknownChunk { id, data, .. }) => {
                        unknown_chunks.push((id.clone(), data.clone()));
                    }
                    _ => {}
                }
            }
        }

        chunks.push(chunk);
    }

    // Assert that we have multiple streaming chunks (indicates streaming is working)
    assert!(
        chunks.len() >= 3,
        "Expected at least 3 streaming chunks, but got {}. Streaming may not be working properly.",
        chunks.len()
    );

    // Assert that all chunks are Chat type
    for chunk in &chunks {
        assert!(
            matches!(chunk, InferenceResponseChunk::Chat(_)),
            "Expected all chunks to be Chat type, but found: {chunk:?}",
        );
    }

    // Assert that the last chunk has usage information
    if let Some(InferenceResponseChunk::Chat(last_chunk)) = chunks.last() {
        assert!(
            last_chunk.usage.is_some(),
            "Expected the last chunk to have usage information, but it was None"
        );
    } else {
        panic!("No chunks received");
    }

    // Assert that the last chunk has a finish_reason
    if let Some(InferenceResponseChunk::Chat(last_chunk)) = chunks.last() {
        assert!(
            last_chunk.finish_reason.is_some(),
            "Expected the last chunk to have a finish_reason, but it was None"
        );
    }

    // Assert that we received Unknown chunks for web_search_call
    assert!(
        !unknown_chunks.is_empty(),
        "Expected at least one Unknown chunk during streaming, but found none"
    );

    // Verify that at least one Unknown chunk contains web_search_call type
    let has_web_search_chunk = unknown_chunks.iter().any(|(_, data)| {
        data.get("type")
            .and_then(|t| t.as_str())
            .map(|t| t == "web_search_call")
            .unwrap_or(false)
    });
    assert!(
        has_web_search_chunk,
        "Expected at least one Unknown chunk with type 'web_search_call', but found none. Unknown chunks: {unknown_chunks:#?}",
    );

    // Assert that the concatenated text contains citations (markdown links)
    assert!(
        full_text.contains("]("),
        "Expected concatenated text to contain citations in markdown format [text](url), but found none. Text length: {}",
        full_text.len()
    );

    // Sleep for 1 second to allow writing to ClickHouse
    tokio::time::sleep(std::time::Duration::from_secs(1)).await;
    let clickhouse = get_clickhouse().await;

    let inference_id = inference_id.expect("Should have extracted inference_id from chunks");

    // Fetch the model inference data from ClickHouse
    let model_inference = select_model_inference_clickhouse(&clickhouse, inference_id)
        .await
        .unwrap();

    let raw_response = model_inference
        .get("raw_response")
        .unwrap()
        .as_str()
        .unwrap();

    // Assert that raw_response contains web_search_call (confirms web search was used)
    assert!(
        raw_response.contains("web_search_call"),
        "Expected raw_response to contain 'web_search_call', but it was not found"
    );

    // Assert that raw_response contains response.completed event
    assert!(
        raw_response.contains("response.completed"),
        "Expected raw_response to contain 'response.completed' event, but it was not found"
    );
}

#[tokio::test(flavor = "multi_thread")]
pub async fn test_openai_built_in_websearch_dynamic() {
    // Create a config WITHOUT provider_tools in the model config
    // We'll pass the provider_tools dynamically at inference time
    let config = r#"

gateway.debug = true
[models."test-model"]
routing = ["test-provider"]

[models."test-model".providers.test-provider]
type = "openai"
model_name = "gpt-5-nano"
api_type = "responses"

[functions.basic_test]
type = "chat"

[functions.basic_test.variants.default]
type = "chat_completion"
model = "test-model"
"#;

    // Create an embedded gateway with this config
    let client = tensorzero::test_helpers::make_embedded_gateway_with_config(config).await;

    // Make a simple inference request with dynamic provider_tools
    let episode_id = Uuid::now_v7();
    let result = client
        .inference(ClientInferenceParams {
            function_name: Some("basic_test".to_string()),
            variant_name: Some("default".to_string()),
            episode_id: Some(episode_id),
            input: ClientInput {
                system: None,
                messages: vec![ClientInputMessage {
                    role: Role::User,
                    content: vec![ClientInputMessageContent::Text(TextKind::Text {
                        text: WEB_SEARCH_PROMPT.to_string(),
                    })],
                }],
            },
            stream: Some(false),
            dynamic_tool_params: tensorzero_core::tool::DynamicToolParams {
                allowed_tools: None,
                additional_tools: None,
                tool_choice: None,
                parallel_tool_calls: None,
                provider_tools: vec![
                    ProviderTool {
                        scope: ProviderToolScope::Unscoped,
                        tool: json!({"type": "web_search"}),
                    },
                    // This should get filtered out
                    ProviderTool {
                        scope: ProviderToolScope::ModelProvider(ProviderToolScopeModelProvider {
                            model_name: "garbage".to_string(),
                            provider_name: Some("model".to_string()),
                        }),
                        tool: json!({"type": "garbage"}),
                    },
                ],
            },
            ..Default::default()
        })
        .await;

    // Assert the inference succeeded
    let response = result.unwrap();
    println!("response: {response:?}");

    // Extract the chat response
    let InferenceOutput::NonStreaming(response) = response else {
        panic!("Expected non-streaming inference response");
    };

    let InferenceResponse::Chat(chat_response) = response else {
        panic!("Expected chat inference response");
    };

    // Assert that we have at least one Unknown content block with type "web_search_call"
    let web_search_blocks: Vec<_> = chat_response
        .content
        .iter()
        .filter(|block| {
            if let ContentBlockChatOutput::Unknown(unknown) = block {
                unknown
                    .data
                    .get("type")
                    .and_then(|t| t.as_str())
                    .map(|t| t == "web_search_call")
                    .unwrap_or(false)
            } else {
                false
            }
        })
        .collect();

    assert!(
        !web_search_blocks.is_empty(),
        "Expected at least one Unknown content block with type 'web_search_call', but found none. Content blocks: {:#?}",
        chat_response.content
    );

    // Assert that we have exactly one Text content block
    let text_blocks: Vec<_> = chat_response
        .content
        .iter()
        .filter_map(|block| {
            if let ContentBlockChatOutput::Text(text) = block {
                Some(text)
            } else {
                None
            }
        })
        .collect();

    assert_eq!(
        text_blocks.len(),
        1,
        "Expected exactly one Text content block, but found {}. Content blocks: {:#?}",
        text_blocks.len(),
        chat_response.content
    );

    // Assert that the text block contains citations (markdown links)
    let text_content = &text_blocks[0].text;
    assert!(
        text_content.contains("]("),
        "Expected text content to contain citations in markdown format [text](url), but found none. Text: {text_content}",
    );

    // Round-trip test: Convert output content blocks back to input and make another inference
    let assistant_content: Vec<ClientInputMessageContent> = chat_response
        .content
        .iter()
        .map(|block| match block {
            ContentBlockChatOutput::Text(text) => ClientInputMessageContent::Text(TextKind::Text {
                text: text.text.clone(),
            }),
            ContentBlockChatOutput::ToolCall(tool_call) => ClientInputMessageContent::ToolCall(
                ToolCallWrapper::InferenceResponseToolCall(tool_call.clone()),
            ),
            ContentBlockChatOutput::Thought(thought) => {
                ClientInputMessageContent::Thought(thought.clone())
            }
            ContentBlockChatOutput::Unknown(unknown) => {
                ClientInputMessageContent::Unknown(unknown.clone())
            }
        })
        .collect();

    // Make a second inference with the assistant's response and a new user question
    let result2 = client
        .inference(ClientInferenceParams {
            function_name: Some("basic_test".to_string()),
            variant_name: Some("default".to_string()),
            episode_id: Some(episode_id),
            input: ClientInput {
                system: None,
                messages: vec![
                    ClientInputMessage {
                        role: Role::User,
                        content: vec![ClientInputMessageContent::Text(TextKind::Text {
                            text: WEB_SEARCH_PROMPT.to_string(),
                        })],
                    },
                    ClientInputMessage {
                        role: Role::Assistant,
                        content: assistant_content,
                    },
                    ClientInputMessage {
                        role: Role::User,
                        content: vec![ClientInputMessageContent::Text(TextKind::Text {
                            text: "Can you summarize what you just told me in one sentence?"
                                .to_string(),
                        })],
                    },
                ],
            },
            stream: Some(false),
            dynamic_tool_params: tensorzero_core::tool::DynamicToolParams {
                allowed_tools: None,
                additional_tools: None,
                tool_choice: None,
                parallel_tool_calls: None,
                provider_tools: vec![ProviderTool {
                    scope: ProviderToolScope::Unscoped,
                    tool: json!({"type": "web_search"}),
                }],
            },
            ..Default::default()
        })
        .await;

    // Assert the round-trip inference succeeded
    let response2 = result2.unwrap();
    println!("Round-trip response: {response2:?}");

    let InferenceOutput::NonStreaming(response2) = response2 else {
        panic!("Expected non-streaming inference response for round-trip");
    };

    let InferenceResponse::Chat(chat_response2) = response2 else {
        panic!("Expected chat inference response for round-trip");
    };

    // Assert that the second response has at least one text block
    let has_text = chat_response2
        .content
        .iter()
        .any(|block| matches!(block, ContentBlockChatOutput::Text(_)));
    assert!(
        has_text,
        "Expected at least one text content block in round-trip response. Content blocks: {:#?}",
        chat_response2.content
    );
}

/// Tests using the shorthand form for the OpenAI Responses API.
/// This works because gpt-5-codex is only available via the responses API,
/// so the shorthand form "openai::responses::gpt-5-codex" correctly identifies
/// both the provider (openai) and the API type (responses).
#[tokio::test]
async fn test_responses_api_shorthand() {
    let client = Client::new();
    let episode_id = Uuid::now_v7();

    let payload = json!({
        "model_name": "openai::responses::gpt-5-codex",
        "episode_id": episode_id,
        "input": {
            "messages": [
                {
                    "role": "user",
                    "content": "What is the capital of France?"
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
    println!("response: {response_json:#}");
    assert!(!content_blocks.is_empty());

    // Find the text block (there should be exactly one)
    let text_blocks: Vec<&Value> = content_blocks
        .iter()
        .filter(|block| block.get("type").unwrap().as_str().unwrap() == "text")
        .collect();
    assert_eq!(text_blocks.len(), 1, "Should have exactly one text block");

    let text_block = text_blocks.first().unwrap();
    let content = text_block.get("text").unwrap().as_str().unwrap();
    // Assert that Paris is in the content
    assert!(content.contains("Paris"), "Content should mention Paris");
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
                "content": [{"type": "text", "text": "What is the capital of France?"}]
            }
        ]
    });
    assert_eq!(input, correct_input);
    let content_blocks = result.get("output").unwrap().as_str().unwrap();
    // Check that content_blocks is a list of blocks with at least 1 block
    let content_blocks: Vec<Value> = serde_json::from_str(content_blocks).unwrap();
    assert!(!content_blocks.is_empty());

    // Find the text block (there should be exactly one)
    let text_blocks: Vec<&Value> = content_blocks
        .iter()
        .filter(|block| block.get("type").unwrap().as_str().unwrap() == "text")
        .collect();
    assert_eq!(text_blocks.len(), 1, "Should have exactly one text block");

    let text_block = text_blocks.first().unwrap();
    let clickhouse_content = text_block.get("text").unwrap().as_str().unwrap();
    assert_eq!(clickhouse_content, content);
    // Check that episode_id is here and correct
    let retrieved_episode_id = result.get("episode_id").unwrap().as_str().unwrap();
    let retrieved_episode_id = Uuid::parse_str(retrieved_episode_id).unwrap();
    assert_eq!(retrieved_episode_id, episode_id);
    // Check the variant name
    let variant_name = result.get("variant_name").unwrap().as_str().unwrap();
    assert_eq!(variant_name, "openai::responses::gpt-5-codex");
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
    assert_eq!(model_name, "openai::responses::gpt-5-codex");
    let model_provider_name = result.get("model_provider_name").unwrap().as_str().unwrap();
    assert_eq!(model_provider_name, "openai");
    let raw_request = result.get("raw_request").unwrap().as_str().unwrap();
    assert!(raw_request.to_lowercase().contains("france"));
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
}

#[tokio::test]
async fn test_file_custom_filename_sent_to_openai() {
    // Test that custom filename is sent to OpenAI API
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

    let response = client
        .inference(ClientInferenceParams {
            model_name: Some("openai::gpt-4o-mini".to_string()),
            input: ClientInput {
                messages: vec![ClientInputMessage {
                    role: Role::User,
                    content: vec![
                        ClientInputMessageContent::Text(TextKind::Text {
                            text: "Describe the contents of the PDF".to_string(),
                        }),
                        ClientInputMessageContent::File(File::Url(UrlFile {
                            url: Url::parse("https://raw.githubusercontent.com/tensorzero/tensorzero/ac37477d56deaf6e0585a394eda68fd4f9390cab/tensorzero-core/tests/e2e/providers/deepseek_paper.pdf").unwrap(),
                            mime_type: Some(mime::APPLICATION_PDF),
                            detail: None,
                            filename: Some("custom.pdf".to_string()),
                        })),
                    ],
                }],
                ..Default::default()
            },
            ..Default::default()
        })
        .await
        .unwrap();

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

    // Verify that the custom filename is present in the raw request
    assert!(
        raw_request.contains("\"filename\":\"custom.pdf\""),
        "Expected custom filename 'custom.pdf' in raw_request, got: {raw_request}"
    );
}

#[tokio::test]
async fn test_file_fallback_filename_sent_to_openai() {
    // Test that fallback filename "input.pdf" is used when no custom filename provided
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

    let response = client
        .inference(ClientInferenceParams {
            model_name: Some("openai::gpt-4o-mini".to_string()),
            input: ClientInput {
                messages: vec![ClientInputMessage {
                    role: Role::User,
                    content: vec![
                        ClientInputMessageContent::Text(TextKind::Text {
                            text: "Describe the contents of the PDF".to_string(),
                        }),
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
        })
        .await
        .unwrap();

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

    // Verify that the fallback filename "input.pdf" is present in the raw request
    assert!(
        raw_request.contains("\"filename\":\"input.pdf\""),
        "Expected fallback filename 'input.pdf' in raw_request, got: {raw_request}"
    );
}
