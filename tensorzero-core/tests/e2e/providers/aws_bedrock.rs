use reqwest::Client;

use reqwest::StatusCode;

use serde_json::{json, Value};
use std::collections::HashMap;

use uuid::Uuid;

use crate::common::get_gateway_endpoint;
use crate::providers::common::{E2ETestProvider, E2ETestProviders};

use tensorzero_core::db::clickhouse::test_helpers::{
    get_clickhouse, select_chat_inference_clickhouse, select_model_inference_clickhouse,
};

crate::generate_provider_tests!(get_providers);
crate::generate_batch_inference_tests!(get_providers);

async fn get_providers() -> E2ETestProviders {
    let deepseek_r1_provider = E2ETestProvider {
        supports_batch_inference: false,
        variant_name: "aws-bedrock-deepseek-r1".to_string(),
        model_name: "deepseek-r1-aws-bedrock".into(),
        model_provider_name: "aws_bedrock".into(),
        credentials: HashMap::new(),
    };
    let standard_providers = vec![E2ETestProvider {
        supports_batch_inference: false,
        variant_name: "aws-bedrock".to_string(),
        model_name: "claude-3-haiku-20240307-aws-bedrock".into(),
        model_provider_name: "aws_bedrock".into(),
        credentials: HashMap::new(),
    }];

    let mut simple_inference_providers = standard_providers.clone();
    simple_inference_providers.push(deepseek_r1_provider.clone());

    let extra_body_providers = vec![E2ETestProvider {
        supports_batch_inference: false,
        variant_name: "aws-bedrock-extra-body".to_string(),
        model_name: "claude-3-haiku-20240307-aws-bedrock".into(),
        model_provider_name: "aws_bedrock".into(),
        credentials: HashMap::new(),
    }];

    let bad_auth_extra_headers = vec![E2ETestProvider {
        supports_batch_inference: false,
        variant_name: "aws-bedrock-extra-headers".to_string(),
        model_name: "claude-3-haiku-20240307-aws-bedrock".into(),
        model_provider_name: "aws_bedrock".into(),
        credentials: HashMap::new(),
    }];

    let json_providers = vec![
        E2ETestProvider {
            supports_batch_inference: false,
            variant_name: "aws-bedrock".to_string(),
            model_name: "claude-3-haiku-20240307-aws-bedrock".into(),
            model_provider_name: "aws_bedrock".into(),
            credentials: HashMap::new(),
        },
        E2ETestProvider {
            supports_batch_inference: false,
            variant_name: "aws-bedrock-implicit".to_string(),
            model_name: "claude-3-haiku-20240307-aws-bedrock".into(),
            model_provider_name: "aws_bedrock".into(),
            credentials: HashMap::new(),
        },
        E2ETestProvider {
            supports_batch_inference: false,
            variant_name: "aws-bedrock-strict".to_string(),
            model_name: "claude-3-haiku-20240307-aws-bedrock".into(),
            model_provider_name: "aws_bedrock".into(),
            credentials: HashMap::new(),
        },
    ];

    let json_mode_off_providers = vec![E2ETestProvider {
        supports_batch_inference: false,
        variant_name: "aws_bedrock_json_mode_off".to_string(),
        model_name: "claude-3-haiku-20240307-aws-bedrock".into(),
        model_provider_name: "aws_bedrock".into(),
        credentials: HashMap::new(),
    }];

    E2ETestProviders {
        simple_inference: simple_inference_providers,
        extra_body_inference: extra_body_providers,
        bad_auth_extra_headers,
        reasoning_inference: vec![],
        embeddings: vec![],
        inference_params_inference: standard_providers.clone(),
        inference_params_dynamic_credentials: vec![],
        provider_type_default_credentials: vec![],
        provider_type_default_credentials_shorthand: vec![],
        tool_use_inference: standard_providers.clone(),
        tool_multi_turn_inference: standard_providers.clone(),
        dynamic_tool_use_inference: standard_providers.clone(),
        parallel_tool_use_inference: vec![],
        json_mode_inference: json_providers.clone(),
        json_mode_off_inference: json_mode_off_providers.clone(),
        image_inference: standard_providers.clone(),
        pdf_inference: standard_providers.clone(),
        input_audio: vec![],

        shorthand_inference: vec![],
        // AWS bedrock only works with SDK credentials
        credential_fallbacks: vec![],
    }
}

#[tokio::test]
async fn test_inference_with_explicit_region() {
    let client = Client::new();
    let episode_id = Uuid::now_v7();

    let payload = json!({
        "function_name": "basic_test",
        "variant_name": "aws-bedrock-us-east-1",
        "episode_id": episode_id,
        "input":
            {"system": {"assistant_name": "AskJeeves"},
            "messages": [
                {
                    "role": "user",
                    "content": "Hello, world!"
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
                "content": [{"type": "text", "text": "Hello, world!"}]
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
    assert_eq!(variant_name, "aws-bedrock-us-east-1");
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
    assert_eq!(model_name, "claude-3-haiku-20240307-us-east-1");
    let model_provider_name = result.get("model_provider_name").unwrap().as_str().unwrap();
    assert_eq!(model_provider_name, "aws-bedrock-us-east-1");
    let raw_request = result.get("raw_request").unwrap().as_str().unwrap();
    assert!(raw_request.to_lowercase().contains("world!"));
    // Check that raw_request is valid JSON
    let _: Value = serde_json::from_str(raw_request).expect("raw_request should be valid JSON");
    let input_tokens = result.get("input_tokens").unwrap().as_u64().unwrap();
    assert!(input_tokens > 5);
    let output_tokens = result.get("output_tokens").unwrap().as_u64().unwrap();
    assert!(output_tokens > 5);
    let response_time_ms = result.get("response_time_ms").unwrap().as_u64().unwrap();
    assert!(response_time_ms > 0);
    assert!(result.get("ttft_ms").unwrap().is_null());
    let raw_response = result.get("raw_response").unwrap();
    let raw_response_json: Value = serde_json::from_str(raw_response.as_str().unwrap()).unwrap();
    assert!(
        !raw_response_json["output"]["message"]["content"]
            .as_array()
            .unwrap()
            .is_empty(),
        "Unexpected raw response: {raw_response_json}"
    );
}

#[tokio::test]
async fn test_inference_with_explicit_broken_region() {
    let client = Client::new();
    let episode_id = Uuid::now_v7();

    let payload = json!({
        "function_name": "basic_test",
        "variant_name": "aws-bedrock-uk-hogwarts-1",
        "episode_id": episode_id,
        "input":
            {"system": {"assistant_name": "Dumbledore"},
            "messages": [
                {
                    "role": "user",
                    "content": "Hello, world!"
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

    assert_eq!(response.status(), StatusCode::INTERNAL_SERVER_ERROR);

    let response_json = response.json::<Value>().await.unwrap();

    response_json.get("error").unwrap();
}

#[tokio::test]
async fn test_inference_with_empty_system() {
    let client = Client::new();
    let episode_id = Uuid::now_v7();

    let payload = json!({
        "function_name": "write_haiku",
        "variant_name": "aws_bedrock",
        "episode_id": episode_id,
        "input":
            {"system": "",
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "template", "name": "user", "arguments": {"topic": "artificial intelligence"}}]
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
    content_block.get("text").unwrap().as_str().unwrap();
}

#[tokio::test]
async fn test_inference_with_thinking_budget_tokens() {
    let client = Client::new();
    let episode_id = Uuid::now_v7();

    let payload = json!({
        "function_name": "basic_test",
        "variant_name": "aws-bedrock-thinking",
        "episode_id": episode_id,
        "input": {
            "system": {"assistant_name": "Bedrock Thinker"},
            "messages": [
                {
                    "role": "user",
                    "content": "Share a short fun fact."
                }
            ]
        },
        "stream": false,
        "params": {
            "chat_completion": {
                "thinking_budget_tokens": 1024
            }
        }
    });

    let response = client
        .post(get_gateway_endpoint("/inference"))
        .json(&payload)
        .send()
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    let response_json = response.json::<Value>().await.unwrap();
    let content_blocks = response_json
        .get("content")
        .and_then(Value::as_array)
        .expect("content must be an array");
    assert!(
        content_blocks.iter().any(|block| {
            block
                .get("type")
                .and_then(Value::as_str)
                .is_some_and(|t| t == "thought")
        }),
        "response should include at least one thought block: {response_json:#?}"
    );
    assert!(
        content_blocks.iter().any(|block| {
            block
                .get("type")
                .and_then(Value::as_str)
                .is_some_and(|t| t == "text")
        }),
        "response should include at least one text block: {response_json:#?}"
    );

    let inference_id = response_json
        .get("inference_id")
        .and_then(Value::as_str)
        .unwrap();
    let inference_id = Uuid::parse_str(inference_id).unwrap();

    tokio::time::sleep(std::time::Duration::from_secs(1)).await;

    let clickhouse = get_clickhouse().await;
    let model_inference = select_model_inference_clickhouse(&clickhouse, inference_id)
        .await
        .unwrap();

    let raw_request = model_inference
        .get("raw_request")
        .and_then(Value::as_str)
        .unwrap();

    let raw_request_json: Value =
        serde_json::from_str(raw_request).expect("raw_request should be valid JSON");

    let thinking = raw_request_json
        .get("additionalModelRequestFields")
        .and_then(|fields| fields.get("thinking"))
        .expect("Expected `thinking` block to be forwarded to AWS Bedrock");

    let thinking_type = thinking
        .get("type")
        .and_then(Value::as_str)
        .expect("Expected thinking type");
    assert_eq!(thinking_type, "enabled");

    let budget_tokens = thinking
        .get("budget_tokens")
        .and_then(Value::as_i64)
        .or_else(|| thinking.get("budgetTokens").and_then(Value::as_i64))
        .expect("Expected thinking budget tokens");
    assert_eq!(budget_tokens, 1024);
}
