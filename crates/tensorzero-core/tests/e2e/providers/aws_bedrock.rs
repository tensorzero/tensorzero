use googletest::prelude::*;
use googletest_matchers::matches_json_literal;
use reqwest::Client;
use reqwest::StatusCode;
use serde_json::{Value, json};
use std::collections::HashMap;
use uuid::Uuid;

use crate::common::get_gateway_endpoint;
use crate::providers::common::{E2ETestProvider, E2ETestProviders};

use tensorzero_core::db::{
    delegating_connection::DelegatingDatabaseConnection,
    inferences::{InferenceQueries, ListInferencesParams},
    model_inferences::ModelInferenceQueries,
    test_helpers::TestDatabaseHelpers,
};
use tensorzero_core::inference::types::{ContentBlockChatOutput, StoredModelInference};
use tensorzero_core::stored_inference::{StoredChatInferenceDatabase, StoredInferenceDatabase};
use tensorzero_core::test_helpers::get_e2e_config;

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

    let claude_thinking_provider = E2ETestProvider {
        supports_batch_inference: false,
        variant_name: "aws-bedrock-thinking".to_string(),
        model_name: "claude-sonnet-4-5-thinking-aws-bedrock".into(),
        model_provider_name: "aws_bedrock".into(),
        credentials: HashMap::new(),
    };

    let standard_providers = vec![E2ETestProvider {
        supports_batch_inference: false,
        variant_name: "aws-bedrock".to_string(),
        model_name: "claude-haiku-4-5-aws-bedrock".into(),
        model_provider_name: "aws_bedrock".into(),
        credentials: HashMap::new(),
    }];

    let mut simple_inference_providers = standard_providers.clone();
    simple_inference_providers.push(deepseek_r1_provider.clone());

    let extra_body_providers = vec![E2ETestProvider {
        supports_batch_inference: false,
        variant_name: "aws-bedrock-extra-body".to_string(),
        model_name: "claude-haiku-4-5-aws-bedrock".into(),
        model_provider_name: "aws_bedrock".into(),
        credentials: HashMap::new(),
    }];

    let bad_auth_extra_headers = vec![E2ETestProvider {
        supports_batch_inference: false,
        variant_name: "aws-bedrock-extra-headers".to_string(),
        model_name: "claude-haiku-4-5-aws-bedrock".into(),
        model_provider_name: "aws_bedrock".into(),
        credentials: HashMap::new(),
    }];

    let json_providers = vec![
        E2ETestProvider {
            supports_batch_inference: false,
            variant_name: "aws-bedrock".to_string(),
            model_name: "claude-haiku-4-5-aws-bedrock".into(),
            model_provider_name: "aws_bedrock".into(),
            credentials: HashMap::new(),
        },
        E2ETestProvider {
            supports_batch_inference: false,
            variant_name: "aws-bedrock-implicit".to_string(),
            model_name: "claude-haiku-4-5-aws-bedrock".into(),
            model_provider_name: "aws_bedrock".into(),
            credentials: HashMap::new(),
        },
        E2ETestProvider {
            supports_batch_inference: false,
            variant_name: "aws-bedrock-strict".to_string(),
            model_name: "claude-haiku-4-5-aws-bedrock".into(),
            model_provider_name: "aws_bedrock".into(),
            credentials: HashMap::new(),
        },
    ];

    let json_mode_off_providers = vec![E2ETestProvider {
        supports_batch_inference: false,
        variant_name: "aws_bedrock_json_mode_off".to_string(),
        model_name: "claude-haiku-4-5-aws-bedrock".into(),
        model_provider_name: "aws_bedrock".into(),
        credentials: HashMap::new(),
    }];

    // Cache providers - use Claude Haiku 4.5 which supports prompt caching on AWS Bedrock
    // (Claude 3 Haiku does not support caching on Bedrock)
    let cache_providers = vec![
        E2ETestProvider {
            supports_batch_inference: false,
            variant_name: "aws-bedrock".to_string(),
            model_name: "claude-haiku-4-5-aws-bedrock".into(),
            model_provider_name: "aws_bedrock".into(),
            credentials: HashMap::new(),
        },
        E2ETestProvider {
            supports_batch_inference: false,
            variant_name: "aws-bedrock".to_string(),
            model_name: "nova-lite-v1".into(),
            model_provider_name: "aws_bedrock".into(),
            credentials: HashMap::new(),
        },
    ];

    // Dynamic region provider - passes region at request time
    let inference_params_dynamic_providers = vec![E2ETestProvider {
        supports_batch_inference: false,
        variant_name: "aws-bedrock-dynamic-region".to_string(),
        model_name: "claude-haiku-4-5-aws-bedrock-dynamic-region".into(),
        model_provider_name: "aws-bedrock-dynamic-region".into(),
        credentials: HashMap::from([("aws_bedrock_region".to_string(), "us-east-1".to_string())]),
    }];

    E2ETestProviders {
        simple_inference: simple_inference_providers.clone(),
        extra_body_inference: extra_body_providers,
        bad_auth_extra_headers,
        // Bedrock JSON + reasoning tests are skipped because Bedrock uses prefill (which conflicts with reasoning)
        // Bedrock's Converse API doesn't support Anthropic's output_format parameter
        reasoning_inference: vec![claude_thinking_provider.clone()],
        reasoning_usage_inference: vec![claude_thinking_provider],
        cache_input_tokens_inference: cache_providers,
        embeddings: vec![],
        inference_params_inference: standard_providers.clone(),
        inference_params_dynamic_credentials: inference_params_dynamic_providers,
        provider_type_default_credentials: vec![],
        provider_type_default_credentials_shorthand: vec![],
        tool_use_inference: standard_providers.clone(),
        tool_multi_turn_inference: standard_providers.clone(),
        dynamic_tool_use_inference: standard_providers.clone(),
        parallel_tool_use_inference: standard_providers.clone(),
        json_mode_inference: json_providers.clone(),
        json_mode_off_inference: json_mode_off_providers.clone(),
        image_inference: standard_providers.clone(),
        pdf_inference: standard_providers.clone(),
        input_audio: vec![],
        shorthand_inference: vec![],
        // AWS Bedrock only works with SDK credentials
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

    let conn = DelegatingDatabaseConnection::new_for_e2e_test().await;
    conn.flush_pending_writes().await;
    conn.sleep_for_writes_to_be_visible().await;
    let config = get_e2e_config().await;

    // Check Inference table
    let inferences = conn
        .list_inferences(
            &config,
            &ListInferencesParams {
                ids: Some(&[inference_id]),
                ..Default::default()
            },
        )
        .await
        .unwrap();
    assert_that!(inferences, len(eq(1)));
    let chat_inf = match &inferences[0] {
        StoredInferenceDatabase::Chat(c) => c,
        StoredInferenceDatabase::Json(_) => panic!("Expected chat inference"),
    };

    expect_that!(
        chat_inf,
        matches_pattern!(StoredChatInferenceDatabase {
            inference_id: eq(&inference_id),
            episode_id: eq(&episode_id),
            variant_name: eq("aws-bedrock-us-east-1"),
            processing_time_ms: some(gt(&0)),
            ..
        })
    );
    expect_that!(
        serde_json::to_value(&chat_inf.input),
        ok(matches_json_literal!({
            "system": {"assistant_name": "AskJeeves"},
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": "Hello, world!"}]
                }
            ]
        }))
    );
    let output = chat_inf.output.as_ref().expect("output should be present");
    assert_that!(output, len(eq(1)));
    let output_text = match &output[0] {
        ContentBlockChatOutput::Text(t) => &t.text,
        _ => panic!("Expected text content block"),
    };
    assert_eq!(output_text, content);

    // Check the ModelInference Table
    let model_inferences = conn
        .get_model_inferences_by_inference_id(inference_id)
        .await
        .unwrap();
    assert_that!(model_inferences, len(eq(1)));
    let mi = &model_inferences[0];

    let raw_request_str = mi
        .raw_request
        .as_ref()
        .expect("raw_request should be present");
    assert!(raw_request_str.to_lowercase().contains("world!"));
    let _: Value = serde_json::from_str(raw_request_str).expect("raw_request should be valid JSON");

    let raw_response_str = mi
        .raw_response
        .as_ref()
        .expect("raw_response should be present");
    let raw_response_json: Value = serde_json::from_str(raw_response_str).unwrap();
    assert!(
        !raw_response_json["output"]["message"]["content"]
            .as_array()
            .unwrap()
            .is_empty(),
        "Unexpected raw response: {raw_response_json}"
    );

    expect_that!(
        mi,
        matches_pattern!(StoredModelInference {
            inference_id: eq(&inference_id),
            model_name: eq("claude-haiku-4-5-us-east-1"),
            model_provider_name: eq("aws-bedrock-us-east-1"),
            input_tokens: some(gt(&5)),
            output_tokens: some(gt(&5)),
            response_time_ms: some(gt(&0)),
            ttft_ms: none(),
            ..
        })
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

    let conn = DelegatingDatabaseConnection::new_for_e2e_test().await;
    conn.flush_pending_writes().await;
    conn.sleep_for_writes_to_be_visible().await;

    let model_inferences = conn
        .get_model_inferences_by_inference_id(inference_id)
        .await
        .unwrap();
    assert_that!(model_inferences, len(eq(1)));
    let model_inference = &model_inferences[0];

    let raw_request = model_inference
        .raw_request
        .as_deref()
        .expect("raw_request should be present");

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

/// Test that AWS Bedrock API key (bearer token) authentication works in isolation.
/// Removes all IAM credentials to ensure only bearer token auth is used.
#[tokio::test]
async fn test_aws_bedrock_auth_bearer_token_only() {
    use tensorzero::{
        ClientInferenceParams, Input, InputMessage, InputMessageContent, Role,
        test_helpers::make_embedded_gateway_with_config,
    };
    use tensorzero_core::inference::types::Text;

    // Require bearer token to be set
    let _api_key = std::env::var("AWS_BEARER_TOKEN_BEDROCK")
        .expect("AWS_BEARER_TOKEN_BEDROCK must be set to run this test");

    // Remove all IAM credentials to ensure only bearer token auth is used
    tensorzero_unsafe_helpers::remove_env_var_tests_only("AWS_ACCESS_KEY_ID");
    tensorzero_unsafe_helpers::remove_env_var_tests_only("AWS_SECRET_ACCESS_KEY");
    tensorzero_unsafe_helpers::remove_env_var_tests_only("AWS_SESSION_TOKEN");

    let config = r#"
[models.test-bedrock-bearer]
routing = ["aws_bedrock"]

[models.test-bedrock-bearer.providers.aws_bedrock]
type = "aws_bedrock"
model_id = "us.anthropic.claude-3-5-haiku-20241022-v1:0"
region = "us-east-1"
"#;

    let client = make_embedded_gateway_with_config(config).await;

    let result = client
        .inference(ClientInferenceParams {
            model_name: Some("test-bedrock-bearer".to_string()),
            input: Input {
                system: None,
                messages: vec![InputMessage {
                    role: Role::User,
                    content: vec![InputMessageContent::Text(Text {
                        text: "Say hello".into(),
                    })],
                }],
            },
            stream: Some(false),
            ..Default::default()
        })
        .await;

    assert!(
        result.is_ok(),
        "Bearer token authentication failed: {:?}",
        result.err()
    );
}

/// Test that AWS Bedrock IAM (SigV4) authentication works in isolation.
/// Removes bearer token to ensure only IAM auth is used.
#[tokio::test]
async fn test_aws_bedrock_auth_iam_credentials_only() {
    use tensorzero::{
        ClientInferenceParams, Input, InputMessage, InputMessageContent, Role,
        test_helpers::make_embedded_gateway_with_config,
    };
    use tensorzero_core::inference::types::Text;

    // Require IAM credentials to be set
    let _access_key_id =
        std::env::var("AWS_ACCESS_KEY_ID").expect("AWS_ACCESS_KEY_ID must be set to run this test");
    let _secret_access_key = std::env::var("AWS_SECRET_ACCESS_KEY")
        .expect("AWS_SECRET_ACCESS_KEY must be set to run this test");

    // Remove bearer token to ensure only IAM auth is used
    tensorzero_unsafe_helpers::remove_env_var_tests_only("AWS_BEARER_TOKEN_BEDROCK");

    let config = r#"
[models.test-bedrock-iam]
routing = ["aws_bedrock"]

[models.test-bedrock-iam.providers.aws_bedrock]
type = "aws_bedrock"
model_id = "us.anthropic.claude-3-5-haiku-20241022-v1:0"
region = "us-east-1"
"#;

    let client = make_embedded_gateway_with_config(config).await;

    let result = client
        .inference(ClientInferenceParams {
            model_name: Some("test-bedrock-iam".to_string()),
            input: Input {
                system: None,
                messages: vec![InputMessage {
                    role: Role::User,
                    content: vec![InputMessageContent::Text(Text {
                        text: "Say hello".into(),
                    })],
                }],
            },
            stream: Some(false),
            ..Default::default()
        })
        .await;

    assert!(
        result.is_ok(),
        "IAM (SigV4) authentication failed: {:?}",
        result.err()
    );
}
