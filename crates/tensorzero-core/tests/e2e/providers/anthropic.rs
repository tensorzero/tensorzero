#![expect(clippy::print_stdout)]
use std::collections::HashMap;

use futures::StreamExt;
use googletest::prelude::*;
use googletest_matchers::matches_json_literal;
use indexmap::IndexMap;
use reqwest::{Client, StatusCode};
use reqwest_sse_stream::{Event, RequestBuilderExt};
use serde_json::{Value, json};
use tensorzero::{
    ClientInferenceParams, File, InferenceOutput, InferenceResponse, Input, InputMessage,
    InputMessageContent, Role, UrlFile, test_helpers::make_embedded_gateway_with_config,
};
use url::Url;
use uuid::Uuid;

use crate::{
    common::get_gateway_endpoint,
    providers::common::{DEEPSEEK_PAPER_PDF, E2ETestProvider, E2ETestProviders, FERRIS_PNG},
};
use tensorzero_core::db::{
    delegating_connection::DelegatingDatabaseConnection,
    inferences::{InferenceQueries, ListInferencesParams},
    model_inferences::ModelInferenceQueries,
    test_helpers::TestDatabaseHelpers,
};
use tensorzero_core::inference::types::{ContentBlockChatOutput, StoredModelInference, Text};
use tensorzero_core::stored_inference::{StoredChatInferenceDatabase, StoredInferenceDatabase};
use tensorzero_core::test_helpers::get_e2e_config;

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
        model_name: "claude-haiku-4-5-anthropic".into(),
        model_provider_name: "anthropic".into(),
        credentials: HashMap::new(),
    }];

    let image_providers = vec![E2ETestProvider {
        supports_batch_inference: false,
        variant_name: "anthropic".to_string(),
        model_name: "anthropic::claude-haiku-4-5".into(),
        model_provider_name: "anthropic".into(),
        credentials: HashMap::new(),
    }];

    let pdf_providers = vec![E2ETestProvider {
        supports_batch_inference: false,
        variant_name: "anthropic".to_string(),
        model_name: "anthropic::claude-sonnet-4-5".into(),
        model_provider_name: "anthropic".into(),
        credentials: HashMap::new(),
    }];

    let extra_body_providers = vec![E2ETestProvider {
        supports_batch_inference: false,
        variant_name: "anthropic-extra-body".to_string(),
        model_name: "claude-haiku-4-5-anthropic".into(),
        model_provider_name: "anthropic".into(),
        credentials: HashMap::new(),
    }];

    let bad_auth_extra_headers = vec![E2ETestProvider {
        supports_batch_inference: false,
        variant_name: "anthropic-extra-headers".to_string(),
        model_name: "claude-haiku-4-5-anthropic".into(),
        model_provider_name: "anthropic".into(),
        credentials: HashMap::new(),
    }];

    let inference_params_dynamic_providers = vec![E2ETestProvider {
        supports_batch_inference: false,
        variant_name: "anthropic-dynamic".to_string(),
        model_name: "claude-haiku-4-5-anthropic-dynamic".into(),
        model_provider_name: "anthropic".into(),
        credentials,
    }];

    let json_providers = vec![
        E2ETestProvider {
            supports_batch_inference: false,
            variant_name: "anthropic".to_string(),
            model_name: "claude-haiku-4-5-anthropic".into(),
            model_provider_name: "anthropic".into(),
            credentials: HashMap::new(),
        },
        E2ETestProvider {
            supports_batch_inference: false,
            variant_name: "anthropic-implicit".to_string(),
            model_name: "claude-haiku-4-5-anthropic".into(),
            model_provider_name: "anthropic".into(),
            credentials: HashMap::new(),
        },
        E2ETestProvider {
            supports_batch_inference: false,
            variant_name: "anthropic-strict".to_string(),
            model_name: "claude-haiku-4-5-anthropic".into(),
            model_provider_name: "anthropic".into(),
            credentials: HashMap::new(),
        },
    ];

    let json_mode_off_providers = vec![E2ETestProvider {
        supports_batch_inference: false,
        variant_name: "anthropic_json_mode_off".to_string(),
        model_name: "claude-haiku-4-5-anthropic".into(),
        model_provider_name: "anthropic".into(),
        credentials: HashMap::new(),
    }];

    let shorthand_providers = vec![E2ETestProvider {
        supports_batch_inference: false,
        variant_name: "anthropic-shorthand".to_string(),
        model_name: "anthropic::claude-haiku-4-5".into(),
        model_provider_name: "anthropic".into(),
        credentials: HashMap::new(),
    }];

    let provider_type_default_credentials_providers = vec![E2ETestProvider {
        supports_batch_inference: false,
        variant_name: "anthropic".to_string(),
        model_name: "claude-haiku-4-5".into(),
        model_provider_name: "anthropic".into(),
        credentials: HashMap::new(),
    }];

    let provider_type_default_credentials_shorthand_providers = vec![E2ETestProvider {
        supports_batch_inference: false,
        variant_name: "anthropic-shorthand".to_string(),
        model_name: "anthropic::claude-haiku-4-5".into(),
        model_provider_name: "anthropic".into(),
        credentials: HashMap::new(),
    }];

    let credential_fallbacks = vec![ModelTestProvider {
        provider_type: "anthropic".into(),
        model_info: HashMap::from([("model_name".to_string(), "claude-haiku-4-5".to_string())]),
        use_modal_headers: false,
    }];

    let reasoning_providers = vec![
        E2ETestProvider {
            supports_batch_inference: false,
            variant_name: "anthropic-haiku-4-5-thinking".to_string(),
            model_name: "claude-haiku-4-5-thinking".into(),
            model_provider_name: "anthropic".into(),
            credentials: HashMap::new(),
        },
        E2ETestProvider {
            supports_batch_inference: false,
            variant_name: "anthropic-sonnet-4-6-reasoning".to_string(),
            model_name: "claude-sonnet-4-6".into(),
            model_provider_name: "anthropic".into(),
            credentials: HashMap::new(),
        },
    ];

    E2ETestProviders {
        simple_inference: standard_providers.clone(),
        bad_auth_extra_headers,
        extra_body_inference: extra_body_providers,
        reasoning_inference: reasoning_providers.clone(),
        reasoning_usage_inference: reasoning_providers,
        cache_input_tokens_inference: standard_providers.clone(),
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

    // NOTE: This test expects an error response (BAD_GATEWAY), which provider-proxy
    // does not cache. Therefore, this test will always hit live providers.
    // Consider moving to periodic live tests if this causes merge queue instability.
    // See: https://github.com/tensorzero/tensorzero/issues/5380
    let random = Uuid::now_v7();
    let payload = json!({
        "model_name": "anthropic::claude-sonnet-4-5",
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

#[tokio::test]
async fn test_empty_chunks_success() {
    let payload = json!({
        "model_name": "anthropic::claude-sonnet-4-5",
        "input":{
            "messages": [
                {
                    "role": "assistant",
                    "content": "Can you clarify that?"
                }
            ]},
        "params": {
            "chat_completion": {
                "max_tokens": 256,
            }
        },
        "stream": true,
    });

    let client = Client::new();

    let mut event_source = client
        .post(get_gateway_endpoint("/inference"))
        .json(&payload)
        .eventsource()
        .await
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

    println!("Chunks: {chunks:?}");
    let mut first_inference_id = None;

    for chunk in chunks {
        let chunk_json: Value = serde_json::from_str(&chunk).unwrap();
        let inference_id = chunk_json.get("inference_id").unwrap().as_str().unwrap();
        let inference_id = Uuid::parse_str(inference_id).unwrap();
        if first_inference_id.is_none() {
            first_inference_id = Some(inference_id);
        }
        assert_eq!(chunk_json["content"].as_array().unwrap().len(), 0);
    }

    let conn = DelegatingDatabaseConnection::new_for_e2e_test().await;
    conn.flush_pending_writes().await;
    conn.sleep_for_writes_to_be_visible().await;
    let config = get_e2e_config().await;

    let inferences = conn
        .list_inferences(
            &config,
            &ListInferencesParams {
                ids: Some(&[first_inference_id.unwrap()]),
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
    println!("Chat inference: {chat_inf:?}");
    let output = chat_inf.output.as_ref().expect("output should be present");
    assert!(output.is_empty(), "Expected empty output, got {output:?}");
    expect_that!(
        chat_inf,
        matches_pattern!(StoredChatInferenceDatabase {
            ttft_ms: some(gt(&0u64)),
            ..
        })
    );
}

#[tokio::test]
pub async fn test_thinking_signature() {
    test_thinking_signature_helper(
        "anthropic-thinking",
        "anthropic::claude-sonnet-4-5",
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

    let conn = DelegatingDatabaseConnection::new_for_e2e_test().await;
    conn.flush_pending_writes().await;
    conn.sleep_for_writes_to_be_visible().await;
    let config = get_e2e_config().await;

    // First, check Inference table
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
            function_name: eq("weather_helper"),
            episode_id: eq(&episode_id),
            variant_name: eq(variant_name),
            processing_time_ms: some(gt(&0u64)),
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
                    "content": [{"type": "text", "text": "Hi I'm visiting Brooklyn from Brazil. What's the weather (use degrees Celsius)?"}]
                }
            ]
        }))
    );

    // Check the ModelInference Table
    let model_inferences = conn
        .get_model_inferences_by_inference_id(inference_id)
        .await
        .unwrap();
    assert_that!(model_inferences, len(eq(1)));
    let mi = &model_inferences[0];

    let raw_request = mi
        .raw_request
        .as_ref()
        .expect("raw_request should be present");
    assert!(raw_request.to_lowercase().contains("weather"));
    let _: Value = serde_json::from_str(raw_request).expect("raw_request should be valid JSON");

    let raw_response = mi
        .raw_response
        .as_ref()
        .expect("raw_response should be present");
    let _raw_response_json: Value = serde_json::from_str(raw_response).unwrap();

    expect_that!(
        mi,
        matches_pattern!(StoredModelInference {
            inference_id: eq(&inference_id),
            model_name: eq(model_name),
            model_provider_name: eq(model_provider_name),
            input_tokens: some(gt(&5u32)),
            output_tokens: some(gt(&5u32)),
            response_time_ms: some(gt(&0u32)),
            ttft_ms: none(),
            ..
        })
    );

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
    test_redacted_thinking_helper("anthropic::claude-sonnet-4-5", "anthropic", "anthropic").await;
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
    assert_eq!(first_block["provider_type"], provider_type);
    assert!(first_block["signature"].as_str().is_some());

    let second_block = &content_blocks[1];
    assert_eq!(second_block["type"], "text");
    let content = second_block.get("text").unwrap().as_str().unwrap();
    // Assert that Tokyo is in the content
    assert!(content.contains("Tokyo"), "Content should mention Tokyo");
    // Check that inference_id is here
    let inference_id = response_json.get("inference_id").unwrap().as_str().unwrap();
    let inference_id = Uuid::parse_str(inference_id).unwrap();

    let conn = DelegatingDatabaseConnection::new_for_e2e_test().await;
    conn.flush_pending_writes().await;
    conn.sleep_for_writes_to_be_visible().await;
    let config = get_e2e_config().await;

    // First, check Inference table
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
            function_name: eq("tensorzero::default"),
            episode_id: eq(&episode_id),
            variant_name: eq(model_name),
            processing_time_ms: some(gt(&0u64)),
            ..
        })
    );
    expect_that!(
        serde_json::to_value(&chat_inf.input),
        ok(matches_json_literal!({
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
        }))
    );

    let output = chat_inf.output.as_ref().expect("output should be present");
    assert_that!(output, len(eq(2)));

    // Check first block is a thought block
    let output_json = serde_json::to_value(output).unwrap();
    let first_block = &output_json[0];
    assert_eq!(first_block["type"], "thought");
    assert_eq!(first_block["provider_type"], provider_type);
    assert!(first_block["signature"].as_str().is_some());

    // Check second block is text with correct content
    let second_block = &output_json[1];
    assert_eq!(second_block["type"], "text");
    let db_content = second_block.get("text").unwrap().as_str().unwrap();
    assert_eq!(db_content, content);

    // Check the ModelInference Table
    let model_inferences = conn
        .get_model_inferences_by_inference_id(inference_id)
        .await
        .unwrap();
    assert_that!(model_inferences, len(eq(1)));
    let mi = &model_inferences[0];

    let raw_request = mi
        .raw_request
        .as_ref()
        .expect("raw_request should be present");
    assert!(raw_request.to_lowercase().contains("japan"));
    let _: Value = serde_json::from_str(raw_request).expect("raw_request should be valid JSON");

    let raw_response = mi
        .raw_response
        .as_ref()
        .expect("raw_response should be present");
    let _raw_response_json: Value = serde_json::from_str(raw_response).unwrap();

    expect_that!(
        mi,
        matches_pattern!(StoredModelInference {
            inference_id: eq(&inference_id),
            model_name: eq(model_name),
            model_provider_name: eq(model_provider_name),
            input_tokens: some(gt(&5u32)),
            output_tokens: some(gt(&5u32)),
            response_time_ms: some(gt(&0u32)),
            ttft_ms: none(),
            ..
        })
    );

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
    array.push(serde_json::json!({
        "role": "user",
        "content": [{"type": "text", "text": "What were you thinking about?"}],
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

    println!("New response JSON: {response_json}");

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
            .await
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

    let conn = DelegatingDatabaseConnection::new_for_e2e_test().await;
    conn.flush_pending_writes().await;
    conn.sleep_for_writes_to_be_visible().await;

    let model_inferences = conn
        .get_model_inferences_by_inference_id(inference_id)
        .await
        .unwrap();
    assert_that!(model_inferences, len(eq(1)));
    let mi = &model_inferences[0];
    println!("Result: {mi:?}");

    let raw_request = mi
        .raw_request
        .as_ref()
        .expect("raw_request should be present");
    if stream {
        assert_eq!(
            raw_request,
            "{\"model\":\"claude-sonnet-4-5\",\"messages\":[{\"role\":\"user\",\"content\":[{\"type\":\"text\",\"text\":\"What is the capital of Japan?\"}]}],\"max_tokens\":100,\"stream\":true,\"output_config\":{\"format\":{\"type\":\"json_schema\",\"schema\":{\"type\":\"object\",\"properties\":{\"answer\":{\"type\":\"string\"}},\"required\":[\"answer\"],\"additionalProperties\":false}}}}"
        );
    } else {
        assert_eq!(
            raw_request,
            "{\"model\":\"claude-sonnet-4-5\",\"messages\":[{\"role\":\"user\",\"content\":[{\"type\":\"text\",\"text\":\"What is the capital of Japan?\"}]}],\"max_tokens\":100,\"stream\":false,\"output_config\":{\"format\":{\"type\":\"json_schema\",\"schema\":{\"type\":\"object\",\"properties\":{\"answer\":{\"type\":\"string\"}},\"required\":[\"answer\"],\"additionalProperties\":false}}}}"
        );
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
            .await
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

    let conn = DelegatingDatabaseConnection::new_for_e2e_test().await;
    conn.flush_pending_writes().await;
    conn.sleep_for_writes_to_be_visible().await;

    let model_inferences = conn
        .get_model_inferences_by_inference_id(inference_id)
        .await
        .unwrap();
    assert_that!(model_inferences, len(eq(1)));
    let mi = &model_inferences[0];
    println!("Result: {mi:?}");

    let raw_request = mi
        .raw_request
        .as_ref()
        .expect("raw_request should be present");
    if stream {
        assert_eq!(
            raw_request,
            "{\"model\":\"claude-sonnet-4-5\",\"messages\":[{\"role\":\"user\",\"content\":[{\"type\":\"text\",\"text\":\"What is the capital of Japan?\"}]}],\"max_tokens\":100,\"stream\":true,\"tool_choice\":{\"type\":\"auto\",\"disable_parallel_tool_use\":false},\"tools\":[{\"name\":\"answer_question\",\"description\":\"End the search process and answer a question. Returns the answer to the question.\",\"input_schema\":{\"$schema\":\"http://json-schema.org/draft-07/schema#\",\"type\":\"object\",\"description\":\"End the search process and answer a question. Returns the answer to the question.\",\"properties\":{\"answer\":{\"type\":\"string\",\"description\":\"The answer to the question.\"}},\"required\":[\"answer\"],\"additionalProperties\":false},\"strict\":true}]}"
        );
    } else {
        assert_eq!(
            raw_request,
            "{\"model\":\"claude-sonnet-4-5\",\"messages\":[{\"role\":\"user\",\"content\":[{\"type\":\"text\",\"text\":\"What is the capital of Japan?\"}]}],\"max_tokens\":100,\"stream\":false,\"tool_choice\":{\"type\":\"auto\",\"disable_parallel_tool_use\":false},\"tools\":[{\"name\":\"answer_question\",\"description\":\"End the search process and answer a question. Returns the answer to the question.\",\"input_schema\":{\"$schema\":\"http://json-schema.org/draft-07/schema#\",\"type\":\"object\",\"description\":\"End the search process and answer a question. Returns the answer to the question.\",\"properties\":{\"answer\":{\"type\":\"string\",\"description\":\"The answer to the question.\"}},\"required\":[\"answer\"],\"additionalProperties\":false},\"strict\":true}]}"
        );
    }
}

/// This test checks that streaming inference works as expected.
#[tokio::test]
async fn test_streaming_thinking() {
    test_streaming_thinking_helper("anthropic::claude-sonnet-4-5", "anthropic").await;
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
        .await
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

    let conn = DelegatingDatabaseConnection::new_for_e2e_test().await;
    conn.flush_pending_writes().await;
    conn.sleep_for_writes_to_be_visible().await;
    let config = get_e2e_config().await;

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
    assert_that!(chat_inf.inference_id, eq(inference_id));

    expect_that!(
        serde_json::to_value(&chat_inf.input),
        ok(matches_json_literal!({
            "system": "Always thinking before responding",
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": "What is the capital of Japan?"}]
                }
            ]
        }))
    );

    // Check content blocks
    let output = chat_inf.output.as_ref().expect("output should be present");
    let db_content_blocks = serde_json::to_value(output).unwrap();
    let db_content_blocks = db_content_blocks.as_array().unwrap();
    println!("Got content blocks: {db_content_blocks:?}");
    assert!(
        db_content_blocks.len() == 2 || db_content_blocks.len() == 3,
        "Expected 2 or 3 content blocks in database, got {}",
        db_content_blocks.len()
    );
    assert_eq!(db_content_blocks[0]["type"], "thought");

    let tool_call_index = db_content_blocks.len() - 1;
    let has_text_block = db_content_blocks.len() == 3;

    if has_text_block {
        assert_eq!(db_content_blocks[1]["type"], "text");
    }
    assert_eq!(db_content_blocks[tool_call_index]["type"], "tool_call");

    assert_eq!(
        db_content_blocks[0],
        serde_json::json!({
            "type": "thought",
            "text": content_blocks[&("thought".to_string(), "0".to_string())],
            "signature": content_block_signatures["0"],
            "provider_type": provider_type,
        })
    );

    if has_text_block {
        assert_eq!(
            db_content_blocks[1]["text"],
            content_blocks[&("text".to_string(), "1".to_string())]
        );
    }

    let tool_call_id = db_content_blocks[tool_call_index]["id"].as_str().unwrap();

    assert_eq!(
        db_content_blocks[tool_call_index]["raw_arguments"],
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
                    "content": db_content_blocks
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
        model_name: Some("anthropic::claude-haiku-4-5".to_string()),
        input: Input {
            messages: vec![InputMessage {
                role: Role::User,
                content: vec![InputMessageContent::Text(Text { text: "Describe the contents of the image".to_string() }),
                InputMessageContent::File(File::Url(UrlFile {
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

    let conn = DelegatingDatabaseConnection::new_for_e2e_test().await;
    conn.flush_pending_writes().await;
    conn.sleep_for_writes_to_be_visible().await;

    let model_inferences = conn
        .get_model_inferences_by_inference_id(response.inference_id())
        .await
        .unwrap();
    assert_that!(model_inferences, len(eq(1)));
    let mi = &model_inferences[0];

    let raw_request = mi
        .raw_request
        .as_ref()
        .expect("raw_request should be present");
    assert_eq!(
        raw_request,
        "{\"model\":\"claude-haiku-4-5\",\"messages\":[{\"role\":\"user\",\"content\":[{\"type\":\"text\",\"text\":\"Describe the contents of the image\"},{\"type\":\"image\",\"source\":{\"type\":\"url\",\"url\":\"https://raw.githubusercontent.com/tensorzero/tensorzero/ff3e17bbd3e32f483b027cf81b54404788c90dc1/tensorzero-internal/tests/e2e/providers/ferris.png\"}}]}],\"max_tokens\":64000,\"stream\":false}"
    );

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
        model_name: Some("anthropic::claude-sonnet-4-5".to_string()),
        input: Input {
            messages: vec![InputMessage {
                role: Role::User,
                content: vec![InputMessageContent::Text(Text { text: "Describe the contents of the PDF".to_string() }),
                InputMessageContent::File(File::Url(UrlFile {
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

    let conn = DelegatingDatabaseConnection::new_for_e2e_test().await;
    conn.flush_pending_writes().await;
    conn.sleep_for_writes_to_be_visible().await;

    let model_inferences = conn
        .get_model_inferences_by_inference_id(response.inference_id())
        .await
        .unwrap();
    assert_that!(model_inferences, len(eq(1)));
    let mi = &model_inferences[0];

    let raw_request = mi
        .raw_request
        .as_ref()
        .expect("raw_request should be present");
    assert_eq!(
        raw_request,
        "{\"model\":\"claude-sonnet-4-5\",\"messages\":[{\"role\":\"user\",\"content\":[{\"type\":\"text\",\"text\":\"Describe the contents of the PDF\"},{\"type\":\"document\",\"source\":{\"type\":\"url\",\"url\":\"https://raw.githubusercontent.com/tensorzero/tensorzero/ac37477d56deaf6e0585a394eda68fd4f9390cab/tensorzero-core/tests/e2e/providers/deepseek_paper.pdf\"}}]}],\"max_tokens\":64000,\"stream\":false}"
    );

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
