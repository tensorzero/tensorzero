use crate::common::get_gateway_endpoint;
use crate::utils::skip_for_postgres;
use crate::{
    otel::{
        CapturingOtelExporter, SpanMap, attrs_to_map, build_span_map,
        install_capturing_otel_exporter,
    },
    providers::common::FERRIS_PNG,
};
use base64::prelude::{BASE64_STANDARD, Engine as Base64Engine};
use futures::StreamExt;
use googletest::prelude::*;
use googletest_matchers::{matches_json_literal, partially};
use opentelemetry_sdk::trace::SpanData;
use reqwest::{Client, StatusCode};
use reqwest_sse_stream::{Event, RequestBuilderExt};
use serde_json::{Value, json};
use std::collections::HashSet;
use tensorzero::{
    ClientInferenceParams, InferenceOutput, InferenceResponse, Input, InputMessage,
    InputMessageContent, StoredChatInferenceDatabase,
};
use tensorzero_core::db::delegating_connection::DelegatingDatabaseConnection;
use tensorzero_core::db::inferences::{InferenceQueries, ListInferencesParams};
use tensorzero_core::db::model_inferences::ModelInferenceQueries;
use tensorzero_core::db::test_helpers::TestDatabaseHelpers;
use tensorzero_core::inference::types::{
    Arguments, ContentBlockOutput, StoredModelInference, System, Template,
};
use tensorzero_core::observability::enter_fake_http_request_otel;
use tensorzero_core::stored_inference::StoredInferenceDatabase;
use tensorzero_core::test_helpers::get_e2e_config;
use tensorzero_core::{
    endpoints::inference::ChatInferenceResponse,
    inference::types::{
        Base64File, File, RawText, Role, StoredContentBlock, StoredInputMessageContent,
        StoredRequestMessage, Text, Unknown,
    },
    providers::dummy::{
        DUMMY_BAD_TOOL_RESPONSE, DUMMY_INFER_RESPONSE_CONTENT, DUMMY_INFER_RESPONSE_RAW,
        DUMMY_JSON_RAW_RESPONSE, DUMMY_JSON_RESPONSE_RAW, DUMMY_RAW_REQUEST,
        DUMMY_STREAMING_RESPONSE, DUMMY_STREAMING_TOOL_RESPONSE, DUMMY_TOOL_RAW_RESPONSE,
        DUMMY_TOOL_RESPONSE,
    },
    tool::{ToolCall, ToolCallWrapper},
};
use uuid::Uuid;

mod extra_body;
mod extra_headers;
pub mod json_mode_tool;
mod openai_compatible_chat_completions;
pub mod tool_params;

#[gtest]
#[tokio::test]
async fn test_inference_dryrun() {
    let payload = json!({
        "function_name": "basic_test",
        "episode_id": Uuid::now_v7(),
        "input": {
            "system": {"assistant_name": "AskJeeves"},
            "messages": [
                {
                    "role": "user",
                    "content": "Hello, world!"
                }
            ]
        },
        "stream": false,
        "dryrun": true,
    });

    let response = Client::new()
        .post(get_gateway_endpoint("/inference"))
        .json(&payload)
        .send()
        .await
        .unwrap();

    // Check Response is OK, then fields in order
    assert_that!(response.status(), eq(StatusCode::OK));
    let response_json = response.json::<Value>().await.unwrap();

    // Check that inference_id is here
    let inference_id = response_json.get("inference_id").unwrap().as_str().unwrap();
    let inference_id = Uuid::parse_str(inference_id).unwrap();

    // Check that no inference was written to the database when dryrun is true
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

    expect_that!(
        inferences,
        is_empty(),
        "No inference should be written when dryrun is true"
    );
}

#[gtest]
#[tokio::test]
async fn test_inference_chat_strip_unknown_block_non_stream() {
    let payload = json!({
        "function_name": "basic_test",
        "episode_id": Uuid::now_v7(),
        "input": {
            "system": {"assistant_name": "AskJeeves"},
            "messages": [
                {
                    "role": "user",
                    "content": "Hello, world!"
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "unknown", "model_name": "test", "provider_name": "good", "data": {"my": "custom data"}},
                        {"type": "unknown", "model_name": "test", "provider_name": "wrong_model_name", "data": {"SHOULD NOT": "SHOW UP"}},
                        {"type": "unknown", "data": "Non-provider-specific unknown block"}
                    ]
                }
            ]
        },
        "stream": false,
        "internal": true, // This also tests that the internal flag is correctly propagated.
        "tags": {
            "tensorzero::tag_key": "tensorzero::tag_value"
        }
    });

    let response = Client::new()
        .post(get_gateway_endpoint("/inference"))
        .json(&payload)
        .send()
        .await
        .unwrap();

    // Check Response is OK, then fields in order
    assert_eq!(response.status(), StatusCode::OK);
    let response_json = response.json::<Value>().await.unwrap();

    // Check that inference_id is here
    let inference_id = response_json.get("inference_id").unwrap().as_str().unwrap();
    let inference_id = Uuid::parse_str(inference_id).unwrap();

    let episode_id = response_json.get("episode_id").unwrap().as_str().unwrap();
    let episode_id = Uuid::parse_str(episode_id).unwrap();

    // Check database
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

    let chat = match &inferences[0] {
        StoredInferenceDatabase::Chat(c) => c,
        StoredInferenceDatabase::Json(_) => panic!("Expected chat inference"),
    };

    expect_that!(
        chat,
        matches_pattern!(StoredChatInferenceDatabase {
            inference_id: eq(&inference_id),
            episode_id: eq(&episode_id),
            variant_name: eq("test"),
            // Should have snapshot_hash
            snapshot_hash: some(anything()),
            ..
        })
    );

    // The input in ChatInference should have *all* of the unknown blocks present
    let input = serde_json::to_value(&chat.input).unwrap();
    expect_that!(
        input,
        matches_json_literal!({
            "system": {
                "assistant_name": "AskJeeves"
            },
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": "Hello, world!"}]
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "unknown", "model_name": "test", "provider_name": "good", "data": {"my": "custom data"}},
                        {"type": "unknown", "model_name": "test", "provider_name": "wrong_model_name", "data": {"SHOULD NOT": "SHOW UP"}},
                        {"type": "unknown", "data": "Non-provider-specific unknown block"}
                    ]
                }
            ]
        })
    );

    // Check that content blocks are correct
    let output = serde_json::to_value(&chat.output).expect("Output should parse correctly");
    let content_blocks = output
        .as_array()
        .expect("Output should be an array of content blocks");
    expect_that!(
        content_blocks,
        elements_are![matches_json_literal!({
            "type": "text",
            "text": DUMMY_INFER_RESPONSE_CONTENT
        })]
    );

    // Check the ModelInference Table
    let model_inferences = conn
        .get_model_inferences_by_inference_id(inference_id)
        .await
        .unwrap();
    assert_that!(model_inferences, len(eq(1)));
    expect_that!(
        model_inferences[0],
        matches_pattern!(StoredModelInference {
            inference_id: eq(&inference_id),
            // Assert ModelInference has snapshot_hash
            snapshot_hash: some(anything()),
            input_messages: some(elements_are![
                eq(&StoredRequestMessage {
                    role: Role::User,
                    content: vec![StoredContentBlock::Text(Text {
                        text: "Hello, world!".to_string(),
                    })]
                }),
                eq(&StoredRequestMessage {
                    role: Role::User,
                    content: vec![
                        StoredContentBlock::Unknown(Unknown {
                            model_name: Some("test".to_string()),
                            provider_name: Some("good".to_string()),
                            data: json!({"my": "custom data"})
                        }),
                        StoredContentBlock::Unknown(Unknown {
                            model_name: None,
                            provider_name: None,
                            data: "Non-provider-specific unknown block".into()
                        })
                    ]
                })
            ]),
            ..
        })
    );
    // Materialized view checks for snapshot_hash are in inference_clickhouse.rs
}

#[tokio::test]
async fn test_dummy_only_inference_chat_strip_unknown_block_stream() {
    let payload = json!({
        "function_name": "basic_test",
        "episode_id": Uuid::now_v7(),
        "input": {
            "system": {"assistant_name": "AskJeeves"},
            "messages": [
                {
                    "role": "user",
                    "content": "Hello, world!"
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "unknown", "model_name": "test", "provider_name": "good", "data": {"my": "custom data"}},
                        {"type": "unknown", "model_name": "test", "provider_name": "wrong_model_name", "data": {"SHOULD NOT": "SHOW UP"}},
                        {"type": "unknown", "data": "Non-provider-specific unknown block"}
                    ]
                }
            ]
        },
        "stream": true,
    });

    let mut event_source = Client::new()
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

    let chunk_json: serde_json::Value = serde_json::from_str(chunks.last().unwrap()).unwrap();
    let inference_id =
        Uuid::parse_str(chunk_json.get("inference_id").unwrap().as_str().unwrap()).unwrap();

    let episode_id =
        Uuid::parse_str(chunk_json.get("episode_id").unwrap().as_str().unwrap()).unwrap();

    // Check database
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
    assert_eq!(inferences.len(), 1);
    let chat = match &inferences[0] {
        StoredInferenceDatabase::Chat(c) => c,
        StoredInferenceDatabase::Json(_) => panic!("Expected chat inference"),
    };
    assert_eq!(chat.inference_id, inference_id);
    let input = serde_json::to_value(&chat.input).unwrap();

    // The input in ChatInference should have *all* of the unknown blocks present
    let correct_input: Value = json!(
        {
            "system": {
                "assistant_name": "AskJeeves"
            },
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": "Hello, world!"}]
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "unknown", "model_name": "test", "provider_name": "good", "data": {"my": "custom data"}},
                        {"type": "unknown", "model_name": "test", "provider_name": "wrong_model_name", "data": {"SHOULD NOT": "SHOW UP"}},
                        {"type": "unknown", "data": "Non-provider-specific unknown block"}
                    ]
                }
            ]
        }
    );
    assert_eq!(input, correct_input);
    // Check that content blocks are correct
    let output = serde_json::to_value(&chat.output).unwrap();
    let content_blocks = output.as_array().unwrap();
    assert_eq!(content_blocks.len(), 1);
    let content_block = content_blocks.first().unwrap();
    let content_block_type = content_block.get("type").unwrap().as_str().unwrap();
    assert_eq!(content_block_type, "text");
    let content = content_block.get("text").unwrap().as_str().unwrap();
    assert_eq!(
        content,
        "Wally, the golden retriever, wagged his tail excitedly as he devoured a slice of cheese pizza."
    );
    // Check that episode_id is here and correct
    assert_eq!(chat.episode_id, episode_id);
    // Check the variant name
    assert_eq!(chat.variant_name, "test");
    // Since the variant was not pinned, the variant_pinned tag should not be present
    assert!(!chat.tags.contains_key("tensorzero::variant_pinned"));

    // Check the ModelInference Table
    let model_inferences = conn
        .get_model_inferences_by_inference_id(inference_id)
        .await
        .unwrap();
    assert_eq!(model_inferences.len(), 1);
    let mi = &model_inferences[0];
    let _ = mi.id;
    assert_eq!(mi.inference_id, inference_id);

    let input_messages = mi.input_messages.as_ref().unwrap();
    assert_eq!(
        *input_messages,
        vec![
            StoredRequestMessage {
                role: Role::User,
                content: vec![StoredContentBlock::Text(Text {
                    text: "Hello, world!".to_string(),
                })]
            },
            StoredRequestMessage {
                role: Role::User,
                content: vec![
                    StoredContentBlock::Unknown(Unknown {
                        model_name: Some("test".to_string()),
                        provider_name: Some("good".to_string()),
                        data: json!({"my": "custom data"})
                    }),
                    StoredContentBlock::Unknown(Unknown {
                        model_name: None,
                        provider_name: None,
                        data: "Non-provider-specific unknown block".into()
                    })
                ]
            },
        ]
    );
}

/// This test calls a function which calls a model where the first provider is broken but
/// then the second provider works fine. We expect this request to work despite the first provider
/// being broken.
#[tokio::test]
async fn test_inference_model_fallback() {
    let episode_id = Uuid::now_v7();

    let payload = json!({
        "function_name": "model_fallback_test",
        "episode_id": episode_id,
        "input":{
            "system": {"assistant_name": "AskJeeves"},
            "messages": [
                {
                    "role": "user",
                    "content": "Hello, world!"
                }
            ]},
        "stream": false,
    });

    let response = Client::new()
        .post(get_gateway_endpoint("/inference"))
        .json(&payload)
        .send()
        .await
        .unwrap();
    // Check Response is OK, then fields in order
    assert_eq!(response.status(), StatusCode::OK);
    let response_json = response.json::<Value>().await.unwrap();
    // Check that inference_id is here
    let inference_id = response_json.get("inference_id").unwrap().as_str().unwrap();
    let inference_id = Uuid::parse_str(inference_id).unwrap();
    // Check that raw_content is same as content
    let content_blocks: &Vec<Value> = response_json.get("content").unwrap().as_array().unwrap();
    assert_eq!(content_blocks.len(), 1);
    let content_block = content_blocks.first().unwrap();
    let content_block_type = content_block.get("type").unwrap().as_str().unwrap();
    assert_eq!(content_block_type, "text");
    let content = content_block.get("text").unwrap().as_str().unwrap();
    assert_eq!(content, DUMMY_INFER_RESPONSE_CONTENT);

    // Check that usage is correct
    let usage = response_json.get("usage").unwrap();
    let usage = usage.as_object().unwrap();
    let input_tokens = usage.get("input_tokens").unwrap().as_u64().unwrap();
    let output_tokens = usage.get("output_tokens").unwrap().as_u64().unwrap();
    assert_eq!(input_tokens, 10);
    assert_eq!(output_tokens, 1);
    // Check database
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
    assert_eq!(inferences.len(), 1);
    let chat = match &inferences[0] {
        StoredInferenceDatabase::Chat(c) => c,
        StoredInferenceDatabase::Json(_) => panic!("Expected chat inference"),
    };
    assert_eq!(chat.inference_id, inference_id);
    let input = serde_json::to_value(&chat.input).unwrap();
    let correct_input: Value = json!(
        {
            "system": {
                "assistant_name": "AskJeeves"
            },
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": "Hello, world!"}]
                }
            ]
        }
    );
    assert_eq!(input, correct_input);
    // Check that content blocks are correct
    let output = serde_json::to_value(&chat.output).unwrap();
    let content_blocks = output.as_array().unwrap();
    assert_eq!(content_blocks.len(), 1);
    let content_block = content_blocks.first().unwrap();
    let content_block_type = content_block.get("type").unwrap().as_str().unwrap();
    assert_eq!(content_block_type, "text");
    let content = content_block.get("text").unwrap().as_str().unwrap();
    assert_eq!(content, DUMMY_INFER_RESPONSE_CONTENT);
    // Check that episode_id is here and correct
    assert_eq!(chat.episode_id, episode_id);
    // Check the variant name
    assert_eq!(chat.variant_name, "test");

    // Check the ModelInference Table
    let model_inferences = conn
        .get_model_inferences_by_inference_id(inference_id)
        .await
        .unwrap();
    assert_eq!(model_inferences.len(), 1);
    let mi = &model_inferences[0];
    let _ = mi.id;
    assert_eq!(mi.inference_id, inference_id);

    assert_eq!(mi.input_tokens.unwrap(), 10);
    assert_eq!(mi.output_tokens.unwrap(), 1);
    assert!(mi.response_time_ms.unwrap() > 0);
    assert!(mi.ttft_ms.is_none());
    assert_eq!(
        mi.raw_response.as_deref().unwrap(),
        DUMMY_INFER_RESPONSE_RAW
    );
    assert_eq!(mi.raw_request.as_deref().unwrap(), DUMMY_RAW_REQUEST);
    assert_eq!(
        mi.system.as_deref().unwrap(),
        "You are a helpful and friendly assistant named AskJeeves"
    );
    let input_messages = mi.input_messages.as_ref().unwrap();
    assert_eq!(input_messages.len(), 1);
    assert_eq!(
        input_messages[0],
        StoredRequestMessage {
            role: Role::User,
            content: vec![StoredContentBlock::Text(Text {
                text: "Hello, world!".to_string(),
            })],
        }
    );
    let output = mi.output.as_ref().unwrap();
    assert_eq!(
        *output,
        vec![ContentBlockOutput::Text(Text {
            text: DUMMY_INFER_RESPONSE_CONTENT.to_string(),
        })]
    );
}

#[gtest]
#[tokio::test]
async fn test_tool_call() {
    let episode_id = Uuid::now_v7();

    let payload = json!({
        "function_name": "weather_helper",
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
    let response = Client::new()
        .post(get_gateway_endpoint("/inference"))
        .json(&payload)
        .send()
        .await
        .unwrap();
    // Check Response is OK, then fields in order
    assert_eq!(response.status(), StatusCode::OK);
    let response_json = response.json::<Value>().await.unwrap();
    // No output schema so parsed content should not be in response
    assert!(response_json.get("parsed_content").is_none());
    // Check that inference_id is here
    let inference_id = response_json.get("inference_id").unwrap().as_str().unwrap();
    let inference_id = Uuid::parse_str(inference_id).unwrap();
    // Check that raw_content is same as content
    let content_blocks: &Vec<Value> = response_json.get("content").unwrap().as_array().unwrap();
    assert_eq!(content_blocks.len(), 1);
    let content_block = content_blocks.first().unwrap();
    let content_block_type = content_block.get("type").unwrap().as_str().unwrap();
    assert_eq!(content_block_type, "tool_call");
    let raw_name = content_block.get("raw_name").unwrap().as_str().unwrap();
    assert_eq!(raw_name, "get_temperature");
    let raw_arguments = content_block
        .get("raw_arguments")
        .unwrap()
        .as_str()
        .unwrap();
    let raw_arguments: Value = serde_json::from_str(raw_arguments).unwrap();
    assert_eq!(raw_arguments, *DUMMY_TOOL_RESPONSE);
    let arguments = content_block.get("arguments").unwrap().as_object().unwrap();
    assert_eq!(arguments, DUMMY_TOOL_RESPONSE.as_object().unwrap());
    let id = content_block.get("id").unwrap().as_str().unwrap();
    assert_eq!(id, "0");
    let name = content_block.get("name").unwrap().as_str().unwrap();
    assert_eq!(name, "get_temperature");

    // Check that type is "chat"
    // Check that usage is correct
    let usage = response_json.get("usage").unwrap();
    let usage = usage.as_object().unwrap();
    let input_tokens = usage.get("input_tokens").unwrap().as_u64().unwrap();
    let output_tokens = usage.get("output_tokens").unwrap().as_u64().unwrap();
    assert_eq!(input_tokens, 10);
    assert_eq!(output_tokens, 1);
    // Check database
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
    assert_eq!(inferences.len(), 1);
    let chat = match &inferences[0] {
        StoredInferenceDatabase::Chat(c) => c,
        StoredInferenceDatabase::Json(_) => panic!("Expected chat inference"),
    };
    assert_eq!(chat.inference_id, inference_id);
    let input = serde_json::to_value(&chat.input).unwrap();
    let correct_input: Value = json!(
        {
            "system": {
                "assistant_name": "AskJeeves"
            },
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": "Hi I'm visiting Brooklyn from Brazil. What's the weather?"}]
                }
            ]
        }
    );
    assert_eq!(input, correct_input);
    // Check that content blocks are correct
    let output = serde_json::to_value(&chat.output).unwrap();
    let content_blocks = output.as_array().unwrap();
    assert_eq!(content_blocks.len(), 1);
    let content_block = content_blocks.first().unwrap();
    let content_block_type = content_block.get("type").unwrap().as_str().unwrap();
    assert_eq!(content_block_type, "tool_call");
    // Check that the tool call is correctly stored
    let id = content_block.get("id").unwrap().as_str().unwrap();
    assert_eq!(id, "0");
    let raw_name = content_block.get("raw_name").unwrap().as_str().unwrap();
    assert_eq!(raw_name, "get_temperature");
    let raw_arguments = content_block
        .get("raw_arguments")
        .unwrap()
        .as_str()
        .unwrap();
    let raw_arguments: Value = serde_json::from_str(raw_arguments).unwrap();
    assert_eq!(raw_arguments, *DUMMY_TOOL_RESPONSE);
    let name = content_block.get("name").unwrap().as_str().unwrap();
    assert_eq!(name, "get_temperature");
    let arguments = content_block.get("arguments").unwrap().as_object().unwrap();
    assert_eq!(arguments, DUMMY_TOOL_RESPONSE.as_object().unwrap());
    // Check that episode_id is here and correct
    assert_eq!(chat.episode_id, episode_id);
    // Check the variant name
    assert_eq!(chat.variant_name, "variant");
    // Check the tool_params
    let tool_params = serde_json::to_value(&chat.tool_params).unwrap();
    expect_that!(
        &tool_params,
        partially(matches_json_literal!({
            "tool_choice": "auto",
            "parallel_tool_calls": null,
            "allowed_tools": {
                "tools": ["get_temperature"],
                "choice": "function_default"
            }
        }))
    );
    // Check the ModelInference Table
    let model_inferences = conn
        .get_model_inferences_by_inference_id(inference_id)
        .await
        .unwrap();
    assert_eq!(model_inferences.len(), 1);
    let mi = &model_inferences[0];
    let _ = mi.id;
    assert_eq!(mi.inference_id, inference_id);

    assert_eq!(mi.input_tokens.unwrap(), 10);
    assert_eq!(mi.output_tokens.unwrap(), 1);
    assert!(mi.response_time_ms.unwrap() > 0);
    assert!(mi.ttft_ms.is_none());
    assert_eq!(
        mi.raw_response.as_deref().unwrap(),
        *DUMMY_TOOL_RAW_RESPONSE
    );
    assert_eq!(mi.raw_request.as_deref().unwrap(), DUMMY_RAW_REQUEST);
    let expected_system = "You are a helpful and friendly assistant named AskJeeves.\n\nPeople will ask you questions about the weather.\n\nIf asked about the weather, just respond with the tool call. Use the \"get_temperature\" tool.\n\nIf provided with a tool result, use it to respond to the user (e.g. \"The weather in New York is 55 degrees Fahrenheit.\").";
    assert_eq!(mi.system.as_deref().unwrap(), expected_system);
    let input_messages = mi.input_messages.as_ref().unwrap();
    let expected_messages = vec![StoredRequestMessage {
        role: Role::User,
        content: vec![
            "Hi I'm visiting Brooklyn from Brazil. What's the weather?"
                .to_string()
                .into(),
        ],
    }];
    assert_eq!(*input_messages, expected_messages);
    let output = mi.output.as_ref().unwrap();
    assert_eq!(output.len(), 1);
    match &output[0] {
        ContentBlockOutput::ToolCall(tool_call) => {
            assert_eq!(tool_call.id, "0");
            assert_eq!(tool_call.name, "get_temperature");
            assert_eq!(
                tool_call.arguments,
                serde_json::to_string(&*DUMMY_TOOL_RESPONSE).unwrap()
            );
        }
        _ => {
            panic!("Expected a tool call block, got {:?}", output[0]);
        }
    }
}

#[gtest]
#[tokio::test]
async fn test_tool_call_malformed() {
    let episode_id = Uuid::now_v7();

    let payload = json!({
        "function_name": "weather_helper",
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
        "variant_name": "bad_tool"
    });
    let response = Client::new()
        .post(get_gateway_endpoint("/inference"))
        .json(&payload)
        .send()
        .await
        .unwrap();
    // Check Response is OK, then fields in order
    assert_eq!(response.status(), StatusCode::OK);
    let response_json = response.json::<Value>().await.unwrap();
    // Check that inference_id is here
    let inference_id = response_json.get("inference_id").unwrap().as_str().unwrap();
    let inference_id = Uuid::parse_str(inference_id).unwrap();
    // Check that raw_content is same as content
    let content_blocks: &Vec<Value> = response_json.get("content").unwrap().as_array().unwrap();
    assert_eq!(content_blocks.len(), 1);
    let content_block = content_blocks.first().unwrap();
    let content_block_type = content_block.get("type").unwrap().as_str().unwrap();
    assert_eq!(content_block_type, "tool_call");
    let raw_name = content_block.get("raw_name").unwrap().as_str().unwrap();
    assert_eq!(raw_name, "get_temperature");
    let raw_arguments = content_block
        .get("raw_arguments")
        .unwrap()
        .as_str()
        .unwrap();
    let raw_arguments: Value = serde_json::from_str(raw_arguments).unwrap();
    assert_eq!(raw_arguments, *DUMMY_BAD_TOOL_RESPONSE);
    let id = content_block.get("id").unwrap().as_str().unwrap();
    assert_eq!(id, "0");
    let name = content_block.get("name").unwrap();
    assert_eq!(name, "get_temperature");
    let arguments = content_block.get("arguments").unwrap();
    assert!(arguments.is_null());

    // Check that type is "chat"
    // Check that usage is correct
    let usage = response_json.get("usage").unwrap();
    let usage = usage.as_object().unwrap();
    let input_tokens = usage.get("input_tokens").unwrap().as_u64().unwrap();
    let output_tokens = usage.get("output_tokens").unwrap().as_u64().unwrap();
    assert_eq!(input_tokens, 10);
    assert_eq!(output_tokens, 1);
    // Check database
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
    assert_eq!(inferences.len(), 1);
    let chat = match &inferences[0] {
        StoredInferenceDatabase::Chat(c) => c,
        StoredInferenceDatabase::Json(_) => panic!("Expected chat inference"),
    };
    assert_eq!(chat.inference_id, inference_id);
    let input = serde_json::to_value(&chat.input).unwrap();
    let correct_input: Value = json!(
        {
            "system": {
                "assistant_name": "AskJeeves"
            },
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": "Hi I'm visiting Brooklyn from Brazil. What's the weather?"}]
                }
            ]
        }
    );
    assert_eq!(input, correct_input);
    // Check that content blocks are correct
    let output = serde_json::to_value(&chat.output).unwrap();
    let content_blocks = output.as_array().unwrap();
    assert_eq!(content_blocks.len(), 1);
    let content_block = content_blocks.first().unwrap();
    let content_block_type = content_block.get("type").unwrap().as_str().unwrap();
    assert_eq!(content_block_type, "tool_call");
    // Check that the tool call is correctly stored
    let raw_arguments = content_block
        .get("raw_arguments")
        .unwrap()
        .as_str()
        .unwrap();
    let raw_arguments: Value = serde_json::from_str(raw_arguments).unwrap();
    assert_eq!(raw_arguments, *DUMMY_BAD_TOOL_RESPONSE);
    let id = content_block.get("id").unwrap().as_str().unwrap();
    assert_eq!(id, "0");
    let raw_name = content_block.get("raw_name").unwrap().as_str().unwrap();
    assert_eq!(raw_name, "get_temperature");
    let name = content_block.get("name").unwrap().as_str().unwrap();
    assert_eq!(name, "get_temperature");
    let arguments = content_block.get("arguments").unwrap();
    assert!(arguments.is_null());
    // Check that episode_id is here and correct
    assert_eq!(chat.episode_id, episode_id);
    // Check the variant name
    assert_eq!(chat.variant_name, "bad_tool");
    // Check the tool_params
    let tool_params = serde_json::to_value(&chat.tool_params).unwrap();
    expect_that!(
        &tool_params,
        partially(matches_json_literal!({
            "tool_choice": "auto",
            "parallel_tool_calls": null,
            "allowed_tools": {
                "tools": ["get_temperature"],
                "choice": "function_default"
            }
        }))
    );
    // Check the ModelInference Table
    let model_inferences = conn
        .get_model_inferences_by_inference_id(inference_id)
        .await
        .unwrap();
    assert_eq!(model_inferences.len(), 1);
    let mi = &model_inferences[0];
    let _ = mi.id;
    assert_eq!(mi.inference_id, inference_id);

    assert_eq!(mi.input_tokens.unwrap(), 10);
    assert_eq!(mi.output_tokens.unwrap(), 1);
    assert!(mi.response_time_ms.unwrap() > 0);
    assert!(mi.ttft_ms.is_none());
    assert!(mi.raw_response.is_some());
    assert_eq!(mi.raw_request.as_deref().unwrap(), DUMMY_RAW_REQUEST);
    let expected_system = "You are a helpful and friendly assistant named AskJeeves.\n\nPeople will ask you questions about the weather.\n\nIf asked about the weather, just respond with the tool call. Use the \"get_temperature\" tool.\n\nIf provided with a tool result, use it to respond to the user (e.g. \"The weather in New York is 55 degrees Fahrenheit.\").";
    assert_eq!(mi.system.as_deref().unwrap(), expected_system);
    let input_messages = mi.input_messages.as_ref().unwrap();
    let expected_messages = vec![StoredRequestMessage {
        role: Role::User,
        content: vec![
            "Hi I'm visiting Brooklyn from Brazil. What's the weather?"
                .to_string()
                .into(),
        ],
    }];
    assert_eq!(*input_messages, expected_messages);
    let output = mi.output.as_ref().unwrap();
    assert_eq!(output.len(), 1);
    match &output[0] {
        ContentBlockOutput::ToolCall(tool_call) => {
            assert_eq!(tool_call.id, "0");
            assert_eq!(tool_call.name, "get_temperature");
            assert_eq!(
                tool_call.arguments,
                serde_json::to_string(&*DUMMY_BAD_TOOL_RESPONSE).unwrap()
            );
        }
        _ => {
            panic!("Expected a tool call block, got {:?}", output[0]);
        }
    }
}

/// This test checks the return type and clickhouse writes for a function with an output schema and
/// a response which does not satisfy the schema.
/// We expect to see a null `parsed_content` field in the response and a null `parsed_content` field in the table.
#[tokio::test]
async fn test_inference_json_fail() {
    let episode_id = Uuid::now_v7();

    let payload = json!({
        "function_name": "json_fail",
        "episode_id": episode_id,
        "input":
            {
                "system": {"assistant_name": "AskJeeves"},
                "messages": [
                {
                    "role": "user",
                    "content": "Hello, world!"
                }
            ]},
        "stream": false,
    });

    let response = Client::new()
        .post(get_gateway_endpoint("/inference"))
        .json(&payload)
        .send()
        .await
        .unwrap();
    // Check Response is OK, then fields in order
    assert_eq!(response.status(), StatusCode::OK);
    let response_json = response.json::<Value>().await.unwrap();
    // Get output field
    let output = response_json.get("output").unwrap();
    assert!(output.get("parsed").unwrap().is_null());
    assert!(output.get("raw").unwrap().as_str().unwrap() == DUMMY_INFER_RESPONSE_CONTENT);
    // Check that inference_id is here
    let inference_id = response_json.get("inference_id").unwrap().as_str().unwrap();
    let inference_id = Uuid::parse_str(inference_id).unwrap();

    // Check that usage is correct
    let usage = response_json.get("usage").unwrap();
    let usage = usage.as_object().unwrap();
    let input_tokens = usage.get("input_tokens").unwrap().as_u64().unwrap();
    let output_tokens = usage.get("output_tokens").unwrap().as_u64().unwrap();
    assert_eq!(input_tokens, 10);
    assert_eq!(output_tokens, 1);
    // Check database
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
    assert_eq!(inferences.len(), 1);
    let json_inf = match &inferences[0] {
        StoredInferenceDatabase::Json(j) => j,
        StoredInferenceDatabase::Chat(_) => panic!("Expected JSON inference"),
    };
    assert_eq!(json_inf.inference_id, inference_id);
    let input = serde_json::to_value(&json_inf.input).unwrap();
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
    assert_eq!(json_inf.episode_id, episode_id);
    let output = serde_json::to_value(&json_inf.output).unwrap();
    assert!(output.get("parsed").unwrap().is_null());
    assert!(output.get("raw").unwrap().as_str().unwrap() == DUMMY_INFER_RESPONSE_CONTENT);
    // Check the variant name
    assert_eq!(json_inf.variant_name, "test");

    // Check the ModelInference Table
    let model_inferences = conn
        .get_model_inferences_by_inference_id(inference_id)
        .await
        .unwrap();
    assert_eq!(model_inferences.len(), 1);
    let mi = &model_inferences[0];
    let _ = mi.id;
    assert_eq!(mi.inference_id, inference_id);

    assert_eq!(mi.input_tokens.unwrap(), 10);
    assert_eq!(mi.output_tokens.unwrap(), 1);
    assert!(mi.response_time_ms.unwrap() > 0);
    assert!(mi.ttft_ms.is_none());
    assert_eq!(
        mi.raw_response.as_deref().unwrap(),
        DUMMY_INFER_RESPONSE_RAW
    );
    assert_eq!(mi.raw_request.as_deref().unwrap(), DUMMY_RAW_REQUEST);
    assert_eq!(
        mi.system.as_deref().unwrap(),
        "You are a helpful and friendly assistant named AskJeeves"
    );
    let input_messages = mi.input_messages.as_ref().unwrap();
    assert_eq!(input_messages.len(), 1);
    assert_eq!(
        input_messages[0],
        StoredRequestMessage {
            role: Role::User,
            content: vec![StoredContentBlock::Text(Text {
                text: "Hello, world!".to_string(),
            })],
        }
    );
    let output = mi.output.as_ref().unwrap();
    assert_eq!(
        *output,
        vec![ContentBlockOutput::Text(Text {
            text: DUMMY_INFER_RESPONSE_CONTENT.to_string(),
        })]
    );
}

/// This test checks the return type and clickhouse writes for a function with an output schema and
/// a response which satisfies the schema.
/// We expect to see a filled-out `content` field in the response and a filled-out `output` field in the table.
#[tokio::test]
async fn test_inference_json_success() {
    let episode_id = Uuid::now_v7();

    let payload = json!({
        "function_name": "json_success",
        "episode_id": episode_id,
        "input":
            {
                "system": {"assistant_name": "AskJeeves"},
                "messages": [
                {
                    "role": "user",
                    "content": [{"type": "template", "name": "user", "arguments": {"country": "Japan"}}]
                }
            ]},
        "stream": false,
    });

    let response = Client::new()
        .post(get_gateway_endpoint("/inference"))
        .json(&payload)
        .send()
        .await
        .unwrap();
    // Check Response is OK, then fields in order
    assert_eq!(response.status(), StatusCode::OK);
    let response_json = response.json::<Value>().await.unwrap();
    let output = response_json.get("output").unwrap().as_object().unwrap();
    let parsed = output.get("parsed").unwrap().as_object().unwrap();
    let answer = parsed.get("answer").unwrap().as_str().unwrap();
    assert_eq!(answer, "Hello");
    let raw = output.get("raw").unwrap().as_str().unwrap();
    assert_eq!(raw, DUMMY_JSON_RESPONSE_RAW);
    // Check that inference_id is here
    let inference_id = response_json.get("inference_id").unwrap().as_str().unwrap();
    let inference_id = Uuid::parse_str(inference_id).unwrap();

    // Check that usage is correct
    let usage = response_json.get("usage").unwrap();
    let usage = usage.as_object().unwrap();
    let input_tokens = usage.get("input_tokens").unwrap().as_u64().unwrap();
    let output_tokens = usage.get("output_tokens").unwrap().as_u64().unwrap();
    assert_eq!(input_tokens, 10);
    assert_eq!(output_tokens, 1);
    // Check database
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
    assert_eq!(inferences.len(), 1);
    let json_inf = match &inferences[0] {
        StoredInferenceDatabase::Json(j) => j,
        StoredInferenceDatabase::Chat(_) => panic!("Expected JSON inference"),
    };
    assert_eq!(json_inf.inference_id, inference_id);
    let correct_input = json!({
        "system": {"assistant_name": "AskJeeves"},
        "messages": [
            {
                "role": "user",
                "content": [{"type": "template", "name": "user", "arguments": {"country": "Japan"}}]
            }
        ]
    });
    let input = serde_json::to_value(&json_inf.input).unwrap();
    assert_eq!(input, correct_input);
    // Check that correctly parsed output is present
    let output = serde_json::to_value(&json_inf.output).unwrap();
    let parsed = output.get("parsed").unwrap().as_object().unwrap();
    let answer = parsed.get("answer").unwrap().as_str().unwrap();
    assert_eq!(answer, "Hello");
    let raw = output.get("raw").unwrap().as_str().unwrap();
    assert_eq!(raw, DUMMY_JSON_RESPONSE_RAW);
    // Check content blocks
    assert_eq!(json_inf.episode_id, episode_id);
    // Check the variant name
    assert_eq!(json_inf.variant_name, "test");
    // Assert JsonInference has snapshot_hash
    // Note: StoredJsonInference does not have a snapshot_hash field in the delegating pattern

    // Check the ModelInference Table
    let model_inferences = conn
        .get_model_inferences_by_inference_id(inference_id)
        .await
        .unwrap();
    assert_eq!(model_inferences.len(), 1);
    let mi = &model_inferences[0];
    let _ = mi.id;
    assert_eq!(mi.inference_id, inference_id);
    // Assert ModelInference has snapshot_hash
    assert!(
        mi.snapshot_hash.is_some(),
        "ModelInference should have snapshot_hash"
    );

    assert_eq!(mi.input_tokens.unwrap(), 10);
    assert_eq!(mi.output_tokens.unwrap(), 1);
    assert!(mi.response_time_ms.unwrap() > 0);
    assert!(mi.ttft_ms.is_none());
    assert_eq!(mi.raw_response.as_deref().unwrap(), DUMMY_JSON_RAW_RESPONSE);
    assert_eq!(mi.raw_request.as_deref().unwrap(), DUMMY_RAW_REQUEST);
    assert_eq!(
        mi.system.as_deref().unwrap(),
        "You are a helpful and friendly assistant named AskJeeves.\n\nPlease answer the questions in a JSON with key \"answer\".\n\nDo not include any other text than the JSON object. Do not include \"```json\" or \"```\" or anything else.\n\nExample Response:\n\n{\n    \"answer\": \"42\"\n}"
    );
    let input_messages = mi.input_messages.as_ref().unwrap();
    assert_eq!(input_messages.len(), 1);
    assert_eq!(
        input_messages[0],
        StoredRequestMessage {
            role: Role::User,
            content: vec![
                "What is the name of the capital city of Japan?"
                    .to_string()
                    .into()
            ],
        }
    );
    let output = mi.output.as_ref().unwrap();
    assert_eq!(
        *output,
        vec![ContentBlockOutput::Text(Text {
            text: "{\"answer\":\"Hello\"}".to_string(),
        })]
    );
}

/// The variant_failover function has two variants: good and error, each with weight 0.5
/// We want to make sure that this does not fail despite the error variant failing every time
/// We do this by making several requests and checking that the response is 200 in each, then checking that
/// the response is correct for the last one.
#[tokio::test]
async fn test_variant_failover() {
    let mut last_response = None;
    let mut last_episode_id = None;
    for _ in 0..50 {
        let episode_id = Uuid::now_v7();

        let payload = json!({
            "function_name": "variant_failover",
            "episode_id": episode_id,
            "input":
                {
                    "system": {
                        "assistant_name": "AskJeeves"
                    },
                    "messages": [
                    {
                        "role": "user",
                        "content": [{"type": "template", "name": "user", "arguments": {"type": "tacos", "quantity": 13}}],
                    }
                ]},
            "stream": false,
        });

        let response = Client::new()
            .post(get_gateway_endpoint("/inference"))
            .json(&payload)
            .send()
            .await
            .unwrap();
        // Check Response is OK, then fields in order
        assert_eq!(response.status(), StatusCode::OK);
        last_response = Some(response);
        last_episode_id = Some(episode_id);
    }
    let response = last_response.unwrap();
    let episode_id = last_episode_id.unwrap();
    let response_json = response.json::<Value>().await.unwrap();
    let content_blocks = response_json.get("content").unwrap().as_array().unwrap();
    assert!(content_blocks.len() == 1);
    let content_block = content_blocks.first().unwrap();
    let content_block_type = content_block.get("type").unwrap().as_str().unwrap();
    assert_eq!(content_block_type, "text");
    let content = content_block.get("text").unwrap().as_str().unwrap();
    assert_eq!(content, DUMMY_INFER_RESPONSE_CONTENT);
    // Check that inference_id is here
    let inference_id = response_json.get("inference_id").unwrap().as_str().unwrap();
    let inference_id = Uuid::parse_str(inference_id).unwrap();

    // Check that usage is correct
    let usage = response_json.get("usage").unwrap();
    let usage = usage.as_object().unwrap();
    let input_tokens = usage.get("input_tokens").unwrap().as_u64().unwrap();
    let output_tokens = usage.get("output_tokens").unwrap().as_u64().unwrap();
    assert_eq!(input_tokens, 10);
    assert_eq!(output_tokens, 1);
    // Check database
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
    assert_eq!(inferences.len(), 1);
    let chat = match &inferences[0] {
        StoredInferenceDatabase::Chat(c) => c,
        StoredInferenceDatabase::Json(_) => panic!("Expected chat inference"),
    };
    assert_eq!(chat.inference_id, inference_id);
    let input = serde_json::to_value(&chat.input).unwrap();
    let correct_input: Value = json!(
        {
            "system": {
                "assistant_name": "AskJeeves"
            },
            "messages": [
            {
                "role": "user",
                "content": [{"type": "template", "name": "user", "arguments": {"type": "tacos", "quantity": 13}}]
            }
        ]}
    );
    assert_eq!(input, correct_input);
    let output = serde_json::to_value(&chat.output).unwrap();
    let content_blocks = output.as_array().unwrap();
    // Check that content_blocks is a list of blocks length 1
    assert_eq!(content_blocks.len(), 1);
    let content_block = content_blocks.first().unwrap();
    // Check the type and content in the block
    let content_block_type = content_block.get("type").unwrap().as_str().unwrap();
    assert_eq!(content_block_type, "text");
    let content = content_block.get("text").unwrap().as_str().unwrap();
    assert_eq!(content, DUMMY_INFER_RESPONSE_CONTENT);
    // Check that episode_id is here and correct
    assert_eq!(chat.episode_id, episode_id);
    // Check the variant name
    assert_eq!(chat.variant_name, "good");

    // Check the inference_params (should be null since neither config or payload has chat_completion)
    let inference_params = serde_json::to_value(&chat.inference_params).unwrap();
    let chat_completion_inference_params = inference_params
        .get("chat_completion")
        .unwrap()
        .as_object()
        .unwrap();

    assert!(
        chat_completion_inference_params
            .get("temperature")
            .is_none()
    );
    let max_tokens = chat_completion_inference_params.get("max_tokens").unwrap();
    assert_eq!(max_tokens.as_u64().unwrap(), 100);
    assert!(chat_completion_inference_params.get("seed").is_none());

    // Check the ModelInference Table
    let model_inferences = conn
        .get_model_inferences_by_inference_id(inference_id)
        .await
        .unwrap();
    assert_eq!(model_inferences.len(), 1);
    let mi = &model_inferences[0];
    assert_eq!(mi.inference_id, inference_id);

    assert_eq!(mi.input_tokens.unwrap(), 10);
    assert_eq!(mi.output_tokens.unwrap(), 1);
    assert!(mi.response_time_ms.unwrap() > 0);
    assert!(mi.ttft_ms.is_none());
    assert_eq!(
        mi.raw_response.as_deref().unwrap(),
        DUMMY_INFER_RESPONSE_RAW
    );
    assert_eq!(mi.raw_request.as_deref().unwrap(), DUMMY_RAW_REQUEST);
    assert_eq!(
        mi.system.as_deref().unwrap(),
        "You are a helpful and friendly assistant named AskJeeves"
    );
    let input_messages = mi.input_messages.as_ref().unwrap();
    assert_eq!(input_messages.len(), 1);
    assert_eq!(
        input_messages[0],
        StoredRequestMessage {
            role: Role::User,
            content: vec![StoredContentBlock::Text(Text {
                text: "I want 13 of tacos, please.".to_string(),
            })],
        }
    );
    let output = mi.output.as_ref().unwrap();
    assert_eq!(*output, vec![ContentBlockOutput::Text(Text {
        text: "Megumin gleefully chanted her spell, unleashing a thunderous explosion that lit up the sky and left a massive crater in its wake.".to_string(),
    })]);
}

#[tokio::test(flavor = "multi_thread")]
async fn test_variant_zero_weight_skip_zero() {
    let client = tensorzero::test_helpers::make_embedded_gateway().await;
    let error = client
        .inference(ClientInferenceParams {
            function_name: Some("variant_failover_zero_weight".to_string()),
            input: Input {
                system: Some(System::Template(Arguments(serde_json::Map::from_iter([(
                    "assistant_name".to_string(),
                    "AskJeeves".into(),
                )])))),
                messages: vec![InputMessage {
                    role: Role::User,
                    content: vec![InputMessageContent::Template(Template {
                        name: "user".to_string(),
                        arguments: Arguments(serde_json::Map::from_iter([
                            ("type".to_string(), "tacos".into()),
                            ("quantity".to_string(), 13.into()),
                        ])),
                    })],
                }],
            },
            ..Default::default()
        })
        .await
        .unwrap_err()
        .to_string();

    assert!(
        error.contains("Error sending request to Dummy provider for model 'error_1'"),
        "Missing 'error_1' in `{error}`"
    );
    assert!(
        error.contains("Error sending request to Dummy provider for model 'error_2'"),
        "Missing 'error_2' in `{error}`"
    );
    assert!(
        error.contains("Error sending request to Dummy provider for model 'error_no_weight'"),
        "Missing 'error_3' in `{error}`"
    );
    assert!(
        !error.contains("error_zero_weight"),
        "error_zero_weight should not have been called: `{error}`"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn test_variant_zero_weight_pin_zero() {
    let client = tensorzero::test_helpers::make_embedded_gateway().await;
    let error = client
        .inference(ClientInferenceParams {
            function_name: Some("variant_failover_zero_weight".to_string()),
            variant_name: Some("zero_weight".to_string()),
            input: Input {
                system: Some(System::Template(Arguments(serde_json::Map::from_iter([(
                    "assistant_name".to_string(),
                    "AskJeeves".into(),
                )])))),
                messages: vec![InputMessage {
                    role: Role::User,
                    content: vec![InputMessageContent::Template(Template {
                        name: "user".to_string(),
                        arguments: Arguments(serde_json::Map::from_iter([
                            ("type".to_string(), "tacos".into()),
                            ("quantity".to_string(), 13.into()),
                        ])),
                    })],
                }],
            },
            ..Default::default()
        })
        .await
        .unwrap_err()
        .to_string();

    assert!(
        error.contains("Error sending request to Dummy provider for model 'error_zero_weight'"),
        "Missing 'error_zero_weight' in `{error}`"
    );

    assert!(!error.contains("error_1"), "Found 'error_1' in `{error}`");
    assert!(!error.contains("error_2"), "Found 'error_2' in `{error}`");
    assert!(
        !error.contains("error_no_weight"),
        "Found 'error_no_weight' in `{error}`"
    );
}

/// This test checks that streaming inference works as expected.
#[tokio::test]
async fn test_streaming() {
    let episode_id = Uuid::now_v7();

    let payload = json!({
        "function_name": "basic_test",
        "episode_id": episode_id,
        "input":
            {
                "system": {
                    "assistant_name": "AskJeeves"
                },
                "messages": [
                {
                    "role": "user",
                    "content": "Hello, world!"
                }
            ]},
        "stream": true,
        "params": {
            "chat_completion": {
                "temperature": 2.0,
            "max_tokens": 200,
            "seed": 420
        }}
    });

    let mut event_source = Client::new()
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
    for (i, chunk) in chunks.iter().enumerate() {
        let chunk_json: Value = serde_json::from_str(chunk).unwrap();
        if i < DUMMY_STREAMING_RESPONSE.len() {
            let content = chunk_json.get("content").unwrap().as_array().unwrap();
            assert_eq!(content.len(), 1);
            let content_block = content.first().unwrap();
            let content_block_type = content_block.get("type").unwrap().as_str().unwrap();
            assert_eq!(content_block_type, "text");
            let content = content_block.get("text").unwrap().as_str().unwrap();
            assert_eq!(content, DUMMY_STREAMING_RESPONSE[i]);
        } else {
            assert!(
                chunk_json
                    .get("content")
                    .unwrap()
                    .as_array()
                    .unwrap()
                    .is_empty()
            );
            let usage = chunk_json.get("usage").unwrap().as_object().unwrap();
            let input_tokens = usage.get("input_tokens").unwrap().as_u64().unwrap();
            let output_tokens = usage.get("output_tokens").unwrap().as_u64().unwrap();
            assert_eq!(input_tokens, 10);
            assert_eq!(output_tokens, 16);
            inference_id = Some(
                Uuid::parse_str(chunk_json.get("inference_id").unwrap().as_str().unwrap()).unwrap(),
            );
        }
    }
    let inference_id = inference_id.unwrap();
    // Check database
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
    assert_eq!(inferences.len(), 1);
    let chat = match &inferences[0] {
        StoredInferenceDatabase::Chat(c) => c,
        StoredInferenceDatabase::Json(_) => panic!("Expected chat inference"),
    };
    assert_eq!(chat.inference_id, inference_id);
    let input = serde_json::to_value(&chat.input).unwrap();
    let correct_input: Value = json!(
        {
            "system": {
                "assistant_name": "AskJeeves"
            },
            "messages": [
            {
                "role": "user",
                "content": [{"type": "text", "text": "Hello, world!"}]
            }
        ]}
    );
    assert_eq!(input, correct_input);
    // Check content blocks
    let output = serde_json::to_value(&chat.output).unwrap();
    let content_blocks = output.as_array().unwrap();
    assert_eq!(content_blocks.len(), 1);
    let content_block = content_blocks.first().unwrap();
    let content_block_type = content_block.get("type").unwrap().as_str().unwrap();
    assert_eq!(content_block_type, "text");
    let content = content_block.get("text").unwrap().as_str().unwrap();
    assert_eq!(content, DUMMY_STREAMING_RESPONSE.join(""));
    assert_eq!(chat.episode_id, episode_id);
    // Check the variant name
    assert_eq!(chat.variant_name, "test");
    // Check the inference_params (set via payload)
    let inference_params = serde_json::to_value(&chat.inference_params).unwrap();
    let chat_completion_inference_params = inference_params
        .get("chat_completion")
        .unwrap()
        .as_object()
        .unwrap();
    let temperature = chat_completion_inference_params
        .get("temperature")
        .unwrap()
        .as_f64()
        .unwrap();
    assert_eq!(temperature, 2.0);
    let max_tokens = chat_completion_inference_params
        .get("max_tokens")
        .unwrap()
        .as_u64()
        .unwrap();
    assert_eq!(max_tokens, 200);
    let seed = chat_completion_inference_params
        .get("seed")
        .unwrap()
        .as_u64()
        .unwrap();
    assert_eq!(seed, 420);

    // Check the ModelInference Table
    let model_inferences = conn
        .get_model_inferences_by_inference_id(inference_id)
        .await
        .unwrap();
    assert_eq!(model_inferences.len(), 1);
    let mi = &model_inferences[0];
    assert_eq!(mi.input_tokens.unwrap(), 10);
    assert_eq!(mi.output_tokens.unwrap(), 16);
    let response_time_ms = mi.response_time_ms.unwrap();
    assert!(response_time_ms > 0);
    let ttft = mi.ttft_ms.unwrap();
    assert!(ttft > 0 && ttft <= response_time_ms);
    assert!(mi.raw_response.is_some());
    assert_eq!(mi.raw_request.as_deref().unwrap(), DUMMY_RAW_REQUEST);
    assert_eq!(
        mi.system.as_deref().unwrap(),
        "You are a helpful and friendly assistant named AskJeeves"
    );
    let input_messages = mi.input_messages.as_ref().unwrap();
    assert_eq!(input_messages.len(), 1);
    assert_eq!(
        input_messages[0],
        StoredRequestMessage {
            role: Role::User,
            content: vec![StoredContentBlock::Text(Text {
                text: "Hello, world!".to_string(),
            })],
        }
    );
    let output = mi.output.as_ref().unwrap();
    assert_eq!(
        *output,
        vec![ContentBlockOutput::Text(Text {
            text: DUMMY_STREAMING_RESPONSE.join("")
        })]
    );
}

/// This test checks that streaming inference works as expected when dryrun is true.
#[tokio::test]
async fn test_streaming_dryrun() {
    let payload = json!({
        "function_name": "basic_test",
        "episode_id": Uuid::now_v7(),
        "input":
            {
                "system": {
                    "assistant_name": "AskJeeves"
                },
                "messages": [
                {
                    "role": "user",
                    "content": "Hello, world!"
                }
            ]},
        "stream": true,
        "dryrun": true,
    });

    let mut event_source = Client::new()
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
    for (i, chunk) in chunks.iter().enumerate() {
        let chunk_json: Value = serde_json::from_str(chunk).unwrap();
        if i < DUMMY_STREAMING_RESPONSE.len() {
            let content = chunk_json.get("content").unwrap().as_array().unwrap();
            assert_eq!(content.len(), 1);
            let content_block = content.first().unwrap();
            let content_block_type = content_block.get("type").unwrap().as_str().unwrap();
            assert_eq!(content_block_type, "text");
            let content = content_block.get("text").unwrap().as_str().unwrap();
            assert_eq!(content, DUMMY_STREAMING_RESPONSE[i]);
        } else {
            assert!(
                chunk_json
                    .get("content")
                    .unwrap()
                    .as_array()
                    .unwrap()
                    .is_empty()
            );
            let usage = chunk_json.get("usage").unwrap().as_object().unwrap();
            let input_tokens = usage.get("input_tokens").unwrap().as_u64().unwrap();
            let output_tokens = usage.get("output_tokens").unwrap().as_u64().unwrap();
            assert_eq!(input_tokens, 10);
            assert_eq!(output_tokens, 16);
            inference_id = Some(
                Uuid::parse_str(chunk_json.get("inference_id").unwrap().as_str().unwrap()).unwrap(),
            );
        }
    }

    // Check that no inference was written to the database when dryrun is true
    let conn = DelegatingDatabaseConnection::new_for_e2e_test().await;
    conn.flush_pending_writes().await;
    conn.sleep_for_writes_to_be_visible().await;
    let config = get_e2e_config().await;
    let inferences = conn
        .list_inferences(
            &config,
            &ListInferencesParams {
                ids: Some(&[inference_id.unwrap()]),
                ..Default::default()
            },
        )
        .await
        .unwrap();
    assert!(
        inferences.is_empty(),
        "No inference should be written when dryrun is true"
    );
}

#[tokio::test]
async fn test_inference_original_response_non_stream() {
    let payload = json!({
        "function_name": "basic_test",
        "episode_id": Uuid::now_v7(),
        "input": {
            "system": {"assistant_name": "AskJeeves"},
            "messages": [
                {
                    "role": "user",
                    "content": "Hello, world!"
                }
            ]
        },
        "stream": false,
        "include_original_response": true,
    });

    let response = Client::new()
        .post(get_gateway_endpoint("/inference"))
        .json(&payload)
        .send()
        .await
        .unwrap();

    // Check Response is OK, then fields in order
    assert_eq!(response.status(), StatusCode::OK);
    let response_json = response.json::<Value>().await.unwrap();

    let original = &response_json["original_response"];
    assert_eq!(original, DUMMY_INFER_RESPONSE_RAW);
    // Don't both checking ClickHouse, as we do that in lots of other tests.
}

#[tokio::test]
async fn test_gateway_template_base_path() {
    let gateway = tensorzero::test_helpers::make_embedded_gateway_with_config(&format!(
        r#"
[functions.my_test]
type = "chat"
system_schema = "{root}/fixtures/config/functions/basic_test/system_schema.json"

[functions.my_test.variants.my_variant]
type = "chat_completion"
system_template = "{root}/fixtures/config/functions/basic_test/prompt/relative_system_template.minijinja"
model = "dummy::good"

[gateway.template_filesystem_access]
enabled = true
base_path = "{root}"
    "#,
        root = env!("CARGO_MANIFEST_DIR")
    ))
    .await;

    let params = ClientInferenceParams {
        function_name: Some("my_test".to_string()),
        input: Input {
            system: Some(System::Template(Arguments(serde_json::Map::from_iter([(
                "assistant_name".to_string(),
                "AskJeeves".into(),
            )])))),
            messages: vec![InputMessage {
                role: Role::User,
                content: vec![InputMessageContent::Text(Text {
                    text: "Please write me a sentence about Megumin making an explosion."
                        .to_string(),
                })],
            }],
        },
        include_original_response: true,
        ..Default::default()
    };

    // Request should be successful
    let response = gateway.inference(params.clone()).await.unwrap();
    let InferenceOutput::NonStreaming(InferenceResponse::Chat(_)) = response else {
        panic!("Expected non-streaming response, got {response:?}");
    };
}

#[tokio::test]
async fn test_gateway_template_no_fs_access() {
    // We use an embedded client so that we can control the number of
    // requests to the flaky judge.
    let gateway = tensorzero::test_helpers::make_embedded_gateway_with_config(&format!(
        r#"
[functions.my_test]
type = "chat"
system_schema = "{root}/fixtures/config/functions/basic_test/system_schema.json"

[functions.my_test.variants.my_variant]
type = "chat_completion"
system_template = "{root}/fixtures/config/functions/basic_test/prompt/system_template.minijinja"
model = "dummy::good"
    "#,
        root = env!("CARGO_MANIFEST_DIR")
    ))
    .await;

    let params = ClientInferenceParams {
        function_name: Some("my_test".to_string()),
        input: Input {
            system: Some(System::Template(Arguments(serde_json::Map::from_iter([(
                "assistant_name".to_string(),
                "AskJeeves".into(),
            )])))),
            messages: vec![InputMessage {
                role: Role::User,
                content: vec![InputMessageContent::Text(Text {
                    text: "Please write me a sentence about Megumin making an explosion."
                        .to_string(),
                })],
            }],
        },
        include_original_response: true,
        ..Default::default()
    };

    // Request should fail due to template trying to include from fs
    let err = gateway
        .inference(params.clone())
        .await
        .unwrap_err()
        .to_string();
    assert!(
        err.contains("Could not load template 'extra_templates/foo.minijinja' - if this a dynamic template included from the filesystem, please set `gateway.template_filesystem_access.enabled` to `true`"),
        "Unexpected error: {err}"
    );
}

#[tokio::test]
async fn test_original_response_best_of_n_flaky_judge() {
    // We use an embedded client so that we can control the number of
    // requests to the flaky judge.
    let gateway = tensorzero::test_helpers::make_embedded_gateway_with_config(
        r#"
[functions.best_of_n]
type = "chat"
[functions.best_of_n.variants.variant0]
type = "chat_completion"
weight = 0
model = "test"
[functions.best_of_n.variants.variant1]
type = "chat_completion"
weight = 0
model = "json"
[functions.best_of_n.variants.flaky_best_of_n_variant]
type = "experimental_best_of_n_sampling"
weight = 1
candidates = ["variant0", "variant1"]
[functions.best_of_n.variants.flaky_best_of_n_variant.evaluator]
model = "flaky_best_of_n_judge"
retries = { num_retries = 0, max_delay_s = 0 }
[models.flaky_best_of_n_judge]
routing = ["dummy"]
[models.flaky_best_of_n_judge.providers.dummy]
type = "dummy"
model_name = "flaky_best_of_n_judge"
[models.test]
routing = ["good"]
[models.test.providers.good]
type = "dummy"
model_name = "good"
[models.json]
routing = ["json"]
[models.json.providers.json]
type = "dummy"
model_name = "json"
    "#,
    )
    .await;

    let params = ClientInferenceParams {
        function_name: Some("best_of_n".to_string()),
        input: Input {
            messages: vec![InputMessage {
                role: Role::User,
                content: vec![InputMessageContent::Text(Text {
                    text: "Please write me a sentence about Megumin making an explosion."
                        .to_string(),
                })],
            }],
            ..Default::default()
        },
        include_original_response: true,
        ..Default::default()
    };

    // First request to the flaky judge should succeed
    let good_response = gateway.inference(params.clone()).await.unwrap();
    let InferenceOutput::NonStreaming(InferenceResponse::Chat(good_response)) = good_response
    else {
        panic!("Expected non-streaming response, got {good_response:?}");
    };

    assert_eq!(
        good_response.original_response.unwrap(),
        DUMMY_INFER_RESPONSE_RAW,
    );

    // Second request to the flaky judge should fail
    let bad_response = gateway.inference(params).await.unwrap();
    let InferenceOutput::NonStreaming(InferenceResponse::Chat(bad_response)) = bad_response else {
        panic!("Expected non-streaming response, got {bad_response:?}");
    };

    assert!(
        bad_response.original_response.is_none(),
        "Expected no original response"
    );
}

#[tokio::test]
async fn test_original_response_mixture_of_n_flaky_fuser() {
    let exporter = install_capturing_otel_exporter().await;
    // We use an embedded client so that we can control the number of
    // requests to the flaky judge.
    let gateway = tensorzero::test_helpers::make_embedded_gateway_with_config(
        r#"
[gateway.export.otlp.traces]
enabled = true

[functions.mixture_of_n]
type = "chat"
[functions.mixture_of_n.variants.variant0]
type = "chat_completion"
weight = 0
model = "dummy::test"
[functions.mixture_of_n.variants.variant1]
type = "chat_completion"
weight = 0
model = "dummy::alternate"
[functions.mixture_of_n.variants.mixture_of_n_variant]
type = "experimental_mixture_of_n"
weight = 1
candidates = ["variant0", "variant1"]
[functions.mixture_of_n.variants.mixture_of_n_variant.fuser]
model = "dummy::flaky_model"
    "#,
    )
    .await;

    let _guard = enter_fake_http_request_otel();

    let params = ClientInferenceParams {
        function_name: Some("mixture_of_n".to_string()),
        input: Input {
            messages: vec![InputMessage {
                role: Role::User,
                content: vec![InputMessageContent::Text(Text {
                    text: "Please write me a sentence about Megumin making an explosion."
                        .to_string(),
                })],
            }],
            ..Default::default()
        },
        include_original_response: true,
        ..Default::default()
    };

    // First request to the flaky judge should succeed
    let good_response = gateway.inference(params.clone()).await.unwrap();
    let InferenceOutput::NonStreaming(InferenceResponse::Chat(good_response)) = good_response
    else {
        panic!("Expected non-streaming response, got {good_response:?}");
    };

    assert_eq!(
        good_response.original_response.as_ref().unwrap(),
        DUMMY_INFER_RESPONSE_RAW,
    );

    check_good_mixture_response(exporter, good_response);

    // Second request to the flaky judge should fail
    let bad_response = gateway.inference(params).await.unwrap();
    let InferenceOutput::NonStreaming(InferenceResponse::Chat(bad_response)) = bad_response else {
        panic!("Expected non-streaming response, got {bad_response:?}");
    };

    assert!(
        bad_response.original_response.is_none(),
        "Expected no original response"
    );

    // Don't check ClickHouse, as we do that in lots of other tests.
}

fn check_good_mixture_response(exporter: CapturingOtelExporter, output: ChatInferenceResponse) {
    let all_spans = exporter.take_spans();
    let num_spans = all_spans.len();
    let spans = build_span_map(all_spans);
    let [root_span] = spans.root_spans.as_slice() else {
        panic!("Expected one root span: {:#?}", spans.root_spans);
    };
    // Since we're using the embedded gateway, the root span will be `function_call`
    // (we won't have a top-level HTTP span)
    assert_eq!(root_span.name, "function_inference");
    let root_attr_map = attrs_to_map(&root_span.attributes);
    assert_eq!(root_attr_map["function_name"], "mixture_of_n".into());
    assert_eq!(root_attr_map.get("model_name"), None);
    assert_eq!(
        root_attr_map["inference_id"],
        output.inference_id.to_string().into()
    );
    assert_eq!(
        root_attr_map["episode_id"],
        output.episode_id.to_string().into()
    );
    assert_eq!(root_attr_map["function_name"], "mixture_of_n".into());
    // We didn't explicitly pin a variant in the inference request, so this is `None` in the top-level function_call span
    assert_eq!(root_attr_map.get("variant_name"), None);

    let root_children = &spans.span_children[&root_span.span_context.span_id()];
    let (variant_span, write_inference_span) = {
        let mut children: Vec<_> = root_children.iter().collect();
        children.sort_by_key(|s| s.name.as_ref());
        match children.as_slice() {
            [variant, write] => (*variant, *write),
            _ => panic!(
                "Expected two child spans (variant_inference, write_inference): {root_children:#?}"
            ),
        }
    };

    assert_eq!(variant_span.name, "variant_inference");
    assert_eq!(write_inference_span.name, "write_inference");
    let variant_attr_map = attrs_to_map(&variant_span.attributes);
    assert_eq!(variant_attr_map["function_name"], "mixture_of_n".into());
    assert_eq!(
        variant_attr_map["variant_name"],
        "mixture_of_n_variant".into()
    );
    assert_eq!(variant_attr_map["stream"], false.into());

    let mut variant_children = spans.span_children[&variant_span.span_context.span_id()].clone();
    variant_children.sort_by_key(|s| {
        s.attributes.iter().find_map(|k| {
            if k.key.as_str() == "variant_name" {
                Some(k.value.as_str().to_string())
            } else {
                None
            }
        })
    });

    let [model_span, variant0_span, variant1_span] = variant_children.as_slice() else {
        panic!("Expected three child spans: {variant_children:#?}");
    };

    check_dummy_model_span(model_span, &spans, "dummy::flaky_model", "flaky_model");

    assert_eq!(variant0_span.name, "variant_inference");
    let variant0_attr_map = attrs_to_map(&variant0_span.attributes);
    assert_eq!(variant0_attr_map["function_name"], "mixture_of_n".into());
    assert_eq!(variant0_attr_map["variant_name"], "variant0".into());
    assert_eq!(variant0_attr_map["stream"], false.into());

    let variant0_children = &spans.span_children[&variant0_span.span_context.span_id()];
    let [variant0_model_span] = variant0_children.as_slice() else {
        panic!("Expected one child span: {variant0_children:#?}");
    };
    check_dummy_model_span(variant0_model_span, &spans, "dummy::test", "test");

    assert_eq!(variant1_span.name, "variant_inference");
    let variant1_attr_map = attrs_to_map(&variant1_span.attributes);
    assert_eq!(variant1_attr_map["function_name"], "mixture_of_n".into());
    assert_eq!(variant1_attr_map["variant_name"], "variant1".into());
    assert_eq!(variant1_attr_map["stream"], false.into());

    let variant1_children = &spans.span_children[&variant1_span.span_context.span_id()];
    let [variant1_model_span] = variant1_children.as_slice() else {
        panic!("Expected one child span: {variant1_children:#?}");
    };
    check_dummy_model_span(variant1_model_span, &spans, "dummy::alternate", "alternate");

    assert_eq!(
        num_spans, 17,
        "Expected 17 spans (added write_inference span)"
    );
}

fn check_dummy_model_span(
    model_span: &SpanData,
    spans: &SpanMap,
    model_name: &str,
    genai_model_name: &str,
) {
    assert_eq!(model_span.name, "model_inference");
    let model_attr_map = attrs_to_map(&model_span.attributes);
    assert_eq!(model_attr_map["model_name"].as_str(), model_name);
    assert_eq!(model_attr_map["stream"], false.into());

    let model_children = &spans.span_children[&model_span.span_context.span_id()];
    let [model_provider_span] = model_children.as_slice() else {
        panic!("Expected one child span: {model_children:#?}");
    };
    assert_eq!(model_provider_span.name, "model_provider_inference");
    let model_provider_attr_map = attrs_to_map(&model_provider_span.attributes);
    assert_eq!(model_provider_attr_map["provider_name"], "dummy".into());
    assert_eq!(
        model_provider_attr_map["gen_ai.operation.name"],
        "chat".into()
    );
    assert_eq!(model_provider_attr_map["gen_ai.system"], "dummy".into());
    assert_eq!(
        model_provider_attr_map["gen_ai.request.model"].as_str(),
        genai_model_name
    );
    assert_eq!(model_attr_map["stream"], false.into());

    let rate_limiting_spans = spans
        .span_children
        .get(&model_provider_span.span_context.span_id())
        .unwrap();

    // Check that we have a 'consume' and 'return' span - we have much more extensive checks in the 'otel' tests.
    let [first_span, second_span] = rate_limiting_spans.as_slice() else {
        panic!("Expected two rate limiting spans: {rate_limiting_spans:#?}");
    };
    let names = HashSet::from([&*first_span.name, &*second_span.name]);
    assert_eq!(
        names,
        HashSet::from([
            "rate_limiting_consume_tickets",
            "rate_limiting_return_tickets"
        ])
    );
}

#[gtest]
#[tokio::test]
async fn test_tool_call_streaming() {
    let episode_id = Uuid::now_v7();

    let payload = json!({
        "function_name": "weather_helper",
        "episode_id": episode_id,
        "input":{
            "system": {"assistant_name": "AskJeeves"},
            "messages": [
                {
                    "role": "user",
                    "content": "Hi I'm visiting Brooklyn from Brazil. What's the weather?"
                }
            ]},
        "stream": true,
    });
    let mut event_source = Client::new()
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
    let mut id: Option<String> = None;

    for (i, chunk) in chunks.iter().enumerate() {
        let chunk_json: Value = serde_json::from_str(chunk).unwrap();
        if i < DUMMY_STREAMING_TOOL_RESPONSE.len() {
            let content = chunk_json.get("content").unwrap().as_array().unwrap();
            assert_eq!(content.len(), 1);
            let content_block = content.first().unwrap();
            let content_block_type = content_block.get("type").unwrap().as_str().unwrap();
            assert_eq!(content_block_type, "tool_call");
            let new_arguments = content_block
                .get("raw_arguments")
                .unwrap()
                .as_str()
                .unwrap();
            assert_eq!(new_arguments, DUMMY_STREAMING_TOOL_RESPONSE[i]);
            let new_id = content_block.get("id").unwrap().as_str().unwrap();
            if i == 0 {
                id = Some(new_id.to_string());
            } else {
                assert_eq!(id, Some(new_id.to_string()));
            }
            if i == 0 {
                assert!(content_block.get("raw_name").is_some());
            } else {
                assert!(
                    content_block
                        .get("raw_name")
                        .unwrap()
                        .as_str()
                        .unwrap()
                        .is_empty(),
                    "Expected empty raw_name in non-first block, got {content_block:#?}",
                );
            }
        } else {
            assert!(
                chunk_json
                    .get("content")
                    .unwrap()
                    .as_array()
                    .unwrap()
                    .is_empty()
            );
            let usage = chunk_json.get("usage").unwrap().as_object().unwrap();
            let input_tokens = usage.get("input_tokens").unwrap().as_u64().unwrap();
            let output_tokens = usage.get("output_tokens").unwrap().as_u64().unwrap();
            assert_eq!(input_tokens, 10);
            assert_eq!(output_tokens, 5);
            inference_id = Some(
                Uuid::parse_str(chunk_json.get("inference_id").unwrap().as_str().unwrap()).unwrap(),
            );
        }
    }
    let inference_id = inference_id.unwrap();
    // Check database
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
    assert_eq!(inferences.len(), 1);
    let chat = match &inferences[0] {
        StoredInferenceDatabase::Chat(c) => c,
        StoredInferenceDatabase::Json(_) => panic!("Expected chat inference"),
    };
    assert_eq!(chat.inference_id, inference_id);
    let input = serde_json::to_value(&chat.input).unwrap();
    let correct_input: Value = json!(
        {
            "system": {
                "assistant_name": "AskJeeves"
            },
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": "Hi I'm visiting Brooklyn from Brazil. What's the weather?"}]
                }
            ]
        }
    );
    assert_eq!(input, correct_input);
    // Check that content blocks are correct
    let output = serde_json::to_value(&chat.output).unwrap();
    let content_blocks = output.as_array().unwrap();
    assert_eq!(content_blocks.len(), 1);
    let content_block = content_blocks.first().unwrap();
    let content_block_type = content_block.get("type").unwrap().as_str().unwrap();
    assert_eq!(content_block_type, "tool_call");
    // Check that the tool call is correctly returned
    let raw_arguments = content_block
        .get("raw_arguments")
        .unwrap()
        .as_str()
        .unwrap();
    let raw_arguments: Value = serde_json::from_str(raw_arguments).unwrap();
    assert_eq!(raw_arguments, *DUMMY_TOOL_RESPONSE);
    let id = content_block.get("id").unwrap().as_str().unwrap();
    assert_eq!(id, "0");
    let raw_name = content_block.get("raw_name").unwrap().as_str().unwrap();
    assert_eq!(raw_name, "get_temperature");
    let name = content_block.get("name").unwrap().as_str().unwrap();
    assert_eq!(name, "get_temperature");
    let arguments = content_block.get("arguments").unwrap();
    assert_eq!(arguments, &*DUMMY_TOOL_RESPONSE,);
    // Check that episode_id is here and correct
    assert_eq!(chat.episode_id, episode_id);
    // Check the variant name
    assert_eq!(chat.variant_name, "variant");
    // Check the tool_params
    let tool_params = serde_json::to_value(&chat.tool_params).unwrap();
    expect_that!(
        &tool_params,
        partially(matches_json_literal!({
            "tool_choice": "auto",
            "parallel_tool_calls": null,
            "allowed_tools": {
                "tools": ["get_temperature"],
                "choice": "function_default"
            }
        }))
    );
    // Check the ModelInference Table
    let model_inferences = conn
        .get_model_inferences_by_inference_id(inference_id)
        .await
        .unwrap();
    assert_eq!(model_inferences.len(), 1);
    let mi = &model_inferences[0];
    let _ = mi.id;
    assert_eq!(mi.inference_id, inference_id);

    assert_eq!(mi.input_tokens.unwrap(), 10);
    assert_eq!(mi.output_tokens.unwrap(), 5);
    assert!(mi.response_time_ms.unwrap() > 0);
    assert!(mi.ttft_ms.unwrap() > 50);
    assert!(mi.raw_response.is_some());
    assert_eq!(mi.raw_request.as_deref().unwrap(), DUMMY_RAW_REQUEST);

    assert_eq!(
        mi.system.as_deref().unwrap(),
        "You are a helpful and friendly assistant named AskJeeves.\n\nPeople will ask you questions about the weather.\n\nIf asked about the weather, just respond with the tool call. Use the \"get_temperature\" tool.\n\nIf provided with a tool result, use it to respond to the user (e.g. \"The weather in New York is 55 degrees Fahrenheit.\")."
    );
    let input_messages = mi.input_messages.as_ref().unwrap();
    assert_eq!(input_messages.len(), 1);
    assert_eq!(
        input_messages[0],
        StoredRequestMessage {
            role: Role::User,
            content: vec![
                "Hi I'm visiting Brooklyn from Brazil. What's the weather?"
                    .to_string()
                    .into()
            ],
        }
    );
    let output = mi.output.as_ref().unwrap();
    assert_eq!(output.len(), 1);
    match &output[0] {
        ContentBlockOutput::ToolCall(tool_call) => {
            assert_eq!(tool_call.name, "get_temperature");
            assert_eq!(
                tool_call.arguments,
                "{\"location\":\"Brooklyn\",\"units\":\"celsius\"}"
            );
            assert_eq!(tool_call.id, "0");
        }
        _ => panic!("Expected a tool call block, got {:?}", output[0]),
    }
}

#[gtest]
#[tokio::test]
async fn test_tool_call_streaming_split_tool_name() {
    let episode_id = Uuid::now_v7();

    let payload = json!({
        "function_name": "weather_helper",
        "episode_id": episode_id,
        "input":{
            "system": {"assistant_name": "AskJeeves"},
            "messages": [
                {
                    "role": "user",
                    "content": "Hi I'm visiting Brooklyn from Brazil. What's the weather?"
                }
            ]},
        "stream": true,
        "variant_name": "split_tool_name",
    });
    let mut event_source = Client::new()
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
    let mut id: Option<String> = None;
    let mut accumulated_tool_name = String::new();

    for (i, chunk) in chunks.iter().enumerate() {
        let chunk_json: Value = serde_json::from_str(chunk).unwrap();
        if i < DUMMY_STREAMING_TOOL_RESPONSE.len() {
            let content = chunk_json.get("content").unwrap().as_array().unwrap();
            assert_eq!(content.len(), 1);
            let content_block = content.first().unwrap();
            let content_block_type = content_block.get("type").unwrap().as_str().unwrap();
            assert_eq!(content_block_type, "tool_call");
            let new_arguments = content_block
                .get("raw_arguments")
                .unwrap()
                .as_str()
                .unwrap();
            assert_eq!(new_arguments, DUMMY_STREAMING_TOOL_RESPONSE[i]);
            let new_id = content_block.get("id").unwrap().as_str().unwrap();
            if i == 0 {
                id = Some(new_id.to_string());
            } else {
                assert_eq!(id, Some(new_id.to_string()));
            }
            let raw_name = content_block.get("raw_name").unwrap().as_str().unwrap();
            accumulated_tool_name.push_str(raw_name);
        } else {
            assert!(
                chunk_json
                    .get("content")
                    .unwrap()
                    .as_array()
                    .unwrap()
                    .is_empty()
            );
            let usage = chunk_json.get("usage").unwrap().as_object().unwrap();
            let input_tokens = usage.get("input_tokens").unwrap().as_u64().unwrap();
            let output_tokens = usage.get("output_tokens").unwrap().as_u64().unwrap();
            assert_eq!(input_tokens, 10);
            assert_eq!(output_tokens, 5);
            inference_id = Some(
                Uuid::parse_str(chunk_json.get("inference_id").unwrap().as_str().unwrap()).unwrap(),
            );
        }
    }
    let inference_id = inference_id.unwrap();
    assert_eq!(accumulated_tool_name, "get_temperature");
    // Check database
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
    assert_eq!(inferences.len(), 1);
    let chat = match &inferences[0] {
        StoredInferenceDatabase::Chat(c) => c,
        StoredInferenceDatabase::Json(_) => panic!("Expected chat inference"),
    };
    assert_eq!(chat.inference_id, inference_id);
    let input = serde_json::to_value(&chat.input).unwrap();
    let correct_input: Value = json!(
        {
            "system": {
                "assistant_name": "AskJeeves"
            },
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": "Hi I'm visiting Brooklyn from Brazil. What's the weather?"}]
                }
            ]
        }
    );
    assert_eq!(input, correct_input);
    // Check that content blocks are correct
    let output = serde_json::to_value(&chat.output).unwrap();
    let content_blocks = output.as_array().unwrap();
    assert_eq!(content_blocks.len(), 1);
    let content_block = content_blocks.first().unwrap();
    let content_block_type = content_block.get("type").unwrap().as_str().unwrap();
    assert_eq!(content_block_type, "tool_call");
    // Check that the tool call is correctly returned
    let raw_arguments = content_block
        .get("raw_arguments")
        .unwrap()
        .as_str()
        .unwrap();
    let raw_arguments: Value = serde_json::from_str(raw_arguments).unwrap();
    assert_eq!(raw_arguments, *DUMMY_TOOL_RESPONSE);
    let id = content_block.get("id").unwrap().as_str().unwrap();
    assert_eq!(id, "0");
    let raw_name = content_block.get("raw_name").unwrap().as_str().unwrap();
    assert_eq!(raw_name, "get_temperature");
    let name = content_block.get("name").unwrap().as_str().unwrap();
    assert_eq!(name, "get_temperature");
    let arguments = content_block.get("arguments").unwrap();
    assert_eq!(arguments, &*DUMMY_TOOL_RESPONSE,);
    // Check that episode_id is here and correct
    assert_eq!(chat.episode_id, episode_id);
    // Check the variant name
    assert_eq!(chat.variant_name, "split_tool_name");
    // Check the tool_params
    let tool_params = serde_json::to_value(&chat.tool_params).unwrap();
    expect_that!(
        &tool_params,
        partially(matches_json_literal!({
            "tool_choice": "auto",
            "parallel_tool_calls": null,
            "allowed_tools": {
                "tools": ["get_temperature"],
                "choice": "function_default"
            }
        }))
    );
    // Check the ModelInference Table
    let model_inferences = conn
        .get_model_inferences_by_inference_id(inference_id)
        .await
        .unwrap();
    assert_eq!(model_inferences.len(), 1);
    let mi = &model_inferences[0];
    let _ = mi.id;
    assert_eq!(mi.inference_id, inference_id);

    assert_eq!(mi.input_tokens.unwrap(), 10);
    assert_eq!(mi.output_tokens.unwrap(), 5);
    assert!(mi.response_time_ms.unwrap() > 0);
    assert!(mi.ttft_ms.unwrap() > 50);
    assert!(mi.raw_response.is_some());
    assert_eq!(mi.raw_request.as_deref().unwrap(), DUMMY_RAW_REQUEST);

    assert_eq!(
        mi.system.as_deref().unwrap(),
        "You are a helpful and friendly assistant named AskJeeves.\n\nPeople will ask you questions about the weather.\n\nIf asked about the weather, just respond with the tool call. Use the \"get_temperature\" tool.\n\nIf provided with a tool result, use it to respond to the user (e.g. \"The weather in New York is 55 degrees Fahrenheit.\")."
    );
    let input_messages = mi.input_messages.as_ref().unwrap();
    assert_eq!(input_messages.len(), 1);
    assert_eq!(
        input_messages[0],
        StoredRequestMessage {
            role: Role::User,
            content: vec![
                "Hi I'm visiting Brooklyn from Brazil. What's the weather?"
                    .to_string()
                    .into()
            ],
        }
    );
    let output = mi.output.as_ref().unwrap();
    assert_eq!(output.len(), 1);
    match &output[0] {
        ContentBlockOutput::ToolCall(tool_call) => {
            assert_eq!(tool_call.name, "get_temperature");
            assert_eq!(
                tool_call.arguments,
                "{\"location\":\"Brooklyn\",\"units\":\"celsius\"}"
            );
            assert_eq!(tool_call.id, "0");
        }
        _ => panic!("Expected a tool call block, got {:?}", output[0]),
    }
}

#[tokio::test]
async fn test_raw_text_http_gateway() {
    test_raw_text(tensorzero::test_helpers::make_http_gateway().await).await;
}

#[tokio::test(flavor = "multi_thread")]
async fn test_raw_text_embedded_gateway() {
    test_raw_text(tensorzero::test_helpers::make_embedded_gateway().await).await;
}

pub async fn test_raw_text(client: tensorzero::Client) {
    let episode_id = Uuid::now_v7();

    let InferenceOutput::NonStreaming(res) = client
        .inference(ClientInferenceParams {
            episode_id: Some(episode_id),
            function_name: Some("json_success".to_string()),
            input: Input {
                system: Some(System::Template(Arguments(serde_json::Map::from_iter([(
                    "assistant_name".to_string(),
                    "Dr. Mehta".into(),
                )])))),
                messages: vec![InputMessage {
                    role: Role::User,
                    content: vec![InputMessageContent::RawText(RawText {
                        value: "This is not the normal input".into(),
                    })],
                }],
            },
            ..Default::default()
        })
        .await
        .unwrap()
    else {
        panic!("Expected non-streaming response");
    };

    let inference_id = res.inference_id();
    let episode_id_response = res.episode_id();
    assert_eq!(episode_id_response, episode_id);

    let variant_name = res.variant_name();
    assert_eq!(variant_name, "test");

    let InferenceResponse::Json(json_res) = res else {
        panic!("Expected JSON response");
    };

    let content = json_res.output.parsed.unwrap();
    assert_eq!(
        content,
        serde_json::json!({
            "answer": "Hello"
        })
    );

    // Check database
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
    assert_eq!(inferences.len(), 1);
    let json_inf = match &inferences[0] {
        StoredInferenceDatabase::Json(j) => j,
        StoredInferenceDatabase::Chat(_) => panic!("Expected JSON inference"),
    };
    assert_eq!(json_inf.inference_id, inference_id);
    assert_eq!(json_inf.function_name, "json_success");
    assert_eq!(json_inf.variant_name, "test");
    assert_eq!(json_inf.episode_id, episode_id);

    let input = serde_json::to_value(&json_inf.input).unwrap();
    let correct_input = json!({
        "system": {"assistant_name": "Dr. Mehta"},
        "messages": [
            {
                "role": "user",
                "content": [{"type": "raw_text", "value": "This is not the normal input"}]
            }
        ]
    });
    assert_eq!(input, correct_input);
    let output = serde_json::to_value(&json_inf.output).unwrap();
    let parsed = output.get("parsed").unwrap();
    assert_eq!(
        parsed,
        &serde_json::json!({
            "answer": "Hello"
        })
    );
    // It's not necessary to check ModelInference table given how many other places we do that
}

#[tokio::test]
pub async fn test_dynamic_api_key() {
    skip_for_postgres!();
    let episode_id = Uuid::now_v7();

    let payload = json!({
        "function_name": "basic_test",
        "variant_name": "test_dynamic_api_key",
        "episode_id": episode_id,
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
    });

    let response = Client::new()
        .post(get_gateway_endpoint("/inference"))
        .json(&payload)
        .send()
        .await
        .unwrap();
    // Check that the API response is an error since we didn't provide the right key
    assert_eq!(response.status(), StatusCode::INTERNAL_SERVER_ERROR);
    let response_json = response.json::<Value>().await.unwrap();
    let error_message = response_json.get("error").unwrap().as_str().unwrap();
    assert!(
        error_message.contains("API key missing for provider Dummy"),
        "Unexpected error message: {error_message}"
    );

    let payload = json!({
        "function_name": "basic_test",
        "variant_name": "test_dynamic_api_key",
        "episode_id": episode_id,
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
        "credentials": {
            "DUMMY_API_KEY": "good_key",
        }
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

    let inference_id = response_json.get("inference_id").unwrap().as_str().unwrap();
    let inference_id = Uuid::parse_str(inference_id).unwrap();

    let episode_id_response = response_json.get("episode_id").unwrap().as_str().unwrap();
    let episode_id_response = Uuid::parse_str(episode_id_response).unwrap();
    assert_eq!(episode_id_response, episode_id);

    let variant_name = response_json.get("variant_name").unwrap().as_str().unwrap();
    assert_eq!(variant_name, "test_dynamic_api_key");

    let content = response_json.get("content").unwrap().as_array().unwrap();
    assert_eq!(content.len(), 1);
    let content_block = content.first().unwrap();
    let content_block_type = content_block.get("type").unwrap().as_str().unwrap();
    assert_eq!(content_block_type, "text");
    let content = content_block.get("text").unwrap().as_str().unwrap();
    assert_eq!(content, DUMMY_INFER_RESPONSE_CONTENT);

    let usage = response_json.get("usage").unwrap();
    let input_tokens = usage.get("input_tokens").unwrap().as_u64().unwrap();
    assert!(input_tokens > 0);
    let output_tokens = usage.get("output_tokens").unwrap().as_u64().unwrap();
    assert!(output_tokens > 0);

    // Check database
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
    assert_eq!(inferences.len(), 1);
    let chat = match &inferences[0] {
        StoredInferenceDatabase::Chat(c) => c,
        StoredInferenceDatabase::Json(_) => panic!("Expected chat inference"),
    };
    assert_eq!(chat.inference_id, inference_id);
    assert_eq!(chat.function_name, "basic_test");
    assert_eq!(chat.variant_name, "test_dynamic_api_key");
    assert_eq!(chat.episode_id, episode_id);

    let input = serde_json::to_value(&chat.input).unwrap();
    let correct_input = json!({
        "system": {"assistant_name": "Dr. Mehta"},
        "messages": [
            {
                "role": "user",
                "content": [{"type": "text", "text": "What is the name of the capital city of Japan?"}]
            }
        ]
    });
    assert_eq!(input, correct_input);

    let output = serde_json::to_value(&chat.output).unwrap();
    let content_blocks = output.as_array().unwrap();
    assert_eq!(content_blocks.len(), 1);
    let content_block = content_blocks.first().unwrap();
    let content_block_type = content_block.get("type").unwrap().as_str().unwrap();
    assert_eq!(content_block_type, "text");
    let db_content = content_block.get("text").unwrap().as_str().unwrap();
    assert_eq!(db_content, content);

    assert!(chat.tool_params.is_none());

    let inference_params = serde_json::to_value(&chat.inference_params).unwrap();
    let inference_params = inference_params.get("chat_completion").unwrap();
    assert_eq!(
        inference_params
            .get("temperature")
            .unwrap()
            .as_f64()
            .unwrap(),
        1.0
    );
    assert_eq!(inference_params.get("seed").unwrap().as_u64().unwrap(), 69);
    assert_eq!(
        inference_params
            .get("max_tokens")
            .unwrap()
            .as_u64()
            .unwrap(),
        100
    );
    // It's not necessary to check ModelInference table given how many other places we do that
}

#[tokio::test]
async fn test_inference_invalid_params() {
    let episode_id = Uuid::now_v7();

    // Test with invalid params structure (including fake_variant_type)
    let invalid_payload = json!({
        "function_name": "basic_test",
        "episode_id": episode_id,
        "input":
            {
               "system": {"assistant_name": "Dr. Mehta"},
               "messages": [
                {
                    "role": "user",
                    "content": "What is the name of the capital city of Japan?"
                }
            ]},
        "params": {
            "chat_completion": {
                "temperature": 0.9,
                "seed": 1337,
                "max_tokens": 120,
                "top_p": 0.9,
                "presence_penalty": 0.1,
                "frequency_penalty": 0.2,
            },
            "fake_variant_type": {
                "temperature": 0.8,
                "seed": 7331,
                "max_tokens": 80,
                "top_p": 0.9,
                "presence_penalty": 0.1,
                "frequency_penalty": 0.2,
            }
        },
    });

    let response = Client::new()
        .post(get_gateway_endpoint("/inference"))
        .json(&invalid_payload)
        .send()
        .await
        .unwrap();

    // Should fail with 400 Bad Request
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);

    // Create valid payload by removing the fake_variant_type from params
    let mut valid_payload = invalid_payload.clone();
    if let Some(params) = valid_payload
        .get_mut("params")
        .and_then(|p| p.as_object_mut())
    {
        params.remove("fake_variant_type");
    }

    let response = Client::new()
        .post(get_gateway_endpoint("/inference"))
        .json(&valid_payload)
        .send()
        .await
        .unwrap();

    // Should succeed with 200 OK
    assert_eq!(response.status(), StatusCode::OK);
}

#[tokio::test]
async fn test_dummy_only_embedded_gateway_no_config() {
    let client = tensorzero::test_helpers::make_embedded_gateway_no_config().await;
    let response = client
        .inference(ClientInferenceParams {
            model_name: Some("dummy::my-model".to_string()),
            input: Input {
                system: None,
                messages: vec![InputMessage {
                    role: Role::User,
                    content: vec![InputMessageContent::Text(Text {
                        text: "What is the name of the capital city of Japan?".to_string(),
                    })],
                }],
            },
            ..Default::default()
        })
        .await
        .unwrap();
    let InferenceOutput::NonStreaming(response) = response else {
        panic!("Expected non-streaming response");
    };
    let InferenceResponse::Chat(chat_response) = &response else {
        panic!("Expected chat response");
    };
    assert!(!chat_response.inference_id.is_nil());
    assert!(!chat_response.episode_id.is_nil());
    assert!(!chat_response.content.is_empty());
}

// test_dummy_only_replicated_clickhouse has been moved to inference_clickhouse.rs

#[tokio::test]
async fn test_dummy_only_inference_invalid_default_function_arg() {
    // We cannot provide both `function_name` and `model_name`
    let func_and_model = json!({
        "function_name": "basic_test",
        "model_name": "dummy::my-model",
        "input":
            {
               "system": {"assistant_name": "Dr. Mehta"},
               "messages": [
                {
                    "role": "user",
                    "content": "What is the name of the capital city of Japan?"
                }
            ]},
    });

    let response = Client::new()
        .post(get_gateway_endpoint("/inference"))
        .json(&func_and_model)
        .send()
        .await
        .unwrap();

    // Should fail with 400 Bad Request
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let response_text = response.text().await.unwrap();
    assert!(
        response_text.contains("Only one of `function_name` or `model_name` can be provided"),
        "Unexpected error message: {response_text}",
    );

    // We cannot provide both `function_name` and `model_name`
    let func_and_model = json!({
        "function_name": "basic_test",
        "model_name": "dummy::my-model",
        "input":
            {
               "system": {"assistant_name": "Dr. Mehta"},
               "messages": [
                {
                    "role": "user",
                    "content": "What is the name of the capital city of Japan?"
                }
            ]},
    });

    let response = Client::new()
        .post(get_gateway_endpoint("/inference"))
        .json(&func_and_model)
        .send()
        .await
        .unwrap();

    // Should fail with 400 Bad Request
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let response_text = response.text().await.unwrap();
    assert!(
        response_text.contains("Only one of `function_name` or `model_name` can be provided"),
        "Unexpected error message: {response_text}",
    );

    // We must provide `function_name` or `model_name`
    let neither_nor = json!({
        "input":
            {
               "system": {"assistant_name": "Dr. Mehta"},
               "messages": [
                {
                    "role": "user",
                    "content": "What is the name of the capital city of Japan?"
                }
            ]},
    });

    let response = Client::new()
        .post(get_gateway_endpoint("/inference"))
        .json(&neither_nor)
        .send()
        .await
        .unwrap();

    // Should fail with 400 Bad Request
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let response_text = response.text().await.unwrap();
    assert!(
        response_text.contains("Either `function_name` or `model_name` must be provided"),
        "Unexpected error message: {response_text}",
    );

    // We cannot specify both `model_name` and `variant_name`
    let model_and_variant = json!({
        "model_name": "dummy::my-model",
        "variant_name": "test_variant",
        "input":
            {
               "system": {"assistant_name": "Dr. Mehta"},
               "messages": [
                {
                    "role": "user",
                    "content": "What is the name of the capital city of Japan?"
                }
            ]},
    });

    let response = Client::new()
        .post(get_gateway_endpoint("/inference"))
        .json(&model_and_variant)
        .send()
        .await
        .unwrap();

    // Should fail with 400 Bad Request
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let response_text = response.text().await.unwrap();
    assert!(
        response_text.contains("`variant_name` cannot be provided when using `model_name`"),
        "Unexpected error message: {response_text}",
    );

    // We cannot specify a missing model name
    let missing_model = json!({
        "model_name": "missing_model",
        "input":
            {
               "messages": [
                {
                    "role": "user",
                    "content": "What is the name of the capital city of Japan?"
                }
            ]},
    });

    let response = Client::new()
        .post(get_gateway_endpoint("/inference"))
        .json(&missing_model)
        .send()
        .await
        .unwrap();

    // Should fail with 400 Bad Request
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let response_text = response.text().await.unwrap();
    assert!(
        response_text.contains("Model name 'missing_model' not found in model table"),
        "Unexpected error message: {response_text}",
    );

    // We cannot specify a system prompt
    let bad_system_prompt = json!({
        "model_name": "openai::tensorzero-invalid-model",
        "input":
            {
               "system": {"assistant_name": "Dr. Mehta"},
               "messages": [
                {
                    "role": "user",
                    "content": "What is the name of the capital city of Japan?"
                }
            ]},
    });

    let response = Client::new()
        .post(get_gateway_endpoint("/inference"))
        .json(&bad_system_prompt)
        .send()
        .await
        .unwrap();

    // Should fail with 400 Bad Request
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let response_text = response.text().await.unwrap();
    assert!(
        response_text.contains(
            "System message has non-string content but there is no template `system` in any variant"
        ),
        "Unexpected error message: {response_text}",
    );

    let bad_system_prompt_json_function = json!({
        "function_name": "null_json",
        "input":{
            "system": {"assistant_name": "Dr. Mehta"},
            "messages": [
                {
                    "role": "user",
                    "content": "What is the name of the capital city of Japan?"
                }
            ]},
    });

    let response = Client::new()
        .post(get_gateway_endpoint("/inference"))
        .json(&bad_system_prompt_json_function)
        .send()
        .await
        .unwrap();

    // Should fail with 400 Bad Request
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let response_text = response.text().await.unwrap();
    assert!(
        response_text.contains(
            "System message has non-string content but there is no template `system` in any variant"
        ),
        "Unexpected error message: {response_text}",
    );
}

#[tokio::test]

async fn test_image_inference_without_object_store() {
    let client = tensorzero::test_helpers::make_embedded_gateway_no_config().await;
    let err_msg = client
        .inference(ClientInferenceParams {
            model_name: Some("openai::gpt-4o-mini".to_string()),
            input: Input {
                system: None,
                messages: vec![InputMessage {
                    role: Role::User,
                    content: vec![
                        InputMessageContent::Text(Text {
                            text: "Describe the contents of the image".to_string(),
                        }),
                        InputMessageContent::File(File::Base64(
                            Base64File::new(
                                None,
                                Some(mime::IMAGE_PNG),
                                BASE64_STANDARD.encode(FERRIS_PNG),
                                None,
                                None,
                            )
                            .expect("test data should be valid"),
                        )),
                    ],
                }],
            },
            ..Default::default()
        })
        .await
        .unwrap_err()
        .to_string();
    assert!(
        err_msg.contains("Object storage is not configured"),
        "Unexpected error message: {err_msg}"
    );
}

async fn test_inference_zero_tokens_helper(
    model_name: &str,
    expected_input_tokens: Option<u64>,
    expected_output_tokens: Option<u64>,
) {
    let episode_id = Uuid::now_v7();

    let payload = json!({
        "model_name": model_name,
        "episode_id": episode_id,
        "input":{
            "messages": [
                {
                    "role": "user",
                    "content": "Hello, world!"
                }
            ]},
        "stream": false,
    });

    let response = Client::new()
        .post(get_gateway_endpoint("/inference"))
        .json(&payload)
        .send()
        .await
        .unwrap();
    // Check Response is OK, then fields in order
    assert_eq!(response.status(), StatusCode::OK);
    let response_json = response.json::<Value>().await.unwrap();
    // Check that inference_id is here
    let inference_id = response_json.get("inference_id").unwrap().as_str().unwrap();
    let inference_id = Uuid::parse_str(inference_id).unwrap();
    // Check that raw_content is same as content
    let content_blocks: &Vec<Value> = response_json.get("content").unwrap().as_array().unwrap();
    assert_eq!(content_blocks.len(), 1);
    let content_block = content_blocks.first().unwrap();
    let content_block_type = content_block.get("type").unwrap().as_str().unwrap();
    assert_eq!(content_block_type, "text");
    let content = content_block.get("text").unwrap().as_str().unwrap();
    assert_eq!(content, DUMMY_INFER_RESPONSE_CONTENT);

    // Check that usage is correct
    let usage = response_json.get("usage").unwrap();
    let usage = usage.as_object().unwrap();

    let input_tokens = usage.get("input_tokens").unwrap().as_u64().unwrap();
    let output_tokens = usage.get("output_tokens").unwrap().as_u64().unwrap();
    assert_eq!(input_tokens, expected_input_tokens.unwrap_or(0));
    assert_eq!(output_tokens, expected_output_tokens.unwrap_or(0));
    // Check database
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
    assert_eq!(inferences.len(), 1);
    let chat = match &inferences[0] {
        StoredInferenceDatabase::Chat(c) => c,
        StoredInferenceDatabase::Json(_) => panic!("Expected chat inference"),
    };
    assert_eq!(chat.inference_id, inference_id);
    let input = serde_json::to_value(&chat.input).unwrap();
    let correct_input: Value = json!(
        {
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": "Hello, world!"}]
                }
            ]
        }
    );
    assert_eq!(input, correct_input);
    // Check that content blocks are correct
    let output = serde_json::to_value(&chat.output).unwrap();
    let content_blocks = output.as_array().unwrap();
    assert_eq!(content_blocks.len(), 1);
    let content_block = content_blocks.first().unwrap();
    let content_block_type = content_block.get("type").unwrap().as_str().unwrap();
    assert_eq!(content_block_type, "text");
    let content = content_block.get("text").unwrap().as_str().unwrap();
    assert_eq!(content, DUMMY_INFER_RESPONSE_CONTENT);
    // Check that episode_id is here and correct
    assert_eq!(chat.episode_id, episode_id);
    // Check the variant name
    assert_eq!(chat.variant_name, model_name);

    // Check the ModelInference Table
    let model_inferences = conn
        .get_model_inferences_by_inference_id(inference_id)
        .await
        .unwrap();
    assert_eq!(model_inferences.len(), 1);
    let mi = &model_inferences[0];
    let _ = mi.id;
    assert_eq!(mi.inference_id, inference_id);

    assert_eq!(mi.input_tokens.map(u64::from), expected_input_tokens);
    assert_eq!(mi.output_tokens.map(u64::from), expected_output_tokens);
}

#[tokio::test]
async fn test_inference_input_tokens_zero() {
    test_inference_zero_tokens_helper("dummy::input_tokens_zero", None, Some(1)).await;
}

#[tokio::test]
async fn test_inference_output_tokens_zero() {
    test_inference_zero_tokens_helper("dummy::output_tokens_zero", Some(10), None).await;
}

#[tokio::test]
async fn test_inference_input_tokens_output_tokens_zero() {
    test_inference_zero_tokens_helper("dummy::input_tokens_output_tokens_zero", None, None).await;
}

#[tokio::test]
async fn test_tool_call_input_no_warning() {
    let logs_contain = tensorzero_core::utils::testing::capture_logs();
    let client = tensorzero::test_helpers::make_embedded_gateway_no_config().await;
    client
        .inference(ClientInferenceParams {
            model_name: Some("dummy::good".to_string()),
            input: Input {
                system: None,
                messages: vec![InputMessage {
                    role: Role::User,
                    content: vec![
                        InputMessageContent::Text(Text {
                            text: "Describe the contents of the image".to_string(),
                        }),
                        InputMessageContent::ToolCall(ToolCallWrapper::ToolCall(ToolCall {
                            id: "0".to_string(),
                            name: "get_temperature".to_string(),
                            arguments: json!({
                                "location": "Brooklyn",
                                "units": "celsius"
                            })
                            .to_string(),
                        })),
                    ],
                }],
            },
            ..Default::default()
        })
        .await
        .unwrap();
    assert!(!logs_contain("Deprecation"));
    assert!(!logs_contain("deprecation"));
    assert!(!logs_contain("Deprecated"));
    assert!(!logs_contain("deprecated"));
}

/// Test that a json inference with null response (i.e. no generated content blocks) works as expected.
#[tokio::test]
async fn test_chat_function_null_response() {
    let payload = json!({
        "function_name": "null_chat",
        "input": {
            "messages": [
                {
                    "role": "user",
                    "content": "No yapping!"
                }
            ]
        },
    });

    let response = Client::new()
        .post(get_gateway_endpoint("/inference"))
        .json(&payload)
        .send()
        .await
        .unwrap();

    // Check Response is OK, then fields in order
    assert_eq!(response.status(), StatusCode::OK);
    let response_json = response.json::<Value>().await.unwrap();

    // Check that the response content is an empty array (no content blocks)
    assert!(response_json["content"].as_array().unwrap().is_empty());
}

/// Test that a json inference with null response (i.e. no generated content blocks) works as expected.
#[tokio::test]
async fn test_json_function_null_response() {
    let payload = json!({
        "function_name": "null_json",
        "input": {
            "messages": [
                {
                    "role": "user",
                    "content": "Extract no data!"
                }
            ]
        },
    });

    let response = Client::new()
        .post(get_gateway_endpoint("/inference"))
        .json(&payload)
        .send()
        .await
        .unwrap();

    // Check Response is OK, then fields in order
    assert_eq!(response.status(), StatusCode::OK);
    let response_json = response.json::<Value>().await.unwrap();
    // Check that raw and parsed keys exist with null values
    assert!(
        response_json
            .get("output")
            .unwrap()
            .as_object()
            .unwrap()
            .contains_key("raw")
    );
    assert!(
        response_json
            .get("output")
            .unwrap()
            .as_object()
            .unwrap()
            .contains_key("parsed")
    );
    assert!(response_json["output"]["raw"].is_null());
    assert!(response_json["output"]["parsed"].is_null());
}

/// Test that a json inference with 2 text blocks in the message works as expected.
#[tokio::test]
async fn test_multiple_text_blocks_in_message() {
    let payload = json!({
        "model_name": "dummy::multiple-text-blocks",
        "input": {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "content"
                        },
                        {
                            "type": "text",
                            "text": "extra content"
                        }
                    ]
                },
            ]
        },
    });

    let response = Client::new()
        .post(get_gateway_endpoint("/inference"))
        .json(&payload)
        .send()
        .await
        .unwrap();

    // Check Response is OK, then fields in order
    assert_eq!(response.status(), StatusCode::OK);
    let response = response.json::<Value>().await.unwrap();
    let inference_id = response.get("inference_id").unwrap().as_str().unwrap();
    let inference_id = Uuid::parse_str(inference_id).unwrap();
    // Check database
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
    assert_eq!(inferences.len(), 1);
    let chat = match &inferences[0] {
        StoredInferenceDatabase::Chat(c) => c,
        StoredInferenceDatabase::Json(_) => panic!("Expected chat inference"),
    };

    // Check that the inference has multiple content blocks
    let input = chat.input.as_ref().unwrap();
    assert_eq!(input.messages.len(), 1);
    assert_eq!(input.messages[0].content.len(), 2);
    assert!(matches!(
        input.messages[0].content[0],
        StoredInputMessageContent::Text(_)
    ));
    assert!(matches!(
        input.messages[0].content[1],
        StoredInputMessageContent::Text(_)
    ));
}

#[tokio::test]
async fn test_internal_tag_auto_injection() {
    let client = Client::new();

    // Make an inference request with internal=true and a custom tag
    // We should NOT manually set tensorzero::internal - it should be auto-injected
    let payload = json!({
        "function_name": "basic_test",
        "input": {
            "system": {"assistant_name": "Alfred"},
            "messages": [{"role": "user", "content": "Hello!"}]
        },
        "internal": true,
        "tags": {
            "custom_tag": "custom_value"
        },
        "stream": false,
    });

    let response = client
        .post(get_gateway_endpoint("/inference"))
        .json(&payload)
        .send()
        .await
        .unwrap();

    assert!(response.status().is_success());
    let response_json = response.json::<Value>().await.unwrap();
    let inference_id = response_json.get("inference_id").unwrap().as_str().unwrap();
    let inference_id = Uuid::parse_str(inference_id).unwrap();

    // Check database to verify both tags are present
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
    assert_eq!(inferences.len(), 1);
    let chat = match &inferences[0] {
        StoredInferenceDatabase::Chat(c) => c,
        StoredInferenceDatabase::Json(_) => panic!("Expected chat inference"),
    };

    // Verify custom tag is present
    assert_eq!(
        chat.tags.get("custom_tag"),
        Some(&"custom_value".to_string())
    );

    // Verify auto-injected tensorzero::internal tag is present
    assert_eq!(
        chat.tags.get("tensorzero::internal"),
        Some(&"true".to_string())
    );
}
