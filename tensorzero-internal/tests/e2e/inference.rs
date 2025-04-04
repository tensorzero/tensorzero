use crate::providers::common::{make_embedded_gateway_with_config, FERRIS_PNG};
use axum::http::HeaderValue;
use base64::prelude::*;
use futures::StreamExt;
use reqwest::{Client, StatusCode};
use reqwest_eventsource::{Event, RequestBuilderExt};
use serde_json::{json, Value};
use tensorzero::{
    ClientBuilder, ClientBuilderMode, ClientInferenceParams, ClientInput, ClientInputMessage,
    ClientInputMessageContent, InferenceOutput, InferenceResponse,
};
use tensorzero_internal::{
    inference::{
        providers::dummy::{
            DUMMY_BAD_TOOL_RESPONSE, DUMMY_INFER_RESPONSE_CONTENT, DUMMY_INFER_RESPONSE_RAW,
            DUMMY_JSON_RESPONSE_RAW, DUMMY_RAW_REQUEST, DUMMY_STREAMING_RESPONSE,
            DUMMY_STREAMING_TOOL_RESPONSE, DUMMY_TOOL_RESPONSE,
        },
        types::{ContentBlock, Image, ImageKind, RequestMessage, Role, Text, TextKind},
    },
    tool::{ToolCall, ToolCallInput},
};
use tracing_test::traced_test;
use url::Url;
use uuid::Uuid;

use tensorzero_internal::clickhouse::test_helpers::{
    get_clickhouse, select_chat_inference_clickhouse, select_json_inference_clickhouse,
    select_model_inference_clickhouse,
};

use crate::{
    common::get_gateway_endpoint,
    providers::common::{
        make_embedded_gateway, make_embedded_gateway_no_config, make_http_gateway,
    },
};

#[tokio::test]
async fn e2e_test_inference_dryrun() {
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
    assert_eq!(response.status(), StatusCode::OK);
    let response_json = response.json::<Value>().await.unwrap();

    // Check that inference_id is here
    let inference_id = response_json.get("inference_id").unwrap().as_str().unwrap();
    let inference_id = Uuid::parse_str(inference_id).unwrap();

    // Sleep for 1 second to allow time for data to be inserted into ClickHouse (trailing writes from API)
    tokio::time::sleep(std::time::Duration::from_secs(1)).await;

    // Check ClickHouse
    let clickhouse = get_clickhouse().await;
    let result = select_chat_inference_clickhouse(&clickhouse, inference_id).await;
    assert!(result.is_none()); // No inference should be written to ClickHouse when dryrun is true
}

#[tokio::test]
async fn e2e_test_inference_chat_strip_unknown_block_non_stream() {
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
                        {"type": "unknown", "model_provider_name": "bad_model_provider", "data": {} },
                        {"type": "unknown", "model_provider_name": "tensorzero::model_name::test::provider_name::good", "data": {"my": "custom data"}},
                        {"type": "unknown", "model_provider_name": "tensorzero::model_name::test::provider_name::wrong_model_name", "data": {"SHOULD NOT": "SHOW UP"}},
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

    // Sleep for 100ms second to allow time for data to be inserted into ClickHouse (trailing writes from API)
    tokio::time::sleep(std::time::Duration::from_millis(100)).await;

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

    // The input in ChatInference should have *all* of the unknown blocks present
    let correct_input: Value = json!(
        {
            "system": {
                "assistant_name": "AskJeeves"
            },
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "text", "value": "Hello, world!"}]
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "unknown", "model_provider_name": "bad_model_provider", "data": {} },
                        {"type": "unknown", "model_provider_name": "tensorzero::model_name::test::provider_name::good", "data": {"my": "custom data"}},
                        {"type": "unknown", "model_provider_name": "tensorzero::model_name::test::provider_name::wrong_model_name", "data": {"SHOULD NOT": "SHOW UP"}},
                        {"type": "unknown", "model_provider_name": null, "data": "Non-provider-specific unknown block"}
                    ]
                }
            ]
        }
    );
    assert_eq!(input, correct_input);
    // Check that content blocks are correct
    let content_blocks = result.get("output").unwrap().as_str().unwrap();
    let content_blocks: Vec<Value> = serde_json::from_str(content_blocks).unwrap();
    assert_eq!(content_blocks.len(), 1);
    let content_block = content_blocks.first().unwrap();
    let content_block_type = content_block.get("type").unwrap().as_str().unwrap();
    assert_eq!(content_block_type, "text");
    let content = content_block.get("text").unwrap().as_str().unwrap();
    assert_eq!(content, DUMMY_INFER_RESPONSE_CONTENT);
    // Check that episode_id is here and correct
    let retrieved_episode_id = result.get("episode_id").unwrap().as_str().unwrap();
    let retrieved_episode_id = Uuid::parse_str(retrieved_episode_id).unwrap();
    assert_eq!(retrieved_episode_id, episode_id);
    // Check the variant name
    let variant_name = result.get("variant_name").unwrap().as_str().unwrap();
    assert_eq!(variant_name, "test");

    // Check the ModelInference Table
    let result = select_model_inference_clickhouse(&clickhouse, inference_id)
        .await
        .unwrap();
    let id = result.get("id").unwrap().as_str().unwrap();
    let _ = Uuid::parse_str(id).unwrap();
    let inference_id_result = result.get("inference_id").unwrap().as_str().unwrap();
    let inference_id_result = Uuid::parse_str(inference_id_result).unwrap();
    assert_eq!(inference_id_result, inference_id);

    let input_messages = result.get("input_messages").unwrap().as_str().unwrap();
    let input_messages: Vec<RequestMessage> = serde_json::from_str(input_messages).unwrap();
    assert_eq!(
        input_messages,
        vec![
            RequestMessage {
                role: Role::User,
                content: vec![ContentBlock::Text(Text {
                    text: "Hello, world!".to_string(),
                })]
            },
            RequestMessage {
                role: Role::User,
                content: vec![
                    ContentBlock::Unknown {
                        model_provider_name: Some(
                            "tensorzero::model_name::test::provider_name::good".to_string()
                        ),
                        data: json!({"my": "custom data"})
                    },
                    ContentBlock::Unknown {
                        model_provider_name: None,
                        data: "Non-provider-specific unknown block".into()
                    }
                ]
            },
        ]
    );
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
                        {"type": "unknown", "model_provider_name": "bad_model_provider", "data": {} },
                        {"type": "unknown", "model_provider_name": "tensorzero::model_name::test::provider_name::good", "data": {"my": "custom data"}},
                        {"type": "unknown", "model_provider_name": "tensorzero::model_name::test::provider_name::wrong_model_name", "data": {"SHOULD NOT": "SHOW UP"}},
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

    // Sleep for 100ms second to allow time for data to be inserted into ClickHouse (trailing writes from API)
    tokio::time::sleep(std::time::Duration::from_millis(100)).await;

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

    // The input in ChatInference should have *all* of the unknown blocks present
    let correct_input: Value = json!(
        {
            "system": {
                "assistant_name": "AskJeeves"
            },
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "text", "value": "Hello, world!"}]
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "unknown", "model_provider_name": "bad_model_provider", "data": {} },
                        {"type": "unknown", "model_provider_name": "tensorzero::model_name::test::provider_name::good", "data": {"my": "custom data"}},
                        {"type": "unknown", "model_provider_name": "tensorzero::model_name::test::provider_name::wrong_model_name", "data": {"SHOULD NOT": "SHOW UP"}},
                        {"type": "unknown", "model_provider_name": null, "data": "Non-provider-specific unknown block"}
                    ]
                }
            ]
        }
    );
    assert_eq!(input, correct_input);
    // Check that content blocks are correct
    let content_blocks = result.get("output").unwrap().as_str().unwrap();
    let content_blocks: Vec<Value> = serde_json::from_str(content_blocks).unwrap();
    assert_eq!(content_blocks.len(), 1);
    let content_block = content_blocks.first().unwrap();
    let content_block_type = content_block.get("type").unwrap().as_str().unwrap();
    assert_eq!(content_block_type, "text");
    let content = content_block.get("text").unwrap().as_str().unwrap();
    assert_eq!(content, "Wally, the golden retriever, wagged his tail excitedly as he devoured a slice of cheese pizza.");
    // Check that episode_id is here and correct
    let retrieved_episode_id = result.get("episode_id").unwrap().as_str().unwrap();
    let retrieved_episode_id = Uuid::parse_str(retrieved_episode_id).unwrap();
    assert_eq!(retrieved_episode_id, episode_id);
    // Check the variant name
    let variant_name = result.get("variant_name").unwrap().as_str().unwrap();
    assert_eq!(variant_name, "test");

    // Check the ModelInference Table
    let result = select_model_inference_clickhouse(&clickhouse, inference_id)
        .await
        .unwrap();
    let id = result.get("id").unwrap().as_str().unwrap();
    let _ = Uuid::parse_str(id).unwrap();
    let inference_id_result = result.get("inference_id").unwrap().as_str().unwrap();
    let inference_id_result = Uuid::parse_str(inference_id_result).unwrap();
    assert_eq!(inference_id_result, inference_id);

    let input_messages = result.get("input_messages").unwrap().as_str().unwrap();
    let input_messages: Vec<RequestMessage> = serde_json::from_str(input_messages).unwrap();
    assert_eq!(
        input_messages,
        vec![
            RequestMessage {
                role: Role::User,
                content: vec![ContentBlock::Text(Text {
                    text: "Hello, world!".to_string(),
                })]
            },
            RequestMessage {
                role: Role::User,
                content: vec![
                    ContentBlock::Unknown {
                        model_provider_name: Some(
                            "tensorzero::model_name::test::provider_name::good".to_string()
                        ),
                        data: json!({"my": "custom data"})
                    },
                    ContentBlock::Unknown {
                        model_provider_name: None,
                        data: "Non-provider-specific unknown block".into()
                    }
                ]
            },
        ]
    );
}

/// This test calls a function which calls a model where the first provider is broken but
/// then the second provider works fine. We expect this request to work despite the first provider
/// being broken.
#[tokio::test]
async fn e2e_test_inference_model_fallback() {
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
    assert_eq!(output_tokens, 10);
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
            "system": {
                "assistant_name": "AskJeeves"
            },
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "text", "value": "Hello, world!"}]
                }
            ]
        }
    );
    assert_eq!(input, correct_input);
    // Check that content blocks are correct
    let content_blocks = result.get("output").unwrap().as_str().unwrap();
    let content_blocks: Vec<Value> = serde_json::from_str(content_blocks).unwrap();
    assert_eq!(content_blocks.len(), 1);
    let content_block = content_blocks.first().unwrap();
    let content_block_type = content_block.get("type").unwrap().as_str().unwrap();
    assert_eq!(content_block_type, "text");
    let content = content_block.get("text").unwrap().as_str().unwrap();
    assert_eq!(content, DUMMY_INFER_RESPONSE_CONTENT);
    // Check that episode_id is here and correct
    let retrieved_episode_id = result.get("episode_id").unwrap().as_str().unwrap();
    let retrieved_episode_id = Uuid::parse_str(retrieved_episode_id).unwrap();
    assert_eq!(retrieved_episode_id, episode_id);
    // Check the variant name
    let variant_name = result.get("variant_name").unwrap().as_str().unwrap();
    assert_eq!(variant_name, "test");

    // Check the ModelInference Table
    let result = select_model_inference_clickhouse(&clickhouse, inference_id)
        .await
        .unwrap();
    let id = result.get("id").unwrap().as_str().unwrap();
    let _ = Uuid::parse_str(id).unwrap();
    let inference_id_result = result.get("inference_id").unwrap().as_str().unwrap();
    let inference_id_result = Uuid::parse_str(inference_id_result).unwrap();
    assert_eq!(inference_id_result, inference_id);

    let input_tokens = result.get("input_tokens").unwrap().as_u64().unwrap();
    assert_eq!(input_tokens, 10);
    let output_tokens = result.get("output_tokens").unwrap().as_u64().unwrap();
    assert_eq!(output_tokens, 10);
    let response_time_ms = result.get("response_time_ms").unwrap().as_u64().unwrap();
    assert!(response_time_ms > 0);
    assert!(result.get("ttft_ms").unwrap().is_null());
    assert_eq!(
        result.get("raw_response").unwrap().as_str().unwrap(),
        DUMMY_INFER_RESPONSE_RAW
    );
    assert_eq!(
        result.get("raw_request").unwrap().as_str().unwrap(),
        DUMMY_RAW_REQUEST
    );
    let system = result.get("system").unwrap().as_str().unwrap();
    assert_eq!(
        system,
        "You are a helpful and friendly assistant named AskJeeves"
    );
    let input_messages = result.get("input_messages").unwrap().as_str().unwrap();
    let input_messages: Vec<RequestMessage> = serde_json::from_str(input_messages).unwrap();
    assert_eq!(input_messages.len(), 1);
    assert_eq!(
        input_messages[0],
        RequestMessage {
            role: Role::User,
            content: vec!["Hello, world!".to_string().into()],
        }
    );
    let output = result.get("output").unwrap().as_str().unwrap();
    let output: Vec<ContentBlock> = serde_json::from_str(output).unwrap();
    assert_eq!(
        output,
        vec![DUMMY_INFER_RESPONSE_CONTENT.to_string().into()]
    );
}

#[tokio::test]
async fn e2e_test_tool_call() {
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
    assert_eq!(output_tokens, 10);
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
            "system": {
                "assistant_name": "AskJeeves"
            },
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "text", "value": "Hi I'm visiting Brooklyn from Brazil. What's the weather?"}]
                }
            ]
        }
    );
    assert_eq!(input, correct_input);
    // Check that content blocks are correct
    let content_blocks = result.get("output").unwrap().as_str().unwrap();
    let content_blocks: Vec<Value> = serde_json::from_str(content_blocks).unwrap();
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
    let retrieved_episode_id = result.get("episode_id").unwrap().as_str().unwrap();
    let retrieved_episode_id = Uuid::parse_str(retrieved_episode_id).unwrap();
    assert_eq!(retrieved_episode_id, episode_id);
    // Check the variant name
    let variant_name = result.get("variant_name").unwrap().as_str().unwrap();
    assert_eq!(variant_name, "variant");
    // Check the tool_params
    let tool_params = result.get("tool_params").unwrap().as_str().unwrap();
    let tool_params: Value = serde_json::from_str(tool_params).unwrap();
    let tools_available = tool_params
        .get("tools_available")
        .unwrap()
        .as_array()
        .unwrap();
    assert!(tools_available.len() == 1);
    assert!(
        tools_available
            .first()
            .unwrap()
            .get("name")
            .unwrap()
            .as_str()
            .unwrap()
            == "get_temperature"
    );
    assert!(tool_params.get("tool_choice").unwrap().as_str().unwrap() == "auto");
    assert_eq!(
        tool_params.get("parallel_tool_calls").unwrap(),
        &Value::Null
    );
    // Check the ModelInference Table
    let result = select_model_inference_clickhouse(&clickhouse, inference_id)
        .await
        .unwrap();
    let id = result.get("id").unwrap().as_str().unwrap();
    let _ = Uuid::parse_str(id).unwrap();
    let inference_id_result = result.get("inference_id").unwrap().as_str().unwrap();
    let inference_id_result = Uuid::parse_str(inference_id_result).unwrap();
    assert_eq!(inference_id_result, inference_id);

    let input_tokens = result.get("input_tokens").unwrap().as_u64().unwrap();
    assert_eq!(input_tokens, 10);
    let output_tokens = result.get("output_tokens").unwrap().as_u64().unwrap();
    assert_eq!(output_tokens, 10);
    let response_time_ms = result.get("response_time_ms").unwrap().as_u64().unwrap();
    assert!(response_time_ms > 0);
    assert!(result.get("ttft_ms").unwrap().is_null());
    assert_eq!(
        result.get("raw_response").unwrap().as_str().unwrap(),
        serde_json::to_string(&*DUMMY_TOOL_RESPONSE).unwrap()
    );
    assert_eq!(
        result.get("raw_request").unwrap().as_str().unwrap(),
        DUMMY_RAW_REQUEST
    );
    let system = result.get("system").unwrap().as_str().unwrap();
    let expected_system = "You are a helpful and friendly assistant named AskJeeves.\n\nPeople will ask you questions about the weather.\n\nIf asked about the weather, just respond with the tool call. Use the \"get_temperature\" tool.\n\nIf provided with a tool result, use it to respond to the user (e.g. \"The weather in New York is 55 degrees Fahrenheit.\").";
    assert_eq!(system, expected_system);
    let input_messages = result.get("input_messages").unwrap().as_str().unwrap();
    let input_messages: Vec<RequestMessage> = serde_json::from_str(input_messages).unwrap();
    let expected_messages = vec![RequestMessage {
        role: Role::User,
        content: vec!["Hi I'm visiting Brooklyn from Brazil. What's the weather?"
            .to_string()
            .into()],
    }];
    assert_eq!(input_messages, expected_messages);
    let output = result.get("output").unwrap().as_str().unwrap();
    let output: Vec<ContentBlock> = serde_json::from_str(output).unwrap();
    assert_eq!(output.len(), 1);
    match &output[0] {
        ContentBlock::ToolCall(tool_call) => {
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

#[tokio::test]
async fn e2e_test_tool_call_malformed() {
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
    assert_eq!(output_tokens, 10);
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
            "system": {
                "assistant_name": "AskJeeves"
            },
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "text", "value": "Hi I'm visiting Brooklyn from Brazil. What's the weather?"}]
                }
            ]
        }
    );
    assert_eq!(input, correct_input);
    // Check that content blocks are correct
    let content_blocks = result.get("output").unwrap().as_str().unwrap();
    let content_blocks: Vec<Value> = serde_json::from_str(content_blocks).unwrap();
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
    let retrieved_episode_id = result.get("episode_id").unwrap().as_str().unwrap();
    let retrieved_episode_id = Uuid::parse_str(retrieved_episode_id).unwrap();
    assert_eq!(retrieved_episode_id, episode_id);
    // Check the variant name
    let variant_name = result.get("variant_name").unwrap().as_str().unwrap();
    assert_eq!(variant_name, "bad_tool");
    // Check the tool_params
    let tool_params = result.get("tool_params").unwrap().as_str().unwrap();
    let tool_params: Value = serde_json::from_str(tool_params).unwrap();
    let tools_available = tool_params
        .get("tools_available")
        .unwrap()
        .as_array()
        .unwrap();
    assert!(tools_available.len() == 1);
    assert!(
        tools_available
            .first()
            .unwrap()
            .get("name")
            .unwrap()
            .as_str()
            .unwrap()
            == "get_temperature"
    );
    assert!(tool_params.get("tool_choice").unwrap().as_str().unwrap() == "auto");
    assert_eq!(
        tool_params.get("parallel_tool_calls").unwrap(),
        &Value::Null
    );
    // Check the ModelInference Table
    let result = select_model_inference_clickhouse(&clickhouse, inference_id)
        .await
        .unwrap();
    let id = result.get("id").unwrap().as_str().unwrap();
    let _ = Uuid::parse_str(id).unwrap();
    let inference_id_result = result.get("inference_id").unwrap().as_str().unwrap();
    let inference_id_result = Uuid::parse_str(inference_id_result).unwrap();
    assert_eq!(inference_id_result, inference_id);

    let input_tokens = result.get("input_tokens").unwrap().as_u64().unwrap();
    assert_eq!(input_tokens, 10);
    let output_tokens = result.get("output_tokens").unwrap().as_u64().unwrap();
    assert_eq!(output_tokens, 10);
    let response_time_ms = result.get("response_time_ms").unwrap().as_u64().unwrap();
    assert!(response_time_ms > 0);
    assert!(result.get("ttft_ms").unwrap().is_null());
    result.get("raw_response").unwrap().as_str().unwrap();
    assert_eq!(
        result.get("raw_request").unwrap().as_str().unwrap(),
        DUMMY_RAW_REQUEST
    );
    let system = result.get("system").unwrap().as_str().unwrap();
    let expected_system = "You are a helpful and friendly assistant named AskJeeves.\n\nPeople will ask you questions about the weather.\n\nIf asked about the weather, just respond with the tool call. Use the \"get_temperature\" tool.\n\nIf provided with a tool result, use it to respond to the user (e.g. \"The weather in New York is 55 degrees Fahrenheit.\").";
    assert_eq!(system, expected_system);
    let input_messages = result.get("input_messages").unwrap().as_str().unwrap();
    let input_messages: Vec<RequestMessage> = serde_json::from_str(input_messages).unwrap();
    let expected_messages = vec![RequestMessage {
        role: Role::User,
        content: vec!["Hi I'm visiting Brooklyn from Brazil. What's the weather?"
            .to_string()
            .into()],
    }];
    assert_eq!(input_messages, expected_messages);
    let output = result.get("output").unwrap().as_str().unwrap();
    let output: Vec<ContentBlock> = serde_json::from_str(output).unwrap();
    assert_eq!(output.len(), 1);
    match &output[0] {
        ContentBlock::ToolCall(tool_call) => {
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
async fn e2e_test_inference_json_fail() {
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
    assert_eq!(output_tokens, 10);
    // Sleep for 1 second to allow time for data to be inserted into ClickHouse (trailing writes from API)
    tokio::time::sleep(std::time::Duration::from_secs(1)).await;

    // Check ClickHouse
    let clickhouse = get_clickhouse().await;
    let result = select_json_inference_clickhouse(&clickhouse, inference_id)
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
                "content": [{"type": "text", "value": "Hello, world!"}]
            }
        ]
    });
    assert_eq!(input, correct_input);
    let retrieved_episode_id = result.get("episode_id").unwrap().as_str().unwrap();
    let retrieved_episode_id = Uuid::parse_str(retrieved_episode_id).unwrap();
    assert_eq!(retrieved_episode_id, episode_id);
    let output: Value =
        serde_json::from_str(result.get("output").unwrap().as_str().unwrap()).unwrap();
    assert!(output.get("parsed").unwrap().is_null());
    assert!(output.get("raw").unwrap().as_str().unwrap() == DUMMY_INFER_RESPONSE_CONTENT);
    // Check the variant name
    let variant_name = result.get("variant_name").unwrap().as_str().unwrap();
    assert_eq!(variant_name, "test");

    // Check the ModelInference Table
    let result = select_model_inference_clickhouse(&clickhouse, inference_id)
        .await
        .unwrap();
    let id = result.get("id").unwrap().as_str().unwrap();
    let _ = Uuid::parse_str(id).unwrap();
    let inference_id_result = result.get("inference_id").unwrap().as_str().unwrap();
    let inference_id_result = Uuid::parse_str(inference_id_result).unwrap();
    assert_eq!(inference_id_result, inference_id);

    let input_tokens = result.get("input_tokens").unwrap().as_u64().unwrap();
    assert_eq!(input_tokens, 10);
    let output_tokens = result.get("output_tokens").unwrap().as_u64().unwrap();
    assert_eq!(output_tokens, 10);
    let response_time_ms = result.get("response_time_ms").unwrap().as_u64().unwrap();
    assert!(response_time_ms > 0);
    assert!(result.get("ttft_ms").unwrap().is_null());
    assert_eq!(
        result.get("raw_response").unwrap().as_str().unwrap(),
        DUMMY_INFER_RESPONSE_RAW
    );
    assert_eq!(
        result.get("raw_request").unwrap().as_str().unwrap(),
        DUMMY_RAW_REQUEST
    );
    let system = result.get("system").unwrap().as_str().unwrap();
    assert_eq!(
        system,
        "You are a helpful and friendly assistant named AskJeeves"
    );
    let input_messages = result.get("input_messages").unwrap().as_str().unwrap();
    let input_messages: Vec<RequestMessage> = serde_json::from_str(input_messages).unwrap();
    assert_eq!(input_messages.len(), 1);
    assert_eq!(
        input_messages[0],
        RequestMessage {
            role: Role::User,
            content: vec!["Hello, world!".to_string().into()],
        }
    );
    let output = result.get("output").unwrap().as_str().unwrap();
    let output: Vec<ContentBlock> = serde_json::from_str(output).unwrap();
    assert_eq!(
        output,
        vec![DUMMY_INFER_RESPONSE_CONTENT.to_string().into()]
    );
}

/// This test checks the return type and clickhouse writes for a function with an output schema and
/// a response which satisfies the schema.
/// We expect to see a filled-out `content` field in the response and a filled-out `output` field in the table.
#[tokio::test]
async fn e2e_test_inference_json_success() {
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
                    "content": [{"type": "text", "arguments": {"country": "Japan"}}]
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
    assert_eq!(output_tokens, 10);
    // Sleep for 1 second to allow time for data to be inserted into ClickHouse (trailing writes from API)
    tokio::time::sleep(std::time::Duration::from_secs(1)).await;

    // Check ClickHouse
    let clickhouse = get_clickhouse().await;
    let result = select_json_inference_clickhouse(&clickhouse, inference_id)
        .await
        .unwrap();
    let id = result.get("id").unwrap().as_str().unwrap();
    let id_uuid = Uuid::parse_str(id).unwrap();
    assert_eq!(id_uuid, inference_id);
    let correct_input = json!({
        "system": {"assistant_name": "AskJeeves"},
        "messages": [
            {
                "role": "user",
                "content": [{"type": "text", "value": {"country": "Japan"}}]
            }
        ]
    });
    let input: Value =
        serde_json::from_str(result.get("input").unwrap().as_str().unwrap()).unwrap();
    assert_eq!(input, correct_input);
    // Check that correctly parsed output is present
    let output = result.get("output").unwrap().as_str().unwrap();
    let output: Value = serde_json::from_str(output).unwrap();
    let parsed = output.get("parsed").unwrap().as_object().unwrap();
    let answer = parsed.get("answer").unwrap().as_str().unwrap();
    assert_eq!(answer, "Hello");
    let raw = output.get("raw").unwrap().as_str().unwrap();
    assert_eq!(raw, DUMMY_JSON_RESPONSE_RAW);
    // Check content blocks
    let retrieved_episode_id = result.get("episode_id").unwrap().as_str().unwrap();
    let retrieved_episode_id = Uuid::parse_str(retrieved_episode_id).unwrap();
    assert_eq!(retrieved_episode_id, episode_id);
    // Check the variant name
    let variant_name = result.get("variant_name").unwrap().as_str().unwrap();
    assert_eq!(variant_name, "test");

    // Check the ModelInference Table
    let result = select_model_inference_clickhouse(&clickhouse, inference_id)
        .await
        .unwrap();
    let id = result.get("id").unwrap().as_str().unwrap();
    let _ = Uuid::parse_str(id).unwrap();
    let inference_id_result = result.get("inference_id").unwrap().as_str().unwrap();
    let inference_id_result = Uuid::parse_str(inference_id_result).unwrap();
    assert_eq!(inference_id_result, inference_id);

    let input_tokens = result.get("input_tokens").unwrap().as_u64().unwrap();
    assert_eq!(input_tokens, 10);
    let output_tokens = result.get("output_tokens").unwrap().as_u64().unwrap();
    assert_eq!(output_tokens, 10);
    let response_time_ms = result.get("response_time_ms").unwrap().as_u64().unwrap();
    assert!(response_time_ms > 0);
    assert!(result.get("ttft_ms").unwrap().is_null());
    assert_eq!(
        result.get("raw_response").unwrap().as_str().unwrap(),
        DUMMY_JSON_RESPONSE_RAW
    );
    assert_eq!(
        result.get("raw_request").unwrap().as_str().unwrap(),
        DUMMY_RAW_REQUEST
    );
    let system = result.get("system").unwrap().as_str().unwrap();
    assert_eq!(
        system,
        "You are a helpful and friendly assistant named AskJeeves.\n\nPlease answer the questions in a JSON with key \"answer\".\n\nDo not include any other text than the JSON object. Do not include \"```json\" or \"```\" or anything else.\n\nExample Response:\n\n{\n    \"answer\": \"42\"\n}"
    );
    let input_messages = result.get("input_messages").unwrap().as_str().unwrap();
    let input_messages: Vec<RequestMessage> = serde_json::from_str(input_messages).unwrap();
    assert_eq!(input_messages.len(), 1);
    assert_eq!(
        input_messages[0],
        RequestMessage {
            role: Role::User,
            content: vec!["What is the name of the capital city of Japan?"
                .to_string()
                .into()],
        }
    );
    let output = result.get("output").unwrap().as_str().unwrap();
    let output: Vec<ContentBlock> = serde_json::from_str(output).unwrap();
    assert_eq!(output, vec!["{\"answer\":\"Hello\"}".to_string().into()]);
}

/// The variant_failover function has two variants: good and error, each with weight 0.5
/// We want to make sure that this does not fail despite the error variant failing every time
/// We do this by making several requests and checking that the response is 200 in each, then checking that
/// the response is correct for the last one.
#[tokio::test]
async fn e2e_test_variant_failover() {
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
                        "content": [{"type": "text", "value": {"type": "tacos", "quantity": 13}}],
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
    assert_eq!(output_tokens, 10);
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
            "system": {
                "assistant_name": "AskJeeves"
            },
            "messages": [
            {
                "role": "user",
                "content": [{"type": "text", "value": {"type": "tacos", "quantity": 13}}]
            }
        ]}
    );
    assert_eq!(input, correct_input);
    let content_blocks = result.get("output").unwrap().as_str().unwrap();
    // Check that content_blocks is a list of blocks length 1
    let content_blocks: Vec<Value> = serde_json::from_str(content_blocks).unwrap();
    assert_eq!(content_blocks.len(), 1);
    let content_block = content_blocks.first().unwrap();
    // Check the type and content in the block
    let content_block_type = content_block.get("type").unwrap().as_str().unwrap();
    assert_eq!(content_block_type, "text");
    let content = content_block.get("text").unwrap().as_str().unwrap();
    assert_eq!(content, DUMMY_INFER_RESPONSE_CONTENT);
    // Check that episode_id is here and correct
    let retrieved_episode_id = result.get("episode_id").unwrap().as_str().unwrap();
    let retrieved_episode_id = Uuid::parse_str(retrieved_episode_id).unwrap();
    assert_eq!(retrieved_episode_id, episode_id);
    // Check the variant name
    let variant_name = result.get("variant_name").unwrap().as_str().unwrap();
    assert_eq!(variant_name, "good");

    // Check the inference_params (should be null since neither config or payload has chat_completion)
    let inference_params = result.get("inference_params").unwrap().as_str().unwrap();
    let inference_params: Value = serde_json::from_str(inference_params).unwrap();
    let chat_completion_inference_params = inference_params
        .get("chat_completion")
        .unwrap()
        .as_object()
        .unwrap();

    assert!(chat_completion_inference_params
        .get("temperature")
        .is_none());
    let max_tokens = chat_completion_inference_params.get("max_tokens").unwrap();
    assert_eq!(max_tokens.as_u64().unwrap(), 100);
    assert!(chat_completion_inference_params.get("seed").is_none());

    // Check the ModelInference Table
    let result = select_model_inference_clickhouse(&clickhouse, inference_id)
        .await
        .unwrap();
    let inference_id_result = result.get("inference_id").unwrap().as_str().unwrap();
    let inference_id_result = Uuid::parse_str(inference_id_result).unwrap();
    assert_eq!(inference_id_result, inference_id);

    let input_tokens = result.get("input_tokens").unwrap().as_u64().unwrap();
    assert_eq!(input_tokens, 10);
    let output_tokens = result.get("output_tokens").unwrap().as_u64().unwrap();
    assert_eq!(output_tokens, 10);
    let response_time_ms = result.get("response_time_ms").unwrap().as_u64().unwrap();
    assert!(response_time_ms > 0);
    assert!(result.get("ttft_ms").unwrap().is_null());
    assert_eq!(
        result.get("raw_response").unwrap().as_str().unwrap(),
        DUMMY_INFER_RESPONSE_RAW
    );
    assert_eq!(
        result.get("raw_request").unwrap().as_str().unwrap(),
        DUMMY_RAW_REQUEST
    );
    let system = result.get("system").unwrap().as_str().unwrap();
    assert_eq!(
        system,
        "You are a helpful and friendly assistant named AskJeeves"
    );
    let input_messages = result.get("input_messages").unwrap().as_str().unwrap();
    let input_messages: Vec<RequestMessage> = serde_json::from_str(input_messages).unwrap();
    assert_eq!(input_messages.len(), 1);
    assert_eq!(
        input_messages[0],
        RequestMessage {
            role: Role::User,
            content: vec!["I want 13 of tacos, please.".to_string().into()],
        }
    );
    let output = result.get("output").unwrap().as_str().unwrap();
    let output: Vec<ContentBlock> = serde_json::from_str(output).unwrap();
    assert_eq!(output, vec!["Megumin gleefully chanted her spell, unleashing a thunderous explosion that lit up the sky and left a massive crater in its wake.".to_string().into()]);
}

#[tokio::test(flavor = "multi_thread")]
async fn e2e_test_variant_zero_weight_skip_zero() {
    let client = make_embedded_gateway().await;
    let error = client
        .inference(ClientInferenceParams {
            function_name: Some("variant_failover_zero_weight".to_string()),
            input: ClientInput {
                system: Some(serde_json::json!({"assistant_name": "AskJeeves"})),
                messages: vec![ClientInputMessage {
                    role: Role::User,
                    content: vec![ClientInputMessageContent::Text(TextKind::Arguments {
                        arguments: serde_json::json!({"type": "tacos", "quantity": 13})
                            .as_object()
                            .unwrap()
                            .clone(),
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
async fn e2e_test_variant_zero_weight_pin_zero() {
    let client = make_embedded_gateway().await;
    let error = client
        .inference(ClientInferenceParams {
            function_name: Some("variant_failover_zero_weight".to_string()),
            variant_name: Some("zero_weight".to_string()),
            input: ClientInput {
                system: Some(serde_json::json!({"assistant_name": "AskJeeves"})),
                messages: vec![ClientInputMessage {
                    role: Role::User,
                    content: vec![ClientInputMessageContent::Text(TextKind::Arguments {
                        arguments: serde_json::json!({"type": "tacos", "quantity": 13})
                            .as_object()
                            .unwrap()
                            .clone(),
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
async fn e2e_test_streaming() {
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
            assert!(chunk_json
                .get("content")
                .unwrap()
                .as_array()
                .unwrap()
                .is_empty());
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
            "system": {
                "assistant_name": "AskJeeves"
            },
            "messages": [
            {
                "role": "user",
                "content": [{"type": "text", "value": "Hello, world!"}]
            }
        ]}
    );
    assert_eq!(input, correct_input);
    // Check content blocks
    let content_blocks = result.get("output").unwrap().as_str().unwrap();
    let content_blocks: Vec<Value> = serde_json::from_str(content_blocks).unwrap();
    assert_eq!(content_blocks.len(), 1);
    let content_block = content_blocks.first().unwrap();
    let content_block_type = content_block.get("type").unwrap().as_str().unwrap();
    assert_eq!(content_block_type, "text");
    let content = content_block.get("text").unwrap().as_str().unwrap();
    assert_eq!(content, DUMMY_STREAMING_RESPONSE.join(""));
    let retrieved_episode_id = result.get("episode_id").unwrap().as_str().unwrap();
    let retrieved_episode_id = Uuid::parse_str(retrieved_episode_id).unwrap();
    assert_eq!(retrieved_episode_id, episode_id);
    // Check the variant name
    let variant_name = result.get("variant_name").unwrap().as_str().unwrap();
    assert_eq!(variant_name, "test");
    // Check the inference_params (set via payload)
    let inference_params = result.get("inference_params").unwrap().as_str().unwrap();
    let inference_params: Value = serde_json::from_str(inference_params).unwrap();
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
    let result = select_model_inference_clickhouse(&clickhouse, inference_id)
        .await
        .unwrap();
    let input_tokens = result.get("input_tokens").unwrap().as_u64().unwrap();
    assert_eq!(input_tokens, 10);
    let output_tokens = result.get("output_tokens").unwrap().as_u64().unwrap();
    assert_eq!(output_tokens, 16);
    let response_time_ms = result.get("response_time_ms").unwrap().as_u64().unwrap();
    assert!(response_time_ms > 0);
    let ttft = result.get("ttft_ms").unwrap().as_u64().unwrap();
    assert!(ttft > 0 && ttft <= response_time_ms);
    result.get("raw_response").unwrap().as_str().unwrap();
    assert_eq!(
        result.get("raw_request").unwrap().as_str().unwrap(),
        DUMMY_RAW_REQUEST
    );
    let system = result.get("system").unwrap().as_str().unwrap();
    assert_eq!(
        system,
        "You are a helpful and friendly assistant named AskJeeves"
    );
    let input_messages = result.get("input_messages").unwrap().as_str().unwrap();
    let input_messages: Vec<RequestMessage> = serde_json::from_str(input_messages).unwrap();
    assert_eq!(input_messages.len(), 1);
    assert_eq!(
        input_messages[0],
        RequestMessage {
            role: Role::User,
            content: vec!["Hello, world!".to_string().into()],
        }
    );
    let output = result.get("output").unwrap().as_str().unwrap();
    let output: Vec<ContentBlock> = serde_json::from_str(output).unwrap();
    assert_eq!(output, vec![DUMMY_STREAMING_RESPONSE.join("").into()]);
}

/// This test checks that streaming inference works as expected when dryrun is true.
#[tokio::test]
async fn e2e_test_streaming_dryrun() {
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
            assert!(chunk_json
                .get("content")
                .unwrap()
                .as_array()
                .unwrap()
                .is_empty());
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

    // Check ClickHouse
    let clickhouse = get_clickhouse().await;
    let result = select_chat_inference_clickhouse(&clickhouse, inference_id.unwrap()).await;
    assert!(result.is_none()); // No inference should be written to ClickHouse when dryrun is true
}

#[tokio::test]
async fn e2e_test_inference_original_response_non_stream() {
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
async fn e2e_test_inference_original_response_stream() {
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
        "stream": true,
        "include_original_response": true,
    });

    let response = Client::new()
        .post(get_gateway_endpoint("/inference"))
        .json(&payload)
        .send()
        .await
        .unwrap();

    let status = response.status();
    let response_json = response.json::<Value>().await.unwrap();
    assert_eq!(
        status,
        StatusCode::BAD_REQUEST,
        "Expected bad request: {response_json}"
    );

    assert_eq!(
        response_json,
        serde_json::json!({
            "error": "Cannot set both `include_original_response` and `stream` to `true`",
        })
    );
    // Don't both checking ClickHouse, as we do that in lots of other tests.
}

#[tokio::test]
async fn test_gateway_template_no_fs_access() {
    // We use an embedded client so that we can control the number of
    // requests to the flaky judge.
    let gateway = make_embedded_gateway_with_config(&format!(
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
        input: ClientInput {
            system: Some(serde_json::json!({"assistant_name": "AskJeeves"})),
            messages: vec![ClientInputMessage {
                role: Role::User,
                content: vec![ClientInputMessageContent::Text(TextKind::Text {
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
        err.contains("Could not load template 'extra_templates/foo.minijinja' - if this a dynamic template included from the filesystem, please set [gateway.enable_template_filesystem_access] to `true`"),
        "Unexpected error: {err}"
    );
}

#[tokio::test]
async fn test_original_response_best_of_n_flaky_judge() {
    // We use an embedded client so that we can control the number of
    // requests to the flaky judge.
    let gateway = make_embedded_gateway_with_config(
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
        input: ClientInput {
            messages: vec![ClientInputMessage {
                role: Role::User,
                content: vec![ClientInputMessageContent::Text(TextKind::Text {
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
        panic!("Expected non-streaming response, got {:?}", good_response);
    };

    assert_eq!(
        good_response.original_response.unwrap(),
        DUMMY_INFER_RESPONSE_RAW,
    );

    // Second request to the flaky judge should fail
    let bad_response = gateway.inference(params).await.unwrap();
    let InferenceOutput::NonStreaming(InferenceResponse::Chat(bad_response)) = bad_response else {
        panic!("Expected non-streaming response, got {:?}", bad_response);
    };

    assert!(
        bad_response.original_response.is_none(),
        "Expected no original response"
    );
}

#[tokio::test]
async fn test_original_response_mixture_of_n_flaky_fuser() {
    // We use an embedded client so that we can control the number of
    // requests to the flaky judge.
    let gateway = make_embedded_gateway_with_config(
        r#"
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

    let params = ClientInferenceParams {
        function_name: Some("mixture_of_n".to_string()),
        input: ClientInput {
            messages: vec![ClientInputMessage {
                role: Role::User,
                content: vec![ClientInputMessageContent::Text(TextKind::Text {
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
        panic!("Expected non-streaming response, got {:?}", good_response);
    };

    assert_eq!(
        good_response.original_response.unwrap(),
        DUMMY_INFER_RESPONSE_RAW,
    );

    // Second request to the flaky judge should fail
    let bad_response = gateway.inference(params).await.unwrap();
    let InferenceOutput::NonStreaming(InferenceResponse::Chat(bad_response)) = bad_response else {
        panic!("Expected non-streaming response, got {:?}", bad_response);
    };

    assert!(
        bad_response.original_response.is_none(),
        "Expected no original response"
    );

    // Don't check ClickHouse, as we do that in lots of other tests.
}

#[tokio::test]
async fn e2e_test_tool_call_streaming() {
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
    let mut name: Option<String> = None;

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
            let new_name = content_block.get("raw_name").unwrap().as_str().unwrap();
            if i == 0 {
                name = Some(new_name.to_string());
            } else {
                assert_eq!(name, Some(new_name.to_string()));
            }
        } else {
            assert!(chunk_json
                .get("content")
                .unwrap()
                .as_array()
                .unwrap()
                .is_empty());
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
            "system": {
                "assistant_name": "AskJeeves"
            },
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "text", "value": "Hi I'm visiting Brooklyn from Brazil. What's the weather?"}]
                }
            ]
        }
    );
    assert_eq!(input, correct_input);
    // Check that content blocks are correct
    let content_blocks = result.get("output").unwrap().as_str().unwrap();
    let content_blocks: Vec<Value> = serde_json::from_str(content_blocks).unwrap();
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
    let retrieved_episode_id = result.get("episode_id").unwrap().as_str().unwrap();
    let retrieved_episode_id = Uuid::parse_str(retrieved_episode_id).unwrap();
    assert_eq!(retrieved_episode_id, episode_id);
    // Check the variant name
    let variant_name = result.get("variant_name").unwrap().as_str().unwrap();
    assert_eq!(variant_name, "variant");
    // Check the tool_params
    let tool_params = result.get("tool_params").unwrap().as_str().unwrap();
    let tool_params: Value = serde_json::from_str(tool_params).unwrap();
    let tools_available = tool_params
        .get("tools_available")
        .unwrap()
        .as_array()
        .unwrap();
    assert!(tools_available.len() == 1);
    assert!(
        tools_available
            .first()
            .unwrap()
            .get("name")
            .unwrap()
            .as_str()
            .unwrap()
            == "get_temperature"
    );
    assert!(tool_params.get("tool_choice").unwrap().as_str().unwrap() == "auto");
    assert_eq!(
        tool_params.get("parallel_tool_calls").unwrap(),
        &Value::Null
    );
    // Check the ModelInference Table
    let result = select_model_inference_clickhouse(&clickhouse, inference_id)
        .await
        .unwrap();
    let id = result.get("id").unwrap().as_str().unwrap();
    let _ = Uuid::parse_str(id).unwrap();
    let inference_id_result = result.get("inference_id").unwrap().as_str().unwrap();
    let inference_id_result = Uuid::parse_str(inference_id_result).unwrap();
    assert_eq!(inference_id_result, inference_id);

    let input_tokens = result.get("input_tokens").unwrap().as_u64().unwrap();
    assert_eq!(input_tokens, 10);
    let output_tokens = result.get("output_tokens").unwrap().as_u64().unwrap();
    assert_eq!(output_tokens, 5);
    let response_time_ms = result.get("response_time_ms").unwrap().as_u64().unwrap();
    assert!(response_time_ms > 0);
    assert!(result.get("ttft_ms").unwrap().as_u64().unwrap() > 50);
    result.get("raw_response").unwrap().as_str().unwrap();
    assert_eq!(
        result.get("raw_request").unwrap().as_str().unwrap(),
        DUMMY_RAW_REQUEST
    );

    let system = result.get("system").unwrap().as_str().unwrap();
    assert_eq!(
        system,
        "You are a helpful and friendly assistant named AskJeeves.\n\nPeople will ask you questions about the weather.\n\nIf asked about the weather, just respond with the tool call. Use the \"get_temperature\" tool.\n\nIf provided with a tool result, use it to respond to the user (e.g. \"The weather in New York is 55 degrees Fahrenheit.\")."
    );
    let input_messages = result.get("input_messages").unwrap().as_str().unwrap();
    let input_messages: Vec<RequestMessage> = serde_json::from_str(input_messages).unwrap();
    assert_eq!(input_messages.len(), 1);
    assert_eq!(
        input_messages[0],
        RequestMessage {
            role: Role::User,
            content: vec!["Hi I'm visiting Brooklyn from Brazil. What's the weather?"
                .to_string()
                .into()],
        }
    );
    let output = result.get("output").unwrap().as_str().unwrap();
    let output: Vec<ContentBlock> = serde_json::from_str(output).unwrap();
    assert_eq!(
        output,
        vec![ContentBlock::ToolCall(ToolCall {
            name: "get_temperature".to_string(),
            arguments: "{\"location\":\"Brooklyn\",\"units\":\"celsius\"}".to_string(),
            id: "0".to_string(),
        })]
    );
}

#[tokio::test]
async fn test_raw_text_http_gateway() {
    test_raw_text(make_http_gateway().await).await;
}

#[tokio::test(flavor = "multi_thread")]
async fn test_raw_text_embedded_gateway() {
    test_raw_text(make_embedded_gateway().await).await;
}

pub async fn test_raw_text(client: tensorzero::Client) {
    let episode_id = Uuid::now_v7();

    let InferenceOutput::NonStreaming(res) = client
        .inference(ClientInferenceParams {
            episode_id: Some(episode_id),
            function_name: Some("json_success".to_string()),
            input: ClientInput {
                system: Some(serde_json::json!({
                    "assistant_name": "Dr. Mehta"
                })),
                messages: vec![ClientInputMessage {
                    role: Role::User,
                    content: vec![ClientInputMessageContent::RawText {
                        value: "This is not the normal input".into(),
                    }],
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

    // Sleep to allow time for data to be inserted into ClickHouse (trailing writes from API)
    tokio::time::sleep(std::time::Duration::from_secs(1)).await;

    // Check if ClickHouse is ok - JsonInference Table
    let clickhouse = get_clickhouse().await;
    let result = select_json_inference_clickhouse(&clickhouse, inference_id)
        .await
        .unwrap();

    let id = result.get("id").unwrap().as_str().unwrap();
    let id = Uuid::parse_str(id).unwrap();
    assert_eq!(id, inference_id);

    let function_name = result.get("function_name").unwrap().as_str().unwrap();
    assert_eq!(function_name, "json_success");

    let variant_name = result.get("variant_name").unwrap().as_str().unwrap();
    assert_eq!(variant_name, "test");

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
                "content": [{"type": "raw_text", "value": "This is not the normal input"}]
            }
        ]
    });
    assert_eq!(input, correct_input);
    let output = result.get("output").unwrap().as_str().unwrap();
    let output: Value = serde_json::from_str(output).unwrap();
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
pub async fn e2e_test_dynamic_api_key() {
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
    assert_eq!(response.status(), StatusCode::BAD_GATEWAY);
    let response_json = response.json::<Value>().await.unwrap();
    let error_message = response_json.get("error").unwrap().as_str().unwrap();
    assert!(error_message.contains("API key missing for provider: Dummy"));

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

    // Sleep to allow time for data to be inserted into ClickHouse (trailing writes from API)
    tokio::time::sleep(std::time::Duration::from_secs(1)).await;

    // Check if ClickHouse is ok - ChatInference Table
    let clickhouse = get_clickhouse().await;
    let result = select_chat_inference_clickhouse(&clickhouse, inference_id)
        .await
        .unwrap();

    let id = result.get("id").unwrap().as_str().unwrap();
    let id = Uuid::parse_str(id).unwrap();
    assert_eq!(id, inference_id);

    let function_name = result.get("function_name").unwrap().as_str().unwrap();
    assert_eq!(function_name, payload["function_name"]);

    let variant_name = result.get("variant_name").unwrap().as_str().unwrap();
    assert_eq!(variant_name, "test_dynamic_api_key");

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
                "content": [{"type": "text", "value": "What is the name of the capital city of Japan?"}]
            }
        ]
    });
    assert_eq!(input, correct_input);

    let content_blocks = result.get("output").unwrap().as_str().unwrap();
    let content_blocks: Vec<Value> = serde_json::from_str(content_blocks).unwrap();
    assert_eq!(content_blocks.len(), 1);
    let content_block = content_blocks.first().unwrap();
    let content_block_type = content_block.get("type").unwrap().as_str().unwrap();
    assert_eq!(content_block_type, "text");
    let clickhouse_content = content_block.get("text").unwrap().as_str().unwrap();
    assert_eq!(clickhouse_content, content);

    let tool_params = result.get("tool_params").unwrap().as_str().unwrap();
    assert!(tool_params.is_empty());

    let inference_params = result.get("inference_params").unwrap().as_str().unwrap();
    let inference_params: Value = serde_json::from_str(inference_params).unwrap();
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
    let client = make_embedded_gateway_no_config().await;
    let response = client
        .inference(ClientInferenceParams {
            model_name: Some("dummy::my-model".to_string()),
            input: ClientInput {
                system: None,
                messages: vec![ClientInputMessage {
                    role: Role::User,
                    content: vec![ClientInputMessageContent::Text(TextKind::Text {
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

    // Sleep to allow time for data to be inserted into ClickHouse (trailing writes from API)
    tokio::time::sleep(std::time::Duration::from_millis(100)).await;

    // Check if ClickHouse is ok - ChatInference Table
    let clickhouse = get_clickhouse().await;
    let result = select_chat_inference_clickhouse(&clickhouse, response.inference_id())
        .await
        .unwrap();

    let id = result.get("id").unwrap().as_str().unwrap();
    let id = Uuid::parse_str(id).unwrap();
    assert_eq!(id, response.inference_id());

    let function_name = result.get("function_name").unwrap().as_str().unwrap();
    assert_eq!(function_name, "tensorzero::default");
    // It's not necessary to check ModelInference table given how many other places we do that
}

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
            "Message has non-string content but there is no schema given for role system."
        ),
        "Unexpected error message: {response_text}",
    );
}

#[tokio::test]

async fn test_image_inference_without_object_store() {
    let client = make_embedded_gateway_no_config().await;
    let err_msg = client
        .inference(ClientInferenceParams {
            model_name: Some("openai::gpt-4o-mini".to_string()),
            input: ClientInput {
                system: None,
                messages: vec![ClientInputMessage {
                    role: Role::User,
                    content: vec![
                        ClientInputMessageContent::Text(TextKind::Text {
                            text: "Describe the contents of the image".to_string(),
                        }),
                        ClientInputMessageContent::Image(Image::Base64 {
                            mime_type: ImageKind::Png,
                            data: BASE64_STANDARD.encode(FERRIS_PNG),
                        }),
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
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "text", "value": "Hello, world!"}]
                }
            ]
        }
    );
    assert_eq!(input, correct_input);
    // Check that content blocks are correct
    let content_blocks = result.get("output").unwrap().as_str().unwrap();
    let content_blocks: Vec<Value> = serde_json::from_str(content_blocks).unwrap();
    assert_eq!(content_blocks.len(), 1);
    let content_block = content_blocks.first().unwrap();
    let content_block_type = content_block.get("type").unwrap().as_str().unwrap();
    assert_eq!(content_block_type, "text");
    let content = content_block.get("text").unwrap().as_str().unwrap();
    assert_eq!(content, DUMMY_INFER_RESPONSE_CONTENT);
    // Check that episode_id is here and correct
    let retrieved_episode_id = result.get("episode_id").unwrap().as_str().unwrap();
    let retrieved_episode_id = Uuid::parse_str(retrieved_episode_id).unwrap();
    assert_eq!(retrieved_episode_id, episode_id);
    // Check the variant name
    let variant_name = result.get("variant_name").unwrap().as_str().unwrap();
    assert_eq!(variant_name, model_name);

    // Check the ModelInference Table
    let result = select_model_inference_clickhouse(&clickhouse, inference_id)
        .await
        .unwrap();
    let id = result.get("id").unwrap().as_str().unwrap();
    let _ = Uuid::parse_str(id).unwrap();
    let inference_id_result = result.get("inference_id").unwrap().as_str().unwrap();
    let inference_id_result = Uuid::parse_str(inference_id_result).unwrap();
    assert_eq!(inference_id_result, inference_id);

    let input_tokens = result.get("input_tokens").unwrap();
    let input_tokens = if let Some(val) = input_tokens.as_u64() {
        Some(val)
    } else if input_tokens.is_null() {
        None
    } else {
        panic!("input_tokens is not a u64 or null: {input_tokens:?}");
    };

    let output_tokens = result.get("output_tokens").unwrap();
    let output_tokens = if let Some(val) = output_tokens.as_u64() {
        Some(val)
    } else if output_tokens.is_null() {
        None
    } else {
        panic!("output_tokens is not a u64 or null: {output_tokens:?}");
    };
    assert_eq!(input_tokens, expected_input_tokens);
    assert_eq!(output_tokens, expected_output_tokens);
}

#[tokio::test]
async fn test_inference_input_tokens_zero() {
    test_inference_zero_tokens_helper("dummy::input_tokens_zero", None, Some(10)).await;
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
#[traced_test]

async fn test_tool_call_input_no_warning() {
    let client = make_embedded_gateway_no_config().await;
    client
        .inference(ClientInferenceParams {
            model_name: Some("dummy::good".to_string()),
            input: ClientInput {
                system: None,
                messages: vec![ClientInputMessage {
                    role: Role::User,
                    content: vec![
                        ClientInputMessageContent::Text(TextKind::Text {
                            text: "Describe the contents of the image".to_string(),
                        }),
                        ClientInputMessageContent::ToolCall(ToolCallInput {
                            name: Some("get_temperature".to_string()),
                            arguments: Some(json!({
                                "location": "Brooklyn",
                                "units": "celsius"
                            })),
                            raw_arguments: None,
                            raw_name: None,
                            id: "0".to_string(),
                        }),
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

#[tokio::test]
async fn test_client_no_version() {
    let mut version_remove_headers = reqwest::header::HeaderMap::new();
    version_remove_headers.append(
        "x-tensorzero-e2e-version-remove",
        HeaderValue::from_static("true"),
    );

    let http_remove_version = reqwest::ClientBuilder::new()
        .default_headers(version_remove_headers)
        .build()
        .unwrap();

    let remove_version_gateway = ClientBuilder::new(ClientBuilderMode::HTTPGateway {
        url: get_gateway_endpoint("/"),
    })
    .with_http_client(http_remove_version)
    .build()
    .await
    .unwrap();

    // The discovery query should not give a version, due to 'x-tensorzero-e2e-version-remove'
    let version = remove_version_gateway.get_gateway_version().await;
    assert_eq!(version, None);
}

#[tokio::test]
async fn test_client_detect_version() {
    let mut version_override_headers = reqwest::header::HeaderMap::new();
    version_override_headers.append(
        "x-tensorzero-e2e-version-override",
        HeaderValue::from_static("3025.01.12"),
    );

    let http_client = reqwest::ClientBuilder::new()
        .default_headers(version_override_headers)
        .build()
        .unwrap();

    let gateway = ClientBuilder::new(ClientBuilderMode::HTTPGateway {
        url: get_gateway_endpoint("/"),
    })
    .with_http_client(http_client)
    .build()
    .await
    .unwrap();

    // The client should have used the version header (overridden by 'x-tensorzero-e2e-version-override')
    let version = gateway.get_gateway_version().await;
    assert_eq!(version, Some("3025.01.12".to_string()));
}

#[tokio::test]
async fn test_client_adjust_tool_call() {
    let bad_gateway = ClientBuilder::new(ClientBuilderMode::HTTPGateway {
        url: Url::parse("http://tensorzero.invalid").unwrap(),
    })
    .build()
    .await
    .unwrap();

    let params = ClientInferenceParams {
        function_name: Some("basic_test".to_string()),
        input: ClientInput {
            system: None,
            messages: vec![ClientInputMessage {
                role: Role::User,
                content: vec![ClientInputMessageContent::ToolCall(ToolCallInput {
                    name: Some("my_tool_call".to_string()),
                    arguments: Some(json!({
                        "location": "Brooklyn",
                        "units": "celsius"
                    })),
                    id: "my_id".to_string(),
                    raw_arguments: None,
                    raw_name: None,
                })],
            }],
        },
        ..Default::default()
    };

    bad_gateway.inference(params.clone()).await.unwrap_err();
    let stringified_tool_call_args = r#"{"function_name":"basic_test","model_name":null,"episode_id":null,"input":{"messages":[{"role":"user","content":[{"type":"tool_call","name":"my_tool_call","arguments":"{\"location\":\"Brooklyn\",\"units\":\"celsius\"}","id":"my_id"}]}]},"stream":null,"params":{"chat_completion":{}},"variant_name":null,"dryrun":null,"internal":false,"tags":{},"allowed_tools":null,"additional_tools":null,"tool_choice":null,"parallel_tool_calls":null,"output_schema":null,"credentials":{},"cache_options":{"max_age_s":null,"enabled":"write_only"},"include_original_response":false,"extra_body":[]}"#;
    let non_stringified_tool_call_args = r#"{"function_name":"basic_test","model_name":null,"episode_id":null,"input":{"messages":[{"role":"user","content":[{"type":"tool_call","name":"my_tool_call","arguments":{"location":"Brooklyn","units":"celsius"},"id":"my_id"}]}]},"stream":null,"params":{"chat_completion":{}},"variant_name":null,"dryrun":null,"internal":false,"tags":{},"allowed_tools":null,"additional_tools":null,"tool_choice":null,"parallel_tool_calls":null,"output_schema":null,"credentials":{},"cache_options":{"max_age_s":null,"enabled":"write_only"},"include_original_response":false,"extra_body":[]}"#;

    // With an invalid gateway url, we shouldn't get a version
    assert_eq!(bad_gateway.get_gateway_version().await, None);

    let last_body = { bad_gateway.last_body.lock().await.take().unwrap() };

    assert_eq!(last_body, stringified_tool_call_args);

    // Set an older gateway version, and verify that we still stringify the tool call arguments
    bad_gateway
        .e2e_update_gateway_version("2025.03.2".to_string())
        .await;
    bad_gateway.inference(params.clone()).await.unwrap_err();

    let last_body = { bad_gateway.last_body.lock().await.take().unwrap() };
    assert_eq!(last_body, stringified_tool_call_args);

    // Set a newer gateway version, and verify that we do not stringify the arguments
    bad_gateway
        .e2e_update_gateway_version("2025.03.3".to_string())
        .await;
    bad_gateway.inference(params).await.unwrap_err();

    let last_body = { bad_gateway.last_body.lock().await.take().unwrap() };
    assert_eq!(last_body, non_stringified_tool_call_args);
}
