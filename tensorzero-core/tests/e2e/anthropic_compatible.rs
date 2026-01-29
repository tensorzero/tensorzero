#![expect(clippy::print_stdout)]

use std::collections::HashSet;

use axum::extract::State;
use http_body_util::BodyExt;
use reqwest::{Client, StatusCode};
use serde_json::{Value, json};
use uuid::Uuid;

use crate::common::get_gateway_endpoint;

use tensorzero_core::endpoints::anthropic_compatible::messages::messages_handler;
use tensorzero_core::{
    db::clickhouse::test_helpers::{
        get_clickhouse, select_chat_inference_clickhouse, select_model_inference_clickhouse,
    },
    utils::gateway::StructuredJson,
};

#[tokio::test(flavor = "multi_thread")]
async fn test_anthropic_compatible_basic_request() {
    let client = tensorzero::test_helpers::make_embedded_gateway().await;
    let state = client.get_app_state_data().unwrap().clone();
    let episode_id = Uuid::now_v7();

    let response = messages_handler(
        State(state),
        None,
        StructuredJson(
            serde_json::from_value(serde_json::json!({
                "model": "tensorzero::function_name::basic_test_no_system_schema",
                "max_tokens": 100,
                "messages": [
                    {
                        "role": "user",
                        "content": "What is the capital of Japan?"
                    }
                ],
                "tensorzero::tags": {
                    "foo": "bar"
                },
                "tensorzero::episode_id": episode_id.to_string(),
            }))
            .unwrap(),
        ),
    )
    .await
    .unwrap();

    // Check Response is OK
    assert_eq!(response.status(), StatusCode::OK);
    let response_json = response.into_body().collect().await.unwrap().to_bytes();
    let response_json: Value = serde_json::from_slice(&response_json).unwrap();
    println!("response: {response_json:?}");

    // Check basic response structure
    assert_eq!(
        response_json.get("type").unwrap().as_str().unwrap(),
        "message"
    );
    assert_eq!(
        response_json.get("role").unwrap().as_str().unwrap(),
        "assistant"
    );

    // Check content array
    let content = response_json.get("content").unwrap().as_array().unwrap();
    assert_eq!(content.len(), 1);
    let first_block = content.first().unwrap();
    assert_eq!(first_block.get("type").unwrap().as_str().unwrap(), "text");
    let text = first_block.get("text").unwrap().as_str().unwrap();
    assert_eq!(
        text,
        "Megumin gleefully chanted her spell, unleashing a thunderous explosion that lit up the sky and left a massive crater in its wake."
    );

    // Check model prefix
    let response_model = response_json.get("model").unwrap().as_str().unwrap();
    assert_eq!(
        response_model,
        "tensorzero::function_name::basic_test_no_system_schema::variant_name::test"
    );

    // Check stop_reason
    let stop_reason = response_json.get("stop_reason").unwrap().as_str().unwrap();
    assert_eq!(stop_reason, "end_turn");

    // Get inference_id
    let inference_id: Uuid = response_json
        .get("id")
        .unwrap()
        .as_str()
        .unwrap()
        .parse()
        .unwrap();

    // Sleep for 1 second to allow time for data to be inserted into ClickHouse
    tokio::time::sleep(std::time::Duration::from_secs(1)).await;

    // Check ClickHouse
    let clickhouse = get_clickhouse().await;

    // Check Inference table
    let result = select_chat_inference_clickhouse(&clickhouse, inference_id)
        .await
        .unwrap();
    let id = result.get("id").unwrap().as_str().unwrap();
    let id_uuid = Uuid::parse_str(id).unwrap();
    assert_eq!(id_uuid, inference_id);
    let function_name = result.get("function_name").unwrap().as_str().unwrap();
    assert_eq!(function_name, "basic_test_no_system_schema");

    // Check tags
    let tags = result.get("tags").unwrap().as_object().unwrap();
    assert_eq!(tags.get("foo").unwrap().as_str().unwrap(), "bar");
    assert_eq!(tags.len(), 1);

    // Check variant name
    let variant_name = result.get("variant_name").unwrap().as_str().unwrap();
    assert_eq!(variant_name, "test");

    // Check the ModelInference Table
    let result = select_model_inference_clickhouse(&clickhouse, inference_id)
        .await
        .unwrap();
    println!("ModelInference result: {result:?}");
    let inference_id_result = result.get("inference_id").unwrap().as_str().unwrap();
    let inference_id_result = Uuid::parse_str(inference_id_result).unwrap();
    assert_eq!(inference_id_result, inference_id);
    let model_name = result.get("model_name").unwrap().as_str().unwrap();
    assert_eq!(model_name, "test");
    let finish_reason = result.get("finish_reason").unwrap().as_str().unwrap();
    assert_eq!(finish_reason, "stop");
}

#[tokio::test(flavor = "multi_thread")]
async fn test_anthropic_compatible_system_prompt_string() {
    let client = tensorzero::test_helpers::make_embedded_gateway().await;
    let state = client.get_app_state_data().unwrap().clone();

    let response = messages_handler(
        State(state),
        None,
        StructuredJson(
            serde_json::from_value(serde_json::json!({
                "model": "tensorzero::function_name::basic_test_no_system_schema",
                "max_tokens": 100,
                "system": "You are a helpful assistant named TensorBot.",
                "messages": [
                    {
                        "role": "user",
                        "content": "Hello"
                    }
                ]
            }))
            .unwrap(),
        ),
    )
    .await
    .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    let response_json = response.into_body().collect().await.unwrap().to_bytes();
    let response_json: Value = serde_json::from_slice(&response_json).unwrap();
    println!("response: {response_json:?}");

    // Just check that we got a valid response
    assert_eq!(response_json.get("role").unwrap().as_str().unwrap(), "assistant");
    let content = response_json.get("content").unwrap().as_array().unwrap();
    assert!(!content.is_empty());
}

#[tokio::test(flavor = "multi_thread")]
async fn test_anthropic_compatible_system_prompt_array() {
    let client = tensorzero::test_helpers::make_embedded_gateway().await;
    let state = client.get_app_state_data().unwrap().clone();

    let response = messages_handler(
        State(state),
        None,
        StructuredJson(
            serde_json::from_value(serde_json::json!({
                "model": "tensorzero::function_name::basic_test_no_system_schema",
                "max_tokens": 100,
                "system": [
                    {
                        "type": "text",
                        "text": "You are a helpful assistant."
                    }
                ],
                "messages": [
                    {
                        "role": "user",
                        "content": "Hello"
                    }
                ]
            }))
            .unwrap(),
        ),
    )
    .await
    .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    let response_json = response.into_body().collect().await.unwrap().to_bytes();
    let response_json: Value = serde_json::from_slice(&response_json).unwrap();

    // Just check that we got a valid response
    assert_eq!(response_json.get("role").unwrap().as_str().unwrap(), "assistant");
    let content = response_json.get("content").unwrap().as_array().unwrap();
    assert!(!content.is_empty());
}

#[tokio::test(flavor = "multi_thread")]
async fn test_anthropic_compatible_missing_max_tokens() {
    let client = tensorzero::test_helpers::make_embedded_gateway().await;
    let state = client.get_app_state_data().unwrap().clone();

    let error = messages_handler(
        State(state),
        None,
        StructuredJson(
            serde_json::from_value(serde_json::json!({
                "model": "tensorzero::function_name::basic_test_no_system_schema",
                "messages": [
                    {
                        "role": "user",
                        "content": "Hello"
                    }
                ]
            }))
            .unwrap(),
        ),
    )
    .await
    .unwrap_err();

    assert_eq!(error.status(), StatusCode::BAD_REQUEST);
    assert!(error
        .to_string()
        .contains("max_tokens"));
}

#[tokio::test(flavor = "multi_thread")]
async fn test_anthropic_compatible_max_tokens_zero() {
    let client = tensorzero::test_helpers::make_embedded_gateway().await;
    let state = client.get_app_state_data().unwrap().clone();

    let error = messages_handler(
        State(state),
        None,
        StructuredJson(
            serde_json::from_value(serde_json::json!({
                "model": "tensorzero::function_name::basic_test_no_system_schema",
                "max_tokens": 0,
                "messages": [
                    {
                        "role": "user",
                        "content": "Hello"
                    }
                ]
            }))
            .unwrap(),
        ),
    )
    .await
    .unwrap_err();

    assert_eq!(error.status(), StatusCode::BAD_REQUEST);
    assert!(error
        .to_string()
        .contains("max_tokens"));
}

#[tokio::test(flavor = "multi_thread")]
async fn test_anthropic_compatible_invalid_model_prefix() {
    let client = tensorzero::test_helpers::make_embedded_gateway().await;
    let state = client.get_app_state_data().unwrap().clone();

    let error = messages_handler(
        State(state),
        None,
        StructuredJson(
            serde_json::from_value(serde_json::json!({
                "model": "invalid::model::name",
                "max_tokens": 100,
                "messages": [
                    {
                        "role": "user",
                        "content": "Hello"
                    }
                ]
            }))
            .unwrap(),
        ),
    )
    .await
    .unwrap_err();

    // Should get an error about invalid model prefix
    assert!(error.to_string().contains("model") || error.to_string().contains("function"));
}

#[tokio::test(flavor = "multi_thread")]
async fn test_anthropic_compatible_model_name_target() {
    let client = tensorzero::test_helpers::make_embedded_gateway().await;
    let state = client.get_app_state_data().unwrap().clone();

    let response = messages_handler(
        State(state),
        None,
        StructuredJson(
            serde_json::from_value(serde_json::json!({
                "model": "tensorzero::model_name::openai::gpt-4o-mini",
                "max_tokens": 100,
                "messages": [
                    {
                        "role": "user",
                        "content": "Say 'test passed'"
                    }
                ]
            }))
            .unwrap(),
        ),
    )
    .await
    .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    let response_json = response.into_body().collect().await.unwrap().to_bytes();
    let response_json: Value = serde_json::from_slice(&response_json).unwrap();

    // Check model prefix
    let response_model = response_json.get("model").unwrap().as_str().unwrap();
    assert_eq!(response_model, "tensorzero::model_name::openai::gpt-4o-mini");
}

#[tokio::test(flavor = "multi_thread")]
async fn test_anthropic_compatible_streaming() {
    use futures::StreamExt;
    use reqwest_eventsource::{Event, RequestBuilderExt};

    let client = Client::new();
    let episode_id = Uuid::now_v7();

    let body = json!({
        "model": "tensorzero::function_name::basic_test_no_system_schema",
        "max_tokens": 100,
        "messages": [
            {
                "role": "user",
                "content": "What's the reason for why we use AC not DC?"
            }
        ],
        "stream": true,
        "tensorzero::episode_id": episode_id.to_string(),
    });

    let mut response = client
        .post(get_gateway_endpoint("/anthropic/v1/messages"))
        .header("Content-Type", "application/json")
        .json(&body)
        .eventsource()
        .unwrap();

    let mut events = vec![];
    let mut found_message_start = false;
    let mut found_message_stop = false;
    let mut found_content_block_start = false;
    let mut found_content_block_delta = false;
    let mut found_message_delta = false;

    while let Some(event) = response.next().await {
        let event = event.unwrap();
        match event {
            Event::Open => continue,
            Event::Message(message) => {
                if message.data == "[DONE]" {
                    break;
                }
                events.push((message.event, message.data));
            }
        }
    }

    // Check we got the expected event types
    for (event_type, data) in &events {
        let parsed: Value = serde_json::from_str(data).unwrap();
        println!("Event type: {event_type}, Data: {parsed}");

        match event_type.as_str() {
            "message_start" => {
                found_message_start = true;
                assert!(parsed.get("message").is_some());
                assert!(parsed["message"].get("id").is_some());
                assert_eq!(parsed["message"]["type"].as_str().unwrap(), "message");
                assert_eq!(parsed["message"]["role"].as_str().unwrap(), "assistant");
            }
            "content_block_start" => {
                found_content_block_start = true;
                assert!(parsed.get("content_block").is_some());
                assert!(parsed.get("index").is_some());
            }
            "content_block_delta" => {
                found_content_block_delta = true;
                assert!(parsed.get("delta").is_some());
                assert!(parsed.get("index").is_some());
            }
            "content_block_stop" => {
                assert!(parsed.get("index").is_some());
            }
            "message_delta" => {
                found_message_delta = true;
                assert!(parsed.get("delta").is_some());
                assert!(parsed["delta"].get("stop_reason").is_some());
            }
            "message_stop" => {
                found_message_stop = true;
            }
            _ => {}
        }
    }

    assert!(found_message_start, "Should have message_start event");
    assert!(found_content_block_start, "Should have content_block_start event");
    assert!(found_content_block_delta, "Should have content_block_delta event");
    assert!(found_message_delta, "Should have message_delta event");
    assert!(found_message_stop, "Should have message_stop event");
}

#[tokio::test(flavor = "multi_thread")]
async fn test_anthropic_compatible_tool_use() {
    let client = tensorzero::test_helpers::make_embedded_gateway().await;
    let state = client.get_app_state_data().unwrap().clone();
    let episode_id = Uuid::now_v7();

    let response = messages_handler(
        State(state),
        None,
        StructuredJson(
            serde_json::from_value(serde_json::json!({
                "model": "tensorzero::function_name::weather_helper",
                "max_tokens": 1000,
                "messages": [
                    {
                        "role": "user",
                        "content": "What's the weather in Tokyo?"
                    }
                ],
                "tools": [
                    {
                        "name": "get_temperature",
                        "description": "Get the current temperature in a given location",
                        "input_schema": {
                            "type": "object",
                            "properties": {
                                "location": {
                                    "type": "string",
                                    "description": "The city and state, e.g. San Francisco, CA"
                                }
                            },
                            "required": ["location"]
                        }
                    }
                ],
                "tensorzero::episode_id": episode_id.to_string(),
            }))
            .unwrap(),
        ),
    )
    .await
    .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    let response_json = response.into_body().collect().await.unwrap().to_bytes();
    let response_json: Value = serde_json::from_slice(&response_json).unwrap();
    println!("response: {response_json:?}");

    // Check that we got a tool_use block
    let content = response_json.get("content").unwrap().as_array().unwrap();
    let tool_use_block = content
        .iter()
        .find(|block| block.get("type").unwrap().as_str().unwrap() == "tool_use");
    assert!(tool_use_block.is_some());

    let tool_use = tool_use_block.unwrap();
    assert_eq!(tool_use.get("name").unwrap().as_str().unwrap(), "get_temperature");
    assert!(tool_use.get("id").is_some());
    assert!(tool_use.get("input").is_some());

    // Check stop_reason is tool_use
    assert_eq!(
        response_json.get("stop_reason").unwrap().as_str().unwrap(),
        "tool_use"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn test_anthropic_compatible_streaming_tool_use() {
    use futures::StreamExt;
    use reqwest_eventsource::{Event, RequestBuilderExt};

    let client = Client::new();
    let episode_id = Uuid::now_v7();

    let body = json!({
        "model": "tensorzero::function_name::weather_helper",
        "max_tokens": 1000,
        "messages": [
            {
                "role": "user",
                "content": "What's the weather in Tokyo?"
            }
        ],
        "tools": [
            {
                "name": "get_temperature",
                "description": "Get the current temperature in a given location",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city and state, e.g. San Francisco, CA"
                        }
                    },
                    "required": ["location"]
                }
            }
        ],
        "stream": true,
        "tensorzero::episode_id": episode_id.to_string(),
    });

    let mut response = client
        .post(get_gateway_endpoint("/anthropic/v1/messages"))
        .header("Content-Type", "application/json")
        .json(&body)
        .eventsource()
        .unwrap();

    let mut found_tool_use_start = false;
    let mut found_tool_use_delta = false;

    while let Some(event) = response.next().await {
        let event = event.unwrap();
        match event {
            Event::Open => continue,
            Event::Message(message) => {
                if message.data == "[DONE]" {
                    break;
                }
                let parsed: Value = serde_json::from_str(&message.data).unwrap();
                println!("Event: {} - Data: {}", message.event, parsed);

                if message.event == "content_block_start" {
                    if let Some(content_block) = parsed.get("content_block") {
                        if content_block.get("type").unwrap().as_str().unwrap() == "tool_use" {
                            found_tool_use_start = true;
                        }
                    }
                }
                if message.event == "content_block_delta" {
                    if let Some(delta) = parsed.get("delta") {
                        if delta.get("type").unwrap().as_str().unwrap() == "input_json_delta" {
                            found_tool_use_delta = true;
                        }
                    }
                }
            }
        }
    }

    assert!(found_tool_use_start, "Should have tool_use content_block_start event");
    assert!(found_tool_use_delta, "Should have input_json_delta event");
}

#[tokio::test(flavor = "multi_thread")]
async fn test_anthropic_compatible_tool_result() {
    let client = tensorzero::test_helpers::make_embedded_gateway().await;
    let state = client.get_app_state_data().unwrap().clone();
    let episode_id = Uuid::now_v7();

    // First, get a tool use
    let response1 = messages_handler(
        State(state.clone()),
        None,
        StructuredJson(
            serde_json::from_value(serde_json::json!({
                "model": "tensorzero::function_name::weather_helper",
                "max_tokens": 1000,
                "messages": [
                    {
                        "role": "user",
                        "content": "What's the weather in Tokyo?"
                    }
                ],
                "tools": [
                    {
                        "name": "get_temperature",
                        "description": "Get the current temperature in a given location",
                        "input_schema": {
                            "type": "object",
                            "properties": {
                                "location": {
                                    "type": "string",
                                    "description": "The city and state, e.g. San Francisco, CA"
                                }
                            },
                            "required": ["location"]
                        }
                    }
                ],
                "tensorzero::episode_id": episode_id.to_string(),
            }))
            .unwrap(),
        ),
    )
    .await
    .unwrap();

    let response1_json = response1.into_body().collect().await.unwrap().to_bytes();
    let response1_json: Value = serde_json::from_slice(&response1_json).unwrap();

    // Extract the tool_use_id
    let content = response1_json.get("content").unwrap().as_array().unwrap();
    let tool_use_block = content
        .iter()
        .find(|block| block.get("type").unwrap().as_str().unwrap() == "tool_use")
        .unwrap();
    let tool_use_id = tool_use_block.get("id").unwrap().as_str().unwrap();

    // Now send the tool result
    let response2 = messages_handler(
        State(state),
        None,
        StructuredJson(
            serde_json::from_value(serde_json::json!({
                "model": "tensorzero::function_name::weather_helper",
                "max_tokens": 1000,
                "messages": [
                    {
                        "role": "user",
                        "content": "What's the weather in Tokyo?"
                    },
                    {
                        "role": "assistant",
                        "content": [
                            {
                                "type": "tool_use",
                                "id": tool_use_id,
                                "name": "get_temperature",
                                "input": {"location": "Tokyo"}
                            }
                        ]
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": tool_use_id,
                                "content": "The temperature in Tokyo is 22°C"
                            }
                        ]
                    }
                ],
                "tools": [
                    {
                        "name": "get_temperature",
                        "description": "Get the current temperature in a given location",
                        "input_schema": {
                            "type": "object",
                            "properties": {
                                "location": {
                                    "type": "string",
                                    "description": "The city and state, e.g. San Francisco, CA"
                                }
                            },
                            "required": ["location"]
                        }
                    }
                ],
                "tensorzero::episode_id": episode_id.to_string(),
            }))
            .unwrap(),
        ),
    )
    .await
    .unwrap();

    assert_eq!(response2.status(), StatusCode::OK);
    let response2_json = response2.into_body().collect().await.unwrap().to_bytes();
    let response2_json: Value = serde_json::from_slice(&response2_json).unwrap();
    println!("response2: {response2_json:?}");

    // Check that we got a text response (not a tool_use)
    let content2 = response2_json.get("content").unwrap().as_array().unwrap();
    assert!(content2
        .iter()
        .any(|block| block.get("type").unwrap().as_str().unwrap() == "text"));
}

#[tokio::test(flavor = "multi_thread")]
async fn test_anthropic_compatible_tool_choice_auto() {
    let client = tensorzero::test_helpers::make_embedded_gateway().await;
    let state = client.get_app_state_data().unwrap().clone();

    let response = messages_handler(
        State(state),
        None,
        StructuredJson(
            serde_json::from_value(serde_json::json!({
                "model": "tensorzero::function_name::weather_helper",
                "max_tokens": 1000,
                "messages": [
                    {
                        "role": "user",
                        "content": "What's the weather in Tokyo?"
                    }
                ],
                "tools": [
                    {
                        "name": "get_temperature",
                        "description": "Get the current temperature in a given location",
                        "input_schema": {
                            "type": "object",
                            "properties": {
                                "location": {
                                    "type": "string",
                                    "description": "The city and state, e.g. San Francisco, CA"
                                }
                            },
                            "required": ["location"]
                        }
                    }
                ],
                "tool_choice": "auto"
            }))
            .unwrap(),
        ),
    )
    .await
    .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
}

#[tokio::test(flavor = "multi_thread")]
async fn test_anthropic_compatible_tool_choice_any() {
    let client = tensorzero::test_helpers::make_embedded_gateway().await;
    let state = client.get_app_state_data().unwrap().clone();

    let response = messages_handler(
        State(state),
        None,
        StructuredJson(
            serde_json::from_value(serde_json::json!({
                "model": "tensorzero::function_name::weather_helper",
                "max_tokens": 1000,
                "messages": [
                    {
                        "role": "user",
                        "content": "Hello"
                    }
                ],
                "tools": [
                    {
                        "name": "get_temperature",
                        "description": "Get the current temperature in a given location",
                        "input_schema": {
                            "type": "object",
                            "properties": {
                                "location": {
                                    "type": "string",
                                    "description": "The city and state, e.g. San Francisco, CA"
                                }
                            },
                            "required": ["location"]
                        }
                    }
                ],
                "tool_choice": "any"
            }))
            .unwrap(),
        ),
    )
    .await
    .unwrap();

    assert_eq!(response.status(), StatusCode::OK);

    // With "any", it should force a tool call even if not needed
    let response_json = response.into_body().collect().await.unwrap().to_bytes();
    let response_json: Value = serde_json::from_slice(&response_json).unwrap();

    let content = response_json.get("content").unwrap().as_array().unwrap();
    let tool_use_block = content
        .iter()
        .find(|block| block.get("type").unwrap().as_str().unwrap() == "tool_use");
    assert!(tool_use_block.is_some());
}

#[tokio::test(flavor = "multi_thread")]
async fn test_anthropic_compatible_tool_choice_specific() {
    let client = tensorzero::test_helpers::make_embedded_gateway().await;
    let state = client.get_app_state_data().unwrap().clone();

    let response = messages_handler(
        State(state),
        None,
        StructuredJson(
            serde_json::from_value(serde_json::json!({
                "model": "tensorzero::function_name::weather_helper",
                "max_tokens": 1000,
                "messages": [
                    {
                        "role": "user",
                        "content": "Hello"
                    }
                ],
                "tools": [
                    {
                        "name": "get_temperature",
                        "description": "Get the current temperature in a given location",
                        "input_schema": {
                            "type": "object",
                            "properties": {
                                "location": {
                                    "type": "string",
                                    "description": "The city and state, e.g. San Francisco, CA"
                                }
                            },
                            "required": ["location"]
                        }
                    }
                ],
                "tool_choice": {
                    "type": "tool",
                    "name": "get_temperature"
                }
            }))
            .unwrap(),
        ),
    )
    .await
    .unwrap();

    assert_eq!(response.status(), StatusCode::OK);

    // Should use the specific tool
    let response_json = response.into_body().collect().await.unwrap().to_bytes();
    let response_json: Value = serde_json::from_slice(&response_json).unwrap();

    let content = response_json.get("content").unwrap().as_array().unwrap();
    let tool_use_block = content
        .iter()
        .find(|block| block.get("type").unwrap().as_str().unwrap() == "tool_use");
    assert!(tool_use_block.is_some());

    let tool_use = tool_use_block.unwrap();
    assert_eq!(
        tool_use.get("name").unwrap().as_str().unwrap(),
        "get_temperature"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn test_anthropic_compatible_temperature() {
    let client = tensorzero::test_helpers::make_embedded_gateway().await;
    let state = client.get_app_state_data().unwrap().clone();

    let response = messages_handler(
        State(state),
        None,
        StructuredJson(
            serde_json::from_value(serde_json::json!({
                "model": "tensorzero::function_name::basic_test_no_system_schema",
                "max_tokens": 100,
                "messages": [
                    {
                        "role": "user",
                        "content": "Hello"
                    }
                ],
                "temperature": 0.5
            }))
            .unwrap(),
        ),
    )
    .await
    .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
}

#[tokio::test(flavor = "multi_thread")]
async fn test_anthropic_compatible_top_p() {
    let client = tensorzero::test_helpers::make_embedded_gateway().await;
    let state = client.get_app_state_data().unwrap().clone();

    let response = messages_handler(
        State(state),
        None,
        StructuredJson(
            serde_json::from_value(serde_json::json!({
                "model": "tensorzero::function_name::basic_test_no_system_schema",
                "max_tokens": 100,
                "messages": [
                    {
                        "role": "user",
                        "content": "Hello"
                    }
                ],
                "top_p": 0.9
            }))
            .unwrap(),
        ),
    )
    .await
    .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
}

#[tokio::test(flavor = "multi_thread")]
async fn test_anthropic_compatible_top_k() {
    let client = tensorzero::test_helpers::make_embedded_gateway().await;
    let state = client.get_app_state_data().unwrap().clone();

    let response = messages_handler(
        State(state),
        None,
        StructuredJson(
            serde_json::from_value(serde_json::json!({
                "model": "tensorzero::function_name::basic_test_no_system_schema",
                "max_tokens": 100,
                "messages": [
                    {
                        "role": "user",
                        "content": "Hello"
                    }
                ],
                "top_k": 40
            }))
            .unwrap(),
        ),
    )
    .await
    .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
}

#[tokio::test(flavor = "multi_thread")]
async fn test_anthropic_compatible_stop_sequences() {
    let client = tensorzero::test_helpers::make_embedded_gateway().await;
    let state = client.get_app_state_data().unwrap().clone();

    let response = messages_handler(
        State(state),
        None,
        StructuredJson(
            serde_json::from_value(serde_json::json!({
                "model": "tensorzero::function_name::basic_test_no_system_schema",
                "max_tokens": 1000,
                "messages": [
                    {
                        "role": "user",
                        "content": "Count from 1 to 10"
                    }
                ],
                "stop_sequences": ["5"]
            }))
            .unwrap(),
        ),
    )
    .await
    .unwrap();

    assert_eq!(response.status(), StatusCode::OK);

    let response_json = response.into_body().collect().await.unwrap().to_bytes();
    let response_json: Value = serde_json::from_slice(&response_json).unwrap();

    // Check stop_reason
    let stop_reason = response_json.get("stop_reason").unwrap().as_str().unwrap();
    assert_eq!(stop_reason, "stop_sequence");
}

#[tokio::test(flavor = "multi_thread")]
async fn test_anthropic_compatible_usage() {
    let client = tensorzero::test_helpers::make_embedded_gateway().await;
    let state = client.get_app_state_data().unwrap().clone();

    let response = messages_handler(
        State(state),
        None,
        StructuredJson(
            serde_json::from_value(serde_json::json!({
                "model": "tensorzero::function_name::basic_test_no_system_schema",
                "max_tokens": 100,
                "messages": [
                    {
                        "role": "user",
                        "content": "Hello"
                    }
                ]
            }))
            .unwrap(),
        ),
    )
    .await
    .unwrap();

    assert_eq!(response.status(), StatusCode::OK);

    let response_json = response.into_body().collect().await.unwrap().to_bytes();
    let response_json: Value = serde_json::from_slice(&response_json).unwrap();

    // Check usage is present
    let usage = response_json.get("usage").unwrap();
    assert!(usage.get("input_tokens").is_some());
    assert!(usage.get("output_tokens").is_some());

    let input_tokens = usage.get("input_tokens").unwrap().as_u64().unwrap();
    assert!(input_tokens > 0);

    let output_tokens = usage.get("output_tokens").unwrap().as_u64().unwrap();
    assert!(output_tokens > 0);
}

#[tokio::test(flavor = "multi_thread")]
async fn test_anthropic_compatible_models() {
    use reqwest::Client;

    let client = tensorzero::test_helpers::make_embedded_gateway().await;

    let response = Client::new()
        .get(get_gateway_endpoint("/anthropic/v1/models"))
        .send()
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);

    let response_json: Value = response.json().await.unwrap();

    // Check response structure
    assert_eq!(response_json.get("object").unwrap().as_str().unwrap(), "list");

    let data = response_json.get("data").unwrap().as_array().unwrap();
    assert!(!data.is_empty(), "Should return at least one model");

    // Check that each model has the required fields
    for model in data {
        assert!(model.get("id").is_some(), "Model should have 'id' field");
        assert!(model.get("name").is_some(), "Model should have 'name' field");
        assert_eq!(
            model.get("type").unwrap().as_str().unwrap(),
            "model",
            "Model type should be 'model'"
        );
    }

    // Check that function_name models are included
    let has_function_model = data
        .iter()
        .any(|m| {
            m.get("id")
                .and_then(|id| id.as_str())
                .map(|id| id.starts_with("tensorzero::function_name::"))
                .unwrap_or(false)
        });
    assert!(
        has_function_model,
        "Should include at least one function_name model"
    );
}

