//! Tests multi-turn conversation with parallel tool calls.
//!
//! This test verifies that:
//! 1. The model can make parallel tool calls (get_temperature + get_humidity)
//! 2. Tool results can be sent back in a single user message
//! 3. The model responds correctly with the tool results

use crate::providers::common::E2ETestProvider;
use crate::providers::commonv2::models::MODELS_CONFIG;
use serde_json::Value;
use tensorzero::test_helpers::make_embedded_gateway_with_config;
use tensorzero::{
    Client, ClientInferenceParams, InferenceOutput, InferenceResponse, InferenceResponseChunk,
    Input, InputMessage, InputMessageContent, ToolCallWrapper,
};
use tensorzero_core::db::clickhouse::test_helpers::{
    get_clickhouse, select_chat_inference_clickhouse, select_model_inference_clickhouse,
};
use tensorzero_core::inference::types::{
    ContentBlockChatOutput, Role, StoredContentBlock, StoredRequestMessage, Text,
};
use tensorzero_core::tool::{InferenceResponseToolCall, ToolResult};
use tokio_stream::StreamExt;
use uuid::Uuid;

const GET_TEMPERATURE_PARAMS: &str = r#"{
    "type": "object",
    "properties": {
        "location": {
            "type": "string",
            "description": "The location to get the temperature for (e.g. \"New York\")"
        },
        "units": {
            "type": "string",
            "description": "The units to get the temperature in (must be \"fahrenheit\" or \"celsius\")",
            "enum": ["fahrenheit", "celsius"]
        }
    },
    "required": ["location"],
    "additionalProperties": false
}"#;

const GET_HUMIDITY_PARAMS: &str = r#"{
    "type": "object",
    "properties": {
        "location": {
            "type": "string",
            "description": "The location to get the humidity for (e.g. \"New York\")"
        }
    },
    "required": ["location"],
    "additionalProperties": false
}"#;

const SYSTEM_TEMPLATE: &str = r#"You are a helpful and friendly assistant named Dr. Mehta.
People will ask you questions about the weather.
When asked about the weather, you MUST use BOTH the "get_temperature" and "get_humidity" tools to get the information.
After receiving tool results, you MUST respond to the user with a natural language summary of the weather (e.g. "The weather in New York is 55 degrees Fahrenheit with 50% humidity.").
"#;

const USER_MESSAGE: &str = "What is the weather like in Tokyo (in Fahrenheit)?";

struct TestSetup {
    client: Client,
    episode_id: Uuid,
    #[expect(dead_code)]
    temp_dir: tempfile::TempDir,
}

async fn setup_test(provider: &E2ETestProvider) -> TestSetup {
    let episode_id = Uuid::now_v7();

    // Create temp dir and write tool parameter files and system template
    let temp_dir = tempfile::tempdir().unwrap();
    let get_temp_path = temp_dir.path().join("get_temperature.json");
    let get_humidity_path = temp_dir.path().join("get_humidity.json");
    let system_template_path = temp_dir.path().join("system_template.minijinja");

    std::fs::write(&get_temp_path, GET_TEMPERATURE_PARAMS).unwrap();
    std::fs::write(&get_humidity_path, GET_HUMIDITY_PARAMS).unwrap();
    std::fs::write(&system_template_path, SYSTEM_TEMPLATE).unwrap();

    // Format config with provider's model and temp file paths
    let config = format!(
        r#"
[functions.multi_turn_parallel_tool_test]
type = "chat"
tools = ["get_temperature", "get_humidity"]
tool_choice = "auto"
parallel_tool_calls = true

[functions.multi_turn_parallel_tool_test.variants.{variant_name}]
type = "chat_completion"
model = "{model}"
system_template = "{system_template_path}"

[tools.get_temperature]
description = "Get the current temperature in a given location"
parameters = "{get_temp_params_path}"

[tools.get_humidity]
description = "Get the current humidity in a given location"
parameters = "{get_humidity_params_path}"

{MODELS_CONFIG}
"#,
        variant_name = provider.variant_name,
        model = provider.model_name,
        system_template_path = system_template_path.display(),
        get_temp_params_path = get_temp_path.display(),
        get_humidity_params_path = get_humidity_path.display(),
        MODELS_CONFIG = MODELS_CONFIG,
    );

    let client = make_embedded_gateway_with_config(&config).await;

    TestSetup {
        client,
        episode_id,
        temp_dir,
    }
}

struct FirstInferenceResult {
    tool_results: Vec<InputMessageContent>,
    assistant_content: Vec<InputMessageContent>,
}

async fn do_first_inference(
    client: &Client,
    provider: &E2ETestProvider,
    episode_id: Uuid,
) -> FirstInferenceResult {
    let response = client
        .inference(ClientInferenceParams {
            function_name: Some("multi_turn_parallel_tool_test".into()),
            variant_name: Some(provider.variant_name.clone()),
            episode_id: Some(episode_id),
            input: Input {
                system: None,
                messages: vec![InputMessage {
                    role: Role::User,
                    content: vec![InputMessageContent::Text(Text {
                        text: USER_MESSAGE.into(),
                    })],
                }],
            },
            ..Default::default()
        })
        .await
        .unwrap();

    let InferenceOutput::NonStreaming(response) = response else {
        panic!("Expected non-streaming inference response");
    };
    let InferenceResponse::Chat(chat) = response else {
        panic!("Expected chat inference response");
    };

    println!("First response: {chat:#?}");

    let tool_calls: Vec<&InferenceResponseToolCall> = chat
        .content
        .iter()
        .filter_map(|block| match block {
            ContentBlockChatOutput::ToolCall(tc) => Some(tc),
            _ => None,
        })
        .collect();

    assert!(
        tool_calls.len() >= 2,
        "Expected at least 2 tool calls, got {}",
        tool_calls.len()
    );

    let tool_results: Vec<InputMessageContent> = tool_calls
        .iter()
        .map(|tc| {
            let result = match tc.raw_name.as_str() {
                "get_temperature" => "70",
                "get_humidity" => "30",
                name => panic!("Unknown tool: {name}"),
            };
            InputMessageContent::ToolResult(ToolResult {
                id: tc.id.clone(),
                name: tc.raw_name.clone(),
                result: result.into(),
            })
        })
        .collect();

    let assistant_content: Vec<InputMessageContent> = tool_calls
        .iter()
        .map(|tc| {
            InputMessageContent::ToolCall(ToolCallWrapper::InferenceResponseToolCall((*tc).clone()))
        })
        .collect();

    FirstInferenceResult {
        tool_results,
        assistant_content,
    }
}

fn build_second_inference_input(
    assistant_content: Vec<InputMessageContent>,
    tool_results: Vec<InputMessageContent>,
) -> Input {
    Input {
        system: None,
        messages: vec![
            InputMessage {
                role: Role::User,
                content: vec![InputMessageContent::Text(Text {
                    text: USER_MESSAGE.into(),
                })],
            },
            InputMessage {
                role: Role::Assistant,
                content: assistant_content,
            },
            InputMessage {
                role: Role::User,
                content: tool_results, // ALL tool results in ONE message
            },
        ],
    }
}

async fn verify_clickhouse_chat_inference(
    inference_id: Uuid,
    episode_id: Uuid,
    provider: &E2ETestProvider,
) {
    let clickhouse = get_clickhouse().await;
    let result = select_chat_inference_clickhouse(&clickhouse, inference_id)
        .await
        .unwrap();

    println!("ClickHouse - ChatInference: {result:#?}");

    let id = result.get("id").unwrap().as_str().unwrap();
    let id_uuid = Uuid::parse_str(id).unwrap();
    assert_eq!(id_uuid, inference_id);

    let function_name = result.get("function_name").unwrap().as_str().unwrap();
    assert_eq!(function_name, "multi_turn_parallel_tool_test");

    let variant_name = result.get("variant_name").unwrap().as_str().unwrap();
    assert_eq!(variant_name, provider.variant_name);

    let episode_id_result = result.get("episode_id").unwrap().as_str().unwrap();
    let episode_id_result = Uuid::parse_str(episode_id_result).unwrap();
    assert_eq!(episode_id_result, episode_id);

    let input: Value =
        serde_json::from_str(result.get("input").unwrap().as_str().unwrap()).unwrap();

    let last_input_message = input["messages"].as_array().unwrap().last().unwrap();
    assert_eq!(last_input_message["role"], "user");
    let last_input_message_content = last_input_message["content"].as_array().unwrap();
    assert_eq!(last_input_message_content.len(), 2);
    for tool_result in last_input_message_content {
        assert_eq!(tool_result["type"], "tool_result");
    }

    let tool_params: Value =
        serde_json::from_str(result.get("tool_params").unwrap().as_str().unwrap()).unwrap();
    assert_eq!(tool_params["tool_choice"], "auto");
    assert_eq!(tool_params["parallel_tool_calls"], true);
}

async fn verify_clickhouse_model_inference(
    inference_id: Uuid,
    provider: &E2ETestProvider,
    is_streaming: bool,
) {
    let clickhouse = get_clickhouse().await;
    let result = select_model_inference_clickhouse(&clickhouse, inference_id)
        .await
        .unwrap();

    println!("ClickHouse - ModelInference: {result:#?}");

    let id = result.get("id").unwrap().as_str().unwrap();
    assert!(Uuid::parse_str(id).is_ok());

    let inference_id_result = result.get("inference_id").unwrap().as_str().unwrap();
    let inference_id_result = Uuid::parse_str(inference_id_result).unwrap();
    assert_eq!(inference_id_result, inference_id);

    let model_name = result.get("model_name").unwrap().as_str().unwrap();
    assert_eq!(model_name, provider.model_name);
    let model_provider_name = result.get("model_provider_name").unwrap().as_str().unwrap();
    assert_eq!(model_provider_name, provider.model_provider_name);

    let raw_request = result.get("raw_request").unwrap().as_str().unwrap();
    assert!(
        serde_json::from_str::<Value>(raw_request).is_ok(),
        "raw_request is not a valid JSON"
    );

    let raw_response = result.get("raw_response").unwrap().as_str().unwrap();
    if is_streaming {
        // For streaming, raw_response is JSONL
        for line in raw_response.lines() {
            assert!(
                serde_json::from_str::<Value>(line).is_ok(),
                "raw_response line is not valid JSON: {line}"
            );
        }
    } else {
        assert!(raw_response.to_lowercase().contains("70"));
        assert!(raw_response.to_lowercase().contains("30"));
    }

    let input_tokens = result.get("input_tokens").unwrap();
    let output_tokens = result.get("output_tokens").unwrap();
    if is_streaming {
        // Some providers don't return tokens during streaming
        if !input_tokens.is_null() {
            assert!(input_tokens.as_u64().unwrap() > 0);
        }
        if !output_tokens.is_null() {
            assert!(output_tokens.as_u64().unwrap() > 0);
        }
    } else {
        assert!(input_tokens.as_u64().unwrap() > 0);
        assert!(output_tokens.as_u64().unwrap() > 0);
    }

    let response_time_ms = result.get("response_time_ms").unwrap().as_u64().unwrap();
    assert!(response_time_ms > 0);

    if is_streaming {
        let ttft_ms = result.get("ttft_ms").unwrap().as_u64().unwrap();
        assert!(ttft_ms >= 1);
        assert!(ttft_ms <= response_time_ms);
    } else {
        assert!(result.get("ttft_ms").unwrap().is_null());
    }

    let system = result.get("system").unwrap().as_str().unwrap();
    assert!(system.contains("Dr. Mehta"));
    assert!(system.contains("get_temperature"));
    assert!(system.contains("get_humidity"));

    let input_messages = result.get("input_messages").unwrap().as_str().unwrap();
    let input_messages: Vec<StoredRequestMessage> = serde_json::from_str(input_messages).unwrap();
    let last_input_message = input_messages.last().unwrap();
    assert_eq!(last_input_message.role, Role::User);
    let last_input_message_content = &last_input_message.content;
    assert_eq!(last_input_message_content.len(), 2);
    for tool_result in last_input_message_content {
        match tool_result {
            StoredContentBlock::ToolResult(tool_result) => {
                assert!(
                    tool_result.name == "get_temperature" || tool_result.name == "get_humidity"
                );
            }
            _ => {
                panic!("Expected a tool result, got {tool_result:?}");
            }
        }
    }

    let output = result.get("output").unwrap().as_str().unwrap();
    let output: Vec<StoredContentBlock> = serde_json::from_str(output).unwrap();
    assert_eq!(output.len(), 1);
    let output_content = output.first().unwrap();
    match output_content {
        StoredContentBlock::Text(text) => {
            assert!(text.text.to_lowercase().contains("70"));
            assert!(text.text.to_lowercase().contains("30"));
        }
        _ => {
            panic!("Expected a text block, got {output_content:?}");
        }
    }
}

pub async fn test_multi_turn_parallel_tool_use_with_provider(provider: E2ETestProvider) {
    let setup = setup_test(&provider).await;
    let first_result = do_first_inference(&setup.client, &provider, setup.episode_id).await;

    // Second inference with tool results - non-streaming
    let response2 = setup
        .client
        .inference(ClientInferenceParams {
            function_name: Some("multi_turn_parallel_tool_test".into()),
            variant_name: Some(provider.variant_name.clone()),
            episode_id: Some(setup.episode_id),
            input: build_second_inference_input(
                first_result.assistant_content,
                first_result.tool_results,
            ),
            ..Default::default()
        })
        .await
        .unwrap();

    let InferenceOutput::NonStreaming(response2) = response2 else {
        panic!("Expected non-streaming inference response");
    };
    let InferenceResponse::Chat(chat2) = response2 else {
        panic!("Expected chat inference response");
    };

    println!("Second response: {chat2:#?}");

    let inference_id = chat2.inference_id;

    let text: String = chat2
        .content
        .iter()
        .filter_map(|b| match b {
            ContentBlockChatOutput::Text(t) => Some(t.text.as_str()),
            _ => None,
        })
        .collect();

    assert!(
        text.contains("70") && text.contains("30"),
        "Expected response to contain '70' and '30', got: {text}"
    );

    // Sleep to allow time for data to be inserted into ClickHouse
    tokio::time::sleep(std::time::Duration::from_millis(200)).await;

    verify_clickhouse_chat_inference(inference_id, setup.episode_id, &provider).await;
    verify_clickhouse_model_inference(inference_id, &provider, false).await;
}

pub async fn test_multi_turn_parallel_tool_use_streaming_with_provider(provider: E2ETestProvider) {
    let setup = setup_test(&provider).await;
    let first_result = do_first_inference(&setup.client, &provider, setup.episode_id).await;

    // Second inference with tool results - STREAMING
    let response2 = setup
        .client
        .inference(ClientInferenceParams {
            function_name: Some("multi_turn_parallel_tool_test".into()),
            variant_name: Some(provider.variant_name.clone()),
            episode_id: Some(setup.episode_id),
            stream: Some(true),
            input: build_second_inference_input(
                first_result.assistant_content,
                first_result.tool_results,
            ),
            ..Default::default()
        })
        .await
        .unwrap();

    let InferenceOutput::Streaming(mut stream) = response2 else {
        panic!("Expected streaming inference response");
    };

    let mut inference_id = None;
    let mut output_content = String::new();

    while let Some(chunk) = stream.next().await {
        let InferenceResponseChunk::Chat(response) = chunk.unwrap() else {
            panic!("Expected chat response chunk");
        };

        println!("Chunk: {response:#?}");

        inference_id = Some(response.inference_id);
        assert_eq!(response.episode_id, setup.episode_id);

        for block in &response.content {
            if let tensorzero::ContentBlockChunk::Text(text_chunk) = block {
                output_content.push_str(&text_chunk.text);
            }
        }
    }

    let inference_id = inference_id.expect("Should have received at least one chunk");

    println!("Output content: {output_content:#?}");

    assert!(
        output_content.contains("70") && output_content.contains("30"),
        "Expected response to contain '70' and '30', got: {output_content}"
    );

    // Sleep to allow time for data to be inserted into ClickHouse
    tokio::time::sleep(std::time::Duration::from_millis(200)).await;

    verify_clickhouse_chat_inference(inference_id, setup.episode_id, &provider).await;
    verify_clickhouse_model_inference(inference_id, &provider, true).await;
}
