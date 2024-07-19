#![cfg(feature = "integration_tests")]
#![cfg(test)]
use api::inference::providers::anthropic;
use api::inference::types::{
    FunctionType, InferenceRequestMessage, ModelInferenceRequest, SystemInferenceRequestMessage,
    Tool, ToolChoice, ToolType, UserInferenceRequestMessage,
};
use futures::StreamExt;
use secrecy::SecretString;
use serde_json::json;
use std::env;

#[tokio::test]
async fn test_infer() {
    // Load API key from environment variable
    let api_key = env::var("ANTHROPIC_API_KEY").expect("ANTHROPIC_API_KEY must be set");
    let api_key = SecretString::new(api_key);
    let model_name = "claude-3-haiku-20240307";
    let client = reqwest::Client::new();
    let messages = vec![
        InferenceRequestMessage::System(SystemInferenceRequestMessage {
            content: "You are a helpful but mischevious assistant.".to_string(),
        }),
        InferenceRequestMessage::User(UserInferenceRequestMessage {
            content: "Is Santa Clause real?".to_string(),
        }),
    ];
    let max_tokens = Some(100);
    let temperature = Some(1.);
    let inference_request = ModelInferenceRequest {
        messages: messages.clone(),
        tools_available: None,
        tool_choice: None,
        parallel_tool_calls: None,
        temperature,
        max_tokens,
        stream: false,
        json_mode: false,
        function_type: FunctionType::Chat,
        output_schema: None,
    };

    let result = anthropic::infer(inference_request, model_name, &client, &api_key).await;
    assert!(result.is_ok());
    assert!(result.unwrap().content.is_some());
}

#[tokio::test]
async fn test_infer_stream() {
    // Load API key from environment variable
    let api_key = env::var("ANTHROPIC_API_KEY").expect("ANTHROPIC_API_KEY must be set");
    let api_key = SecretString::new(api_key);
    let model_name = "claude-3-haiku-20240307";
    let client = reqwest::Client::new();
    let messages = vec![
        InferenceRequestMessage::System(SystemInferenceRequestMessage {
            content: "You are a helpful but mischevious assistant.".to_string(),
        }),
        InferenceRequestMessage::User(UserInferenceRequestMessage {
            content: "Is Santa Clause real?".to_string(),
        }),
    ];
    let max_tokens = Some(100);
    let temperature = Some(1.);
    let inference_request = ModelInferenceRequest {
        messages: messages.clone(),
        tools_available: None,
        tool_choice: None,
        parallel_tool_calls: None,
        temperature,
        max_tokens,
        stream: true,
        json_mode: false,
        function_type: FunctionType::Chat,
        output_schema: None,
    };

    let result = anthropic::infer_stream(inference_request, model_name, &client, &api_key).await;
    assert!(result.is_ok());
    let mut stream = result.unwrap();
    let mut collected_chunks = Vec::new();
    while let Some(chunk) = stream.next().await {
        assert!(chunk.is_ok());
        collected_chunks.push(chunk.unwrap());
    }
    assert!(!collected_chunks.is_empty());
    assert!(collected_chunks.last().unwrap().content.is_some());
}

#[tokio::test]
async fn test_infer_with_tool_calls() {
    // Load API key from environment variable
    let api_key = env::var("ANTHROPIC_API_KEY").expect("ANTHROPIC_API_KEY must be set");
    let api_key = SecretString::new(api_key);
    let model_name = "claude-3-haiku-20240307";
    let client = reqwest::Client::new();

    // Define a tool
    let tool = Tool {
        r#type: ToolType::Function,
        description: Some("Get the current weather in a given location".to_string()),
        name: "get_weather".to_string(),
        parameters: json!({
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "The city and state, e.g. San Francisco, CA"
                },
                "unit": {
                    "type": "string",
                    "enum": ["celsius", "fahrenheit"]
                }
            },
            "required": ["location"]
        }),
    };

    let messages = vec![InferenceRequestMessage::User(UserInferenceRequestMessage {
        content: "What's the weather like in New York?".to_string(),
    })];

    let inference_request = ModelInferenceRequest {
        messages,
        tools_available: Some(vec![tool]),
        tool_choice: Some(ToolChoice::Tool("get_weather".to_string())),
        parallel_tool_calls: None,
        temperature: Some(0.7),
        max_tokens: Some(300),
        stream: false,
        json_mode: false,
        function_type: FunctionType::Tool,
        output_schema: None,
    };

    let result = anthropic::infer(inference_request, model_name, &client, &api_key).await;

    assert!(result.is_ok());
    let response = result.unwrap();
    assert!(response.tool_calls.is_some());
    let tool_calls = response.tool_calls.unwrap();
    assert!(!tool_calls.is_empty());

    let first_tool_call = &tool_calls[0];
    assert_eq!(first_tool_call.name, "get_weather");

    // Parse the arguments to ensure they're valid JSON
    let arguments: serde_json::Value = serde_json::from_str(&first_tool_call.arguments)
        .expect("Failed to parse tool call arguments");
    assert!(arguments.get("location").is_some());
}
