#![cfg(feature = "integration_tests")]
#![cfg(test)]
use api::inference::types::{
    FunctionType, InferenceRequestMessage, ModelInferenceRequest, SystemInferenceRequestMessage,
    Tool, ToolChoice, ToolType, UserInferenceRequestMessage,
};
use serde_json::json;

pub fn create_simple_inference_request() -> ModelInferenceRequest {
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
    ModelInferenceRequest {
        messages,
        tools_available: None,
        tool_choice: None,
        parallel_tool_calls: None,
        temperature,
        max_tokens,
        stream: false,
        json_mode: false,
        function_type: FunctionType::Chat,
        output_schema: None,
    }
}

pub fn create_json_inference_request() -> ModelInferenceRequest {
    let messages = vec![
        InferenceRequestMessage::System(SystemInferenceRequestMessage {
            content: "You are a helpful but mischevious assistant who returns in the JSON form {\"thinking\": \"...\", \"answer\": \"...\"}".to_string(),
        }),
        InferenceRequestMessage::User(UserInferenceRequestMessage {
            content: "Is Santa Clause real? be brief".to_string(),
        }),
    ];
    let max_tokens = Some(400);
    let temperature = Some(1.);
    let output_schema = json!({
        "type": "object",
        "properties": {
            "thinking": {"type": "string"},
            "answer": {"type": "string"}
        },
        "required": ["thinking", "answer"]
    });
    ModelInferenceRequest {
        messages,
        tools_available: None,
        tool_choice: None,
        parallel_tool_calls: None,
        temperature,
        max_tokens,
        stream: false,
        json_mode: false,
        function_type: FunctionType::Chat,
        output_schema: Some(output_schema),
    }
}

pub fn create_tool_inference_request() -> ModelInferenceRequest {
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

    ModelInferenceRequest {
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
    }
}

pub fn create_streaming_inference_request() -> ModelInferenceRequest {
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
    ModelInferenceRequest {
        messages,
        tools_available: None,
        tool_choice: None,
        parallel_tool_calls: None,
        temperature,
        max_tokens,
        stream: true,
        json_mode: false,
        function_type: FunctionType::Chat,
        output_schema: None,
    }
}
