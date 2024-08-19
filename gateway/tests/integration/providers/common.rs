use gateway::inference::types::{
    ContentBlock, FunctionType, JSONMode, ModelInferenceRequest, RequestMessage, Role,
};
use gateway::jsonschema_util::JSONSchemaFromPath;
use gateway::tool::{ToolCall, ToolCallConfig, ToolChoice, ToolConfig, ToolResult};
use lazy_static::lazy_static;
use serde_json::json;

pub fn create_simple_inference_request<'a>() -> ModelInferenceRequest<'a> {
    let messages = vec![RequestMessage {
        role: Role::User,
        content: vec!["Is Santa Clause real?".to_string().into()],
    }];
    let system = Some("You are a helpful but mischevious assistant.".to_string());
    let max_tokens = Some(100);
    let temperature = Some(1.);
    let seed = Some(69);
    ModelInferenceRequest {
        messages,
        system,
        tool_config: None,
        temperature,
        max_tokens,
        seed,
        stream: false,
        json_mode: JSONMode::Off,
        function_type: FunctionType::Chat,
        output_schema: None,
    }
}

pub fn create_json_inference_request<'a>() -> ModelInferenceRequest<'a> {
    let messages = vec![RequestMessage {
        role: Role::User,
        content: vec!["Is Santa Clause real? Be concise.".to_string().into()],
    }];
    let system = Some(
        r#"\
        # Instructions\n\
        You are a helpful but mischevious assistant who returns in the JSON form {"honest_answer": "...", "mischevious_answer": "..."}.\n\
        \n\
        # Examples\n\
        \n\
        {"honest_answer": "Yes.", "mischevious_answer": "No way."}\n\
        \n\
        {"honest_answer": "No.", "mischevious_answer": "Of course!"}\n\
        \n\
        {"honest_answer": "No idea.", "mischevious_answer": "Definitely."}\n\
        \n\
        {"honest_answer": "Frequently.", "mischevious_answer": "Never."}\n\
        "#.into(),
    );
    let max_tokens = Some(400);
    let temperature = Some(1.);
    let seed = Some(420);
    let output_schema = json!({
        "type": "object",
        "properties": {
            "honest_answer": {"type": "string"},
            "mischevious_answer": {"type": "string"}
        },
        "required": ["honest_answer", "mischevious_answer"]
    });
    ModelInferenceRequest {
        messages,
        system,
        tool_config: None,
        temperature,
        max_tokens,
        seed,
        stream: false,
        json_mode: JSONMode::On,
        function_type: FunctionType::Json,
        output_schema: Some(Box::leak(Box::new(output_schema))),
    }
}

pub fn create_streaming_json_inference_request<'a>() -> ModelInferenceRequest<'a> {
    let mut request = create_json_inference_request();
    request.stream = true;
    request
}

lazy_static! {
    static ref WEATHER_TOOL: ToolConfig = ToolConfig {
        name: "get_weather".to_string(),
        description: "Get the current weather in a given location".to_string(),
        parameters: JSONSchemaFromPath::from_value(&json!({
            "type": "object",
            "properties": {
                "location": {"type": "string"},
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
            },
            "required": ["location"]
        }))
    };
    static ref TOOL_CHOICE_SPECIFIC: ToolChoice = ToolChoice::Tool("get_weather".to_string());
    static ref TOOL_CONFIG_SPECIFIC: ToolCallConfig = ToolCallConfig {
        tools_available: vec![&*WEATHER_TOOL],
        tool_choice: &TOOL_CHOICE_SPECIFIC,
        parallel_tool_calls: false,
    };
    static ref TOOL_CHOICE_AUTO: ToolChoice = ToolChoice::Auto;
    static ref TOOL_CONFIG_AUTO: ToolCallConfig = ToolCallConfig {
        tools_available: vec![&*WEATHER_TOOL],
        tool_choice: &TOOL_CHOICE_AUTO,
        parallel_tool_calls: false,
    };
}

pub fn create_tool_inference_request() -> ModelInferenceRequest<'static> {
    let messages = vec![RequestMessage {
        role: Role::User,
        content: vec!["What's the weather like in New York currently?"
            .to_string()
            .into()],
    }];

    ModelInferenceRequest {
        messages,
        system: None,
        tool_config: Some(&*TOOL_CONFIG_SPECIFIC),
        temperature: Some(0.7),
        max_tokens: Some(300),
        seed: None,
        stream: false,
        json_mode: JSONMode::Off,
        function_type: FunctionType::Chat,
        output_schema: None,
    }
}

pub fn create_tool_result_inference_request() -> ModelInferenceRequest<'static> {
    let messages = vec![
        RequestMessage {
            role: Role::User,
            content: vec!["What's the weather like in New York currently?"
                .to_string()
                .into()],
        },
        RequestMessage {
            role: Role::Assistant,
            content: vec![ContentBlock::ToolCall(ToolCall {
                name: "get_weather".to_string(),
                arguments: "{\"location\": \"New York\"}".to_string(),
                id: "1".to_string(),
            })],
        },
        RequestMessage {
            role: Role::User,
            content: vec![ContentBlock::ToolResult(ToolResult {
                name: "get_weather".to_string(),
                id: "1".to_string(),
                result: "70".to_string(),
            })],
        },
    ];

    ModelInferenceRequest {
        messages,
        system: None,
        tool_config: Some(&*TOOL_CONFIG_AUTO),
        temperature: Some(0.7),
        max_tokens: Some(300),
        seed: None,
        stream: false,
        json_mode: JSONMode::Off,
        function_type: FunctionType::Chat,
        output_schema: None,
    }
}

pub fn create_streaming_inference_request<'a>() -> ModelInferenceRequest<'a> {
    let messages = vec![RequestMessage {
        role: Role::User,
        content: vec!["Is Santa Clause real?".to_string().into()],
    }];
    let system = Some("You are a helpful but mischevious assistant.".to_string());
    let max_tokens = Some(100);
    let temperature = Some(1.);
    let seed = Some(69);
    ModelInferenceRequest {
        messages,
        system,
        tool_config: None,
        temperature,
        max_tokens,
        seed,
        stream: true,
        json_mode: JSONMode::Off,
        function_type: FunctionType::Chat,
        output_schema: None,
    }
}
