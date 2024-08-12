use gateway::inference::types::{
    FunctionType, JSONMode, ModelInferenceRequest, RequestMessage, Role, Tool, ToolChoice,
};
use serde_json::json;

pub fn create_simple_inference_request<'a>() -> ModelInferenceRequest<'a> {
    let messages = vec![RequestMessage {
        role: Role::User,
        content: vec!["Is Santa Clause real?".to_string().into()],
    }];
    let system = Some("You are a helpful but mischevious assistant.".to_string());
    let max_tokens = Some(100);
    let temperature = Some(1.);
    ModelInferenceRequest {
        messages,
        system,
        tools_available: None,
        tool_choice: ToolChoice::None,
        parallel_tool_calls: None,
        temperature,
        max_tokens,
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
        tools_available: None,
        tool_choice: ToolChoice::None,
        parallel_tool_calls: None,
        temperature,
        max_tokens,
        stream: false,
        json_mode: JSONMode::On,
        function_type: FunctionType::Chat,
        output_schema: Some(Box::leak(Box::new(output_schema))),
    }
}

pub fn create_tool_inference_request<'a>() -> ModelInferenceRequest<'a> {
    // Define a tool
    let tool = Tool::Function {
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

    let messages = vec![RequestMessage {
        role: Role::User,
        content: vec!["What's the weather like in New York currently?"
            .to_string()
            .into()],
    }];

    ModelInferenceRequest {
        messages,
        system: None,
        tools_available: Some(vec![tool]),
        tool_choice: ToolChoice::Tool("get_weather".to_string()),
        parallel_tool_calls: None,
        temperature: Some(0.7),
        max_tokens: Some(300),
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
    ModelInferenceRequest {
        messages,
        system,
        tools_available: None,
        tool_choice: ToolChoice::None,
        parallel_tool_calls: None,
        temperature,
        max_tokens,
        stream: true,
        json_mode: JSONMode::Off,
        function_type: FunctionType::Chat,
        output_schema: None,
    }
}
