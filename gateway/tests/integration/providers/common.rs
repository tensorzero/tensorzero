use futures::StreamExt;
use lazy_static::lazy_static;
use serde_json::json;

use gateway::inference::providers::azure::AzureProvider;
use gateway::inference::providers::provider_trait::InferenceProvider;
use gateway::inference::types::{
    ContentBlock, ContentBlockChunk, FunctionType, JSONMode, ModelInferenceRequest,
    ModelInferenceResponse, RequestMessage, Role,
};
use gateway::jsonschema_util::JSONSchemaFromPath;
use gateway::model::ProviderConfig;
use gateway::tool::{
    StaticToolConfig, ToolCall, ToolCallConfig, ToolChoice, ToolConfig, ToolResult,
};

pub fn create_simple_inference_request<'a>() -> ModelInferenceRequest<'a> {
    let messages = vec![RequestMessage {
        role: Role::User,
        content: vec!["What is the capital of Japan?".to_string().into()],
    }];

    let system = Some("You are a school teacher.".to_string());

    ModelInferenceRequest {
        messages,
        system,
        tool_config: None,
        temperature: Some(0.5),
        max_tokens: Some(100),
        seed: Some(0),
        stream: false,
        json_mode: JSONMode::Off,
        function_type: FunctionType::Chat,
        output_schema: None,
    }
}

pub fn evaluate_simple_inference_request(result: ModelInferenceResponse) {
    assert!(result.content.len() == 1);

    let content = result.content.first().unwrap();

    match content {
        ContentBlock::Text(block) => {
            assert!(block.text.contains("Tokyo"));
        }
        _ => panic!("Expected a text block"),
    }
}

pub async fn test_simple_inference_request_with_provider(provider: ProviderConfig) {
    let inference_request = create_simple_inference_request();
    let client = reqwest::Client::new();
    let result = provider.infer(&inference_request, &client).await.unwrap();
    evaluate_simple_inference_request(result);
}

pub async fn test_streaming_inference_request_with_provider(provider: ProviderConfig) {
    // Set up the inference request
    let mut inference_request = create_simple_inference_request();
    inference_request.stream = true;
    let client = reqwest::Client::new();

    // Run the inference request
    let result = provider
        .infer_stream(&inference_request, &client)
        .await
        .unwrap();

    // Collect the chunks
    let (first_chunk, mut stream) = result;
    let mut collected_chunks = vec![first_chunk];
    while let Some(chunk) = stream.next().await {
        let chunk = chunk.unwrap();
        assert!(chunk.content.len() <= 1);
        collected_chunks.push(chunk);
    }

    // Check the generation
    let generation = collected_chunks
        .iter()
        .filter(|chunk| !chunk.content.is_empty())
        .map(|chunk| chunk.content.first().unwrap())
        .map(|content| match content {
            ContentBlockChunk::Text(block) => block.text.clone(),
            _ => panic!("Expected a text block"),
        })
        .collect::<Vec<String>>()
        .join("");
    assert!(generation.contains("Tokyo"), "{}", generation);

    // Check the usage
    match provider {
        // NOTE: Azure OpenAI service does not return streaming usage and to the best of our knowledge
        // there's no way to get it to do so.
        ProviderConfig::Azure(AzureProvider { .. }) => {
            assert!(collected_chunks.last().unwrap().usage.is_none());
        }
        _ => {
            assert!(collected_chunks.last().unwrap().usage.is_some());
        }
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
    static ref WEATHER_TOOL_CONFIG: StaticToolConfig = StaticToolConfig {
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
    static ref WEATHER_TOOL: ToolConfig = ToolConfig::Static(&WEATHER_TOOL_CONFIG);
    static ref TOOL_CONFIG_SPECIFIC: ToolCallConfig = ToolCallConfig {
        tools_available: vec![ToolConfig::Static(&WEATHER_TOOL_CONFIG)],
        tool_choice: ToolChoice::Tool("get_weather".to_string()),
        parallel_tool_calls: false,
    };
    static ref TOOL_CONFIG_AUTO: ToolCallConfig = ToolCallConfig {
        tools_available: vec![ToolConfig::Static(&WEATHER_TOOL_CONFIG)],
        tool_choice: ToolChoice::Auto,
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
        max_tokens: Some(100),
        seed: None,
        stream: false,
        json_mode: JSONMode::Off,
        function_type: FunctionType::Chat,
        output_schema: None,
    }
}

pub fn create_streaming_tool_inference_request() -> ModelInferenceRequest<'static> {
    let mut request = create_tool_inference_request();
    request.stream = true;
    request
}

pub fn create_streaming_tool_result_inference_request() -> ModelInferenceRequest<'static> {
    let mut request = create_tool_result_inference_request();
    request.stream = true;
    request
}
