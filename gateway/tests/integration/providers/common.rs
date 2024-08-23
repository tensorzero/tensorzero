#![allow(clippy::print_stdout)]

use futures::StreamExt;
use lazy_static::lazy_static;
use serde_json::json;

use gateway::inference::providers::azure::AzureProvider;
use gateway::inference::providers::provider_trait::InferenceProvider;
use gateway::inference::providers::together::TogetherProvider;
use gateway::inference::types::{
    ContentBlock, ContentBlockChunk, FunctionType, JSONMode, ModelInferenceRequest, RequestMessage,
    Role,
};
use gateway::jsonschema_util::JSONSchemaFromPath;
use gateway::model::ProviderConfig;
use gateway::tool::{
    StaticToolConfig, ToolCall, ToolCallConfig, ToolChoice, ToolConfig, ToolResult,
};

/// Enforce that every provider implements a common set of tests.
///
/// To achieve that, each provider should call the `generate_provider_tests!` macro along with a
/// function that returns a `IntegrationTestProviders` struct.
///
/// If some test doesn't apply to a particular provider (e.g. provider doesn't support tool use),
/// then the provider should return an empty vector for the corresponding test.
pub struct IntegrationTestProviders {
    pub simple_inference: Vec<&'static ProviderConfig>,
    pub simple_streaming_inference: Vec<&'static ProviderConfig>,
    pub tool_use_inference: Vec<&'static ProviderConfig>,
    pub tool_use_streaming_inference: Vec<&'static ProviderConfig>,
    pub tool_multi_turn_inference: Vec<&'static ProviderConfig>,
    pub tool_multi_turn_streaming_inference: Vec<&'static ProviderConfig>,
    pub json_mode_inference: Vec<&'static ProviderConfig>,
    pub json_mode_streaming_inference: Vec<&'static ProviderConfig>,
}

impl IntegrationTestProviders {
    pub fn with_provider(provider: ProviderConfig) -> Self {
        let provider = Box::leak(Box::new(provider));

        Self {
            simple_inference: vec![provider],
            simple_streaming_inference: vec![provider],
            tool_use_inference: vec![provider],
            tool_use_streaming_inference: vec![provider],
            tool_multi_turn_inference: vec![provider],
            tool_multi_turn_streaming_inference: vec![provider],
            json_mode_inference: vec![provider],
            json_mode_streaming_inference: vec![provider],
        }
    }

    pub fn with_providers(providers: Vec<ProviderConfig>) -> Self {
        let mut static_providers: Vec<&'static ProviderConfig> = vec![];

        for provider in providers {
            let static_provider = Box::leak(Box::new(provider));
            static_providers.push(static_provider);
        }

        Self {
            simple_inference: static_providers.clone(),
            simple_streaming_inference: static_providers.clone(),
            tool_use_inference: static_providers.clone(),
            tool_use_streaming_inference: static_providers.clone(),
            tool_multi_turn_inference: static_providers.clone(),
            tool_multi_turn_streaming_inference: static_providers.clone(),
            json_mode_inference: static_providers.clone(),
            json_mode_streaming_inference: static_providers,
        }
    }
}

#[macro_export]
macro_rules! generate_provider_tests {
    ($func:ident) => {
        use $crate::providers::common::test_json_mode_inference_request_with_provider;
        use $crate::providers::common::test_json_mode_streaming_inference_request_with_provider;
        use $crate::providers::common::test_simple_inference_request_with_provider;
        use $crate::providers::common::test_simple_streaming_inference_request_with_provider;
        use $crate::providers::common::test_tool_multi_turn_inference_request_with_provider;
        use $crate::providers::common::test_tool_multi_turn_streaming_inference_request_with_provider;
        use $crate::providers::common::test_tool_use_inference_request_with_provider;
        use $crate::providers::common::test_tool_use_streaming_inference_request_with_provider;

        #[tokio::test]
        async fn test_simple_inference_request() {
            let providers = $func().await.simple_inference;
            for provider in providers {
                test_simple_inference_request_with_provider(provider).await;
            }
        }

        #[tokio::test]
        async fn test_simple_streaming_inference_request() {
            let providers = $func().await.simple_streaming_inference;
            for provider in providers {
                test_simple_streaming_inference_request_with_provider(provider).await;
            }
        }

        #[tokio::test]
        async fn test_tool_use_inference_request() {
            let providers = $func().await.tool_use_inference;
            for provider in providers {
                test_tool_use_inference_request_with_provider(provider).await;
            }
        }

        #[tokio::test]
        async fn test_tool_use_streaming_inference_request() {
            let providers = $func().await.tool_use_streaming_inference;
            for provider in providers {
                test_tool_use_streaming_inference_request_with_provider(provider).await;
            }
        }

        #[tokio::test]
        async fn test_json_mode_inference_request() {
            let providers = $func().await.json_mode_inference;
            for provider in providers {
                test_json_mode_inference_request_with_provider(provider).await;
            }
        }

        #[tokio::test]
        async fn test_json_mode_streaming_inference_request() {
            let providers = $func().await.json_mode_streaming_inference;
            for provider in providers {
                test_json_mode_streaming_inference_request_with_provider(provider).await;
            }
        }

        #[tokio::test]
        async fn test_tool_multi_turn_inference_request() {
            let providers = $func().await.tool_multi_turn_inference;
            for provider in providers {
                test_tool_multi_turn_inference_request_with_provider(provider).await;
            }
        }

        #[tokio::test]
        async fn test_tool_multi_turn_streaming_inference_request() {
            let providers = $func().await.tool_multi_turn_streaming_inference;
            for provider in providers {
                test_tool_multi_turn_streaming_inference_request_with_provider(provider).await;
            }
        }
    };
}

fn create_simple_inference_request<'a>() -> ModelInferenceRequest<'a> {
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

pub async fn test_simple_inference_request_with_provider(provider: &ProviderConfig) {
    // Set up the inference request
    let inference_request = create_simple_inference_request();
    let client = reqwest::Client::new();

    // Run the inference request
    let result = provider.infer(&inference_request, &client).await.unwrap();

    println!("Result: {result:#?}");

    // Evaluate the results
    assert!(result.content.len() == 1);

    let content = result.content.first().unwrap();

    match content {
        ContentBlock::Text(block) => {
            assert!(block.text.contains("Tokyo"));
        }
        _ => panic!("Expected a text block"),
    }
}

pub async fn test_simple_streaming_inference_request_with_provider(provider: &ProviderConfig) {
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
        println!("Chunk: {chunk:#?}");
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

    println!("Generation: {}", generation);

    assert!(generation.contains("Tokyo"));

    // Check the usage
    match provider {
        // NOTE: Azure does not return usage for streaming inference (to the best of our knowledge)
        ProviderConfig::Azure(AzureProvider { .. }) => {
            assert!(collected_chunks.last().unwrap().usage.is_none());
        }
        _ => {
            assert!(collected_chunks.last().unwrap().usage.is_some());
        }
    }
}

pub fn create_tool_use_inference_request() -> ModelInferenceRequest<'static> {
    let messages = vec![RequestMessage {
        role: Role::User,
        content: vec![
            "What's the weather like in New York currently? Use the `get_weather` tool."
                .to_string()
                .into(),
        ],
    }];

    // Fine to leak during test execution
    let tool_config = Box::leak(Box::new(ToolCallConfig {
        tools_available: vec![ToolConfig::Static(&WEATHER_TOOL_CONFIG)],
        tool_choice: ToolChoice::Tool("get_weather".to_string()),
        parallel_tool_calls: false,
    }));

    ModelInferenceRequest {
        messages,
        system: None,
        tool_config: Some(tool_config),
        temperature: Some(0.7),
        max_tokens: Some(300),
        seed: None,
        stream: false,
        json_mode: JSONMode::Off,
        function_type: FunctionType::Chat,
        output_schema: None,
    }
}

pub async fn test_tool_use_inference_request_with_provider(provider: &ProviderConfig) {
    // Set up and make the inference request
    let inference_request = create_tool_use_inference_request();
    let client = reqwest::Client::new();
    let result = provider.infer(&inference_request, &client).await.unwrap();

    println!("Result: {result:#?}");

    // Check the result
    assert!(result.content.len() == 1, "{:#?}", result.content);
    let content = result.content.first().unwrap();
    match content {
        ContentBlock::ToolCall(tool_call) => {
            assert!(tool_call.name == "get_weather");

            let arguments: serde_json::Value = serde_json::from_str(&tool_call.arguments)
                .expect("Failed to parse tool call arguments");
            let arguments = arguments.as_object().unwrap();

            assert!(arguments.len() == 1 || arguments.len() == 2);
            assert!(arguments.keys().any(|key| key == "location"));
            assert!(arguments["location"] == "New York");

            // unit is optional
            if arguments.len() == 2 {
                assert!(arguments.keys().any(|key| key == "unit"));
                assert!(arguments["unit"] == "celsius" || arguments["unit"] == "fahrenheit");
            }
        }
        _ => panic!("Unexpected content block: {:?}", content),
    }
}

pub async fn test_tool_use_streaming_inference_request_with_provider(provider: &ProviderConfig) {
    // Set up and make the inference request
    let client = reqwest::Client::new();
    let mut inference_request = create_tool_use_inference_request();
    inference_request.stream = true;
    let result = provider
        .infer_stream(&inference_request, &client)
        .await
        .unwrap();

    // Collect the chunks
    let (chunk, mut stream) = result;
    let mut collected_chunks = vec![chunk];
    while let Some(chunk) = stream.next().await {
        let chunk = chunk.unwrap();
        println!("Chunk: {chunk:#?}");
        collected_chunks.push(chunk);
    }

    // Check tool name
    for chunk in &collected_chunks {
        match chunk.content.first() {
            Some(ContentBlockChunk::ToolCall(tool_call)) => {
                assert!(tool_call.name == "get_weather");
            }
            None => continue, // we might get empty chunks (e.g. the usage chunk)
            _ => panic!("Unexpected content block"),
        }
    }

    // Check tool arguments
    let arguments = collected_chunks
        .iter()
        .filter(|chunk| !chunk.content.is_empty())
        .map(|chunk| chunk.content.first().unwrap())
        .map(|content| match content {
            ContentBlockChunk::ToolCall(tool_call) => tool_call.arguments.clone(),
            _ => panic!("Unexpected content block: {:?}", content),
        })
        .collect::<Vec<String>>()
        .join("");

    let arguments: serde_json::Value = serde_json::from_str(&arguments).unwrap();
    let arguments = arguments.as_object().unwrap();

    assert!(arguments.len() == 1 || arguments.len() == 2);
    assert!(arguments.keys().any(|key| key == "location"));
    assert!(arguments["location"] == "New York");

    // `unit` is optional
    if arguments.len() == 2 {
        assert!(arguments.keys().any(|key| key == "unit"));
        assert!(arguments["unit"] == "celsius" || arguments["unit"] == "fahrenheit");
    }

    // Check the usage
    match provider {
        // NOTE: Azure and Together do not return usage for streaming tool use (to the best of our knowledge)
        ProviderConfig::Azure(AzureProvider { .. })
        | ProviderConfig::Together(TogetherProvider { .. }) => {
            assert!(collected_chunks.last().unwrap().usage.is_none());
        }
        _ => {
            assert!(collected_chunks.last().unwrap().usage.is_some());
        }
    }
}

fn create_tool_multi_turn_inference_request() -> ModelInferenceRequest<'static> {
    let messages = vec![
        RequestMessage {
            role: Role::User,
            content: vec![
                "What's the weather like in New York currently? Use the `get_weather` tool."
                    .to_string()
                    .into(),
            ],
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

    let tool_config = Box::leak(Box::new(ToolCallConfig {
        tools_available: vec![ToolConfig::Static(&WEATHER_TOOL_CONFIG)],
        tool_choice: ToolChoice::Auto,
        parallel_tool_calls: false,
    }));

    ModelInferenceRequest {
        messages,
        system: None,
        tool_config: Some(tool_config),
        temperature: Some(0.7),
        max_tokens: Some(100),
        seed: None,
        stream: false,
        json_mode: JSONMode::Off,
        function_type: FunctionType::Chat,
        output_schema: None,
    }
}

pub async fn test_tool_multi_turn_inference_request_with_provider(provider: &ProviderConfig) {
    // Set up and make the inference request
    let client = reqwest::Client::new();
    let inference_request = create_tool_multi_turn_inference_request();
    let result = provider.infer(&inference_request, &client).await.unwrap();

    println!("Result: {result:#?}");

    // Check the result
    assert!(result.content.len() == 1);
    let content = result.content.first().unwrap();
    match content {
        ContentBlock::Text(block) => {
            assert!(block.text.contains("New York"), "{}", block.text);
            assert!(block.text.contains("70"), "{}", block.text);
        }
        _ => panic!("Unexpected content block: {:?}", content),
    }
}

pub async fn test_tool_multi_turn_streaming_inference_request_with_provider(
    provider: &ProviderConfig,
) {
    // Set up and make the inference request
    let client = reqwest::Client::new();
    let mut inference_request = create_tool_multi_turn_inference_request();
    inference_request.stream = true;
    let result = provider
        .infer_stream(&inference_request, &client)
        .await
        .unwrap();

    // Collect the chunks
    let (chunk, mut stream) = result;
    let mut collected_chunks = vec![chunk];
    while let Some(chunk) = stream.next().await {
        let chunk = chunk.unwrap();
        println!("Chunk: {chunk:#?}");
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
    assert!(generation.contains("New York"), "{}", generation);
    assert!(generation.contains("70"), "{}", generation);

    // Check the usage
    match provider {
        // NOTE: Azure does not return usage for streaming inference (to the best of our knowledge)
        ProviderConfig::Azure(AzureProvider { .. }) => {
            assert!(collected_chunks.last().unwrap().usage.is_none());
        }
        _ => {
            assert!(collected_chunks.last().unwrap().usage.is_some());
        }
    }
}

pub fn create_json_mode_inference_request<'a>() -> ModelInferenceRequest<'a> {
    let messages = vec![RequestMessage {
        role: Role::User,
        content: vec!["What is 4+4?".to_string().into()],
    }];

    let system = Some(
        r#"\
        # Instructions\n\
        You are a helpful assistant who returns in the JSON form {"reasoning": "...", "answer": "..."}.\n\
        \n\
        # Examples\n\
        \n\
        {"reasoning": "3 + 3 = 6", "answer": "6"}\n\
        \n\
        {"reasoning": "Japan is in Asia. Tokyo is the capital of Japan. Therefore Tokyo is in Asia.", "answer": "Yes"}\n\
        \n\
        {"reasoning": "The square root of 9 is 3.", "answer": "3"}\n\
        "#.into(),
    );
    let max_tokens = Some(400);
    let temperature = Some(1.);
    let seed = Some(420);
    let output_schema = json!({
        "type": "object",
        "properties": {
            "reasoning": {"type": "string"},
            "answer": {"type": "string"}
        },
        "required": ["answer"]
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

pub async fn test_json_mode_inference_request_with_provider(provider: &ProviderConfig) {
    // Set up and make the inference request
    let inference_request = create_json_mode_inference_request();
    let client = reqwest::Client::new();
    let result = provider.infer(&inference_request, &client).await.unwrap();

    println!("Result: {result:#?}");

    // Check the result
    assert!(result.content.len() == 1);
    let content = result.content.first().unwrap();

    match content {
        ContentBlock::Text(block) => {
            let parsed_json: serde_json::Value = serde_json::from_str(&block.text).unwrap();
            let parsed_json = parsed_json.as_object().unwrap();

            assert!(parsed_json.len() == 1 || parsed_json.len() == 2);
            assert!(parsed_json.get("answer").unwrap().as_str().unwrap() == "8");

            // reasoning is optional
            if parsed_json.len() == 2 {
                assert!(parsed_json.keys().any(|key| key == "reasoning"));
            }
        }
        _ => panic!("Unexpected content block: {:?}", content),
    }
}

pub async fn test_json_mode_streaming_inference_request_with_provider(provider: &ProviderConfig) {
    // Set up and make the inference request
    let mut inference_request = create_json_mode_inference_request();
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
        println!("Chunk: {chunk:#?}");
        assert!(chunk.content.len() <= 1);
        collected_chunks.push(chunk);
    }

    // Parse the generation
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

    println!("Generation: {}", generation);

    // Ensure the generation is valid JSON
    let parsed_json: serde_json::Value = serde_json::from_str(&generation).unwrap();
    let parsed_json = parsed_json.as_object().unwrap();

    assert!(parsed_json.len() == 1 || parsed_json.len() == 2);
    assert!(parsed_json.get("answer").unwrap().as_str().unwrap() == "8");

    // `reasoning` is optional
    if parsed_json.len() == 2 {
        assert!(parsed_json.keys().any(|key| key == "reasoning"));
    }

    // Check the usage
    match provider {
        // NOTE: Azure and Together do not return usage for streaming JSON Mode inference (to the best of our knowledge)
        ProviderConfig::Azure(AzureProvider { .. })
        | ProviderConfig::Together(TogetherProvider { .. }) => {
            assert!(collected_chunks.last().unwrap().usage.is_none());
        }
        _ => {
            assert!(collected_chunks.last().unwrap().usage.is_some());
        }
    }
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
        })),
        strict: false,
    };
}
