use futures::StreamExt;
use secrecy::SecretString;
use std::env;

use gateway::inference::providers::{
    provider_trait::InferenceProvider, together::TogetherProvider,
};
use gateway::inference::types::{ContentBlock, Text};
use gateway::model::ProviderConfig;

use crate::providers::common::{
    create_json_inference_request, create_streaming_json_inference_request,
    create_tool_inference_request, test_simple_inference_request_with_provider,
    test_streaming_inference_request_with_provider, TestableProviderConfig,
};

crate::enforce_provider_tests!(TogetherProvider);

impl TestableProviderConfig for TogetherProvider {
    async fn get_simple_inference_request_provider() -> Option<ProviderConfig> {
        Some(get_provider())
    }

    async fn get_streaming_inference_request_provider() -> Option<ProviderConfig> {
        Some(get_provider())
    }
}

/// Get a generic provider for testing
fn get_provider() -> ProviderConfig {
    let api_key = env::var("TOGETHER_API_KEY").expect("TOGETHER_API_KEY must be set");
    let api_key = Some(SecretString::new(api_key));
    let model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo".to_string();

    ProviderConfig::Together(TogetherProvider {
        model_name,
        api_key,
    })
}

#[tokio::test]
async fn test_infer_with_tool_calls() {
    // Load API key from environment variable
    let api_key = env::var("TOGETHER_API_KEY").expect("TOGETHER_API_KEY must be set");
    let api_key = SecretString::new(api_key);
    let model_name = "mistralai/Mixtral-8x7B-Instruct-v0.1";
    let client = reqwest::Client::new();

    let inference_request = create_tool_inference_request();

    let provider = ProviderConfig::Together(TogetherProvider {
        model_name: model_name.to_string(),
        api_key: Some(api_key),
    });
    let result = provider.infer(&inference_request, &client).await;

    assert!(result.is_ok());
    let response = result.unwrap();
    assert!(response.content.len() == 1);
    let content = response.content.first().unwrap();
    match content {
        ContentBlock::ToolCall(tool_call) => {
            assert!(tool_call.name == "get_weather");
            let arguments: serde_json::Value = serde_json::from_str(&tool_call.arguments)
                .expect("Failed to parse tool call arguments");
            assert!(arguments.get("location").is_some());
        }
        _ => panic!("Expected a tool call content block"),
    }
}

#[tokio::test]
async fn test_json_request() {
    // Load API key from environment variable
    let api_key = env::var("TOGETHER_API_KEY").expect("TOGETHER_API_KEY must be set");
    let api_key = SecretString::new(api_key);
    let model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo";
    let client = reqwest::Client::new();
    let inference_request = create_json_inference_request();

    let provider = ProviderConfig::Together(TogetherProvider {
        model_name: model_name.to_string(),
        api_key: Some(api_key),
    });
    let result = provider.infer(&inference_request, &client).await;
    assert!(result.is_ok());
    let result = result.unwrap();
    assert!(result.content.len() == 1);
    let content = result.content.first().unwrap();
    match content {
        ContentBlock::Text(Text { text }) => {
            // parse the result text and see if it matches the output schema
            let result_json: serde_json::Value = serde_json::from_str(text)
                .map_err(|_| format!(r#"Failed to parse JSON: "{text}""#))
                .unwrap();
            assert!(result_json.get("honest_answer").is_some());
            assert!(result_json.get("mischevious_answer").is_some());
        }
        _ => panic!("Expected a text content block"),
    }
}

#[tokio::test]
async fn test_streaming_json_request() {
    // Load API key from environment variable
    let api_key = env::var("TOGETHER_API_KEY").expect("TOGETHER_API_KEY must be set");
    let api_key = SecretString::new(api_key);
    let model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo";
    let client = reqwest::Client::new();
    let inference_request = create_streaming_json_inference_request();

    let provider = ProviderConfig::Together(TogetherProvider {
        model_name: model_name.to_string(),
        api_key: Some(api_key),
    });
    let result = provider.infer_stream(&inference_request, &client).await;
    assert!(result.is_ok());
    let (chunk, mut stream) = result.unwrap();
    let mut collected_chunks = vec![chunk];
    while let Some(chunk) = stream.next().await {
        assert!(chunk.is_ok());
        collected_chunks.push(chunk.unwrap());
    }
    assert!(!collected_chunks.is_empty());
    // Fourth as an arbitrary middle chunk, the first and last may contain only metadata
    assert!(collected_chunks[4].content.len() == 1);
    // Note: Together does not seem to return usage information for streaming JSON requests
    // assert!(collected_chunks.last().unwrap().usage.is_some());
}
