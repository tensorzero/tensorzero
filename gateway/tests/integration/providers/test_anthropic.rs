use crate::integration::providers::common::{
    create_simple_inference_request, create_streaming_inference_request,
    create_tool_inference_request,
};
use futures::StreamExt;
use gateway::inference::providers::provider_trait::InferenceProvider;
use gateway::inference::types::{ContentBlock, Text};
use gateway::{inference::providers::anthropic::AnthropicProvider, model::ProviderConfig};
use secrecy::SecretString;
use std::env;

#[tokio::test]
async fn test_infer() {
    // Load API key from environment variable
    let api_key = env::var("ANTHROPIC_API_KEY").expect("ANTHROPIC_API_KEY must be set");
    let api_key = SecretString::new(api_key);
    let model_name = "claude-3-haiku-20240307";
    let config = ProviderConfig::Anthropic {
        model_name: model_name.to_string(),
        api_key: Some(api_key),
    };
    let client = reqwest::Client::new();
    let inference_request = create_simple_inference_request();
    let result = AnthropicProvider::infer(&inference_request, &config, &client).await;
    assert!(result.is_ok());
    let result = result.unwrap();
    assert!(result.content.len() == 1);
    let content = result.content.first().unwrap();
    match content {
        ContentBlock::Text(Text { text }) => {
            assert!(!text.is_empty());
        }
        _ => unreachable!(),
    }
}

#[tokio::test]
async fn test_infer_stream() {
    // Load API key from environment variable
    let api_key = env::var("ANTHROPIC_API_KEY").expect("ANTHROPIC_API_KEY must be set");
    let api_key = SecretString::new(api_key);
    let model_name = "claude-3-haiku-20240307";
    let config = ProviderConfig::Anthropic {
        model_name: model_name.to_string(),
        api_key: Some(api_key),
    };
    let client = reqwest::Client::new();
    let inference_request = create_streaming_inference_request();
    let result = AnthropicProvider::infer_stream(&inference_request, &config, &client).await;
    assert!(result.is_ok());
    let (chunk, mut stream) = result.unwrap();
    let mut collected_chunks = vec![chunk];
    while let Some(chunk) = stream.next().await {
        assert!(chunk.is_ok());
        collected_chunks.push(chunk.unwrap());
    }
    assert!(!collected_chunks.is_empty());
    // Fourth as an arbitrary middle chunk, the first and last contain only metadata for Anthropic
    assert!(collected_chunks[4].content.len() == 1);
    assert!(collected_chunks.last().unwrap().usage.is_some());
}

#[tokio::test]
async fn test_infer_with_tool_calls() {
    // Load API key from environment variable
    let api_key = env::var("ANTHROPIC_API_KEY").expect("ANTHROPIC_API_KEY must be set");
    let api_key = SecretString::new(api_key);
    let model_name = "claude-3-haiku-20240307";
    let client = reqwest::Client::new();

    let inference_request = create_tool_inference_request();
    let config = ProviderConfig::Anthropic {
        model_name: model_name.to_string(),
        api_key: Some(api_key),
    };
    let result = AnthropicProvider::infer(&inference_request, &config, &client).await;

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
        _ => unreachable!(),
    }
}
