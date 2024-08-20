use gateway::inference::providers::provider_trait::InferenceProvider;
use gateway::inference::types::ContentBlock;
use gateway::{inference::providers::anthropic::AnthropicProvider, model::ProviderConfig};
use secrecy::SecretString;
use std::env;

use crate::providers::common::{
    create_tool_inference_request, test_simple_inference_request_with_provider,
    test_streaming_inference_request_with_provider,
};

/// Get a generic provider for testing
fn get_provider() -> ProviderConfig {
    let api_key = env::var("ANTHROPIC_API_KEY").expect("ANTHROPIC_API_KEY must be set");
    let api_key = Some(SecretString::new(api_key));

    ProviderConfig::Anthropic(AnthropicProvider {
        model_name: "claude-3-haiku-20240307".to_string(),
        api_key,
    })
}

#[tokio::test]
async fn test_simple_inference_request() {
    test_simple_inference_request_with_provider(get_provider()).await;
}

#[tokio::test]
async fn test_streaming_inference_request() {
    test_streaming_inference_request_with_provider(get_provider()).await;
}

#[tokio::test]
async fn test_infer_with_tool_calls() {
    // Load API key from environment variable
    let api_key = env::var("ANTHROPIC_API_KEY").expect("ANTHROPIC_API_KEY must be set");
    let api_key = SecretString::new(api_key);
    let model_name = "claude-3-haiku-20240307";
    let client = reqwest::Client::new();

    let inference_request = create_tool_inference_request();
    let provider = ProviderConfig::Anthropic(AnthropicProvider {
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
