#![cfg(feature = "integration_tests")]
#![cfg(test)]
use std::env;

use crate::inference::providers::common::{
    create_simple_inference_request, create_streaming_inference_request,
    create_tool_inference_request,
};
use api::{config_parser::ProviderConfig, inference::providers::openai};
use futures::StreamExt;
use secrecy::SecretString;

#[tokio::test]
async fn test_infer() {
    // Load API key from environment variable
    let api_key = env::var("OPENAI_API_KEY").expect("OPENAI_API_KEY must be set");
    let api_key = SecretString::new(api_key);
    let model_name = "gpt-4o-mini";
    let client = reqwest::Client::new();
    let inference_request = create_simple_inference_request();

    let base_url = None;
    let provider_config = ProviderConfig::OpenAI {
        model_name: model_name.to_string(),
        api_base: base_url,
    };
    let result = openai::infer(&inference_request, &provider_config, &client, &api_key).await;
    assert!(result.is_ok());
    assert!(result.unwrap().content.is_some());
}

#[tokio::test]
async fn test_infer_with_tool_calls() {
    // Load API key from environment variable
    let api_key = env::var("OPENAI_API_KEY").expect("OPENAI_API_KEY must be set");
    let api_key = SecretString::new(api_key);
    let model_name = "gpt-4o-mini";
    let client = reqwest::Client::new();

    let inference_request = create_tool_inference_request();

    let base_url = None;
    let provider_config = ProviderConfig::OpenAI {
        model_name: model_name.to_string(),
        api_base: base_url,
    };
    let result = openai::infer(&inference_request, &provider_config, &client, &api_key).await;

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

#[tokio::test]
async fn test_infer_stream() {
    // Load API key from environment variable
    let api_key = env::var("OPENAI_API_KEY").expect("OPENAI_API_KEY must be set");
    let api_key = SecretString::new(api_key);
    let model_name = "gpt-4o-mini";
    let client = reqwest::Client::new();
    let inference_request = create_streaming_inference_request();

    let base_url = None;
    let provider_config = ProviderConfig::OpenAI {
        model_name: model_name.to_string(),
        api_base: base_url,
    };
    let result =
        openai::infer_stream(&inference_request, &provider_config, &client, &api_key).await;
    assert!(result.is_ok());
    let mut stream = result.unwrap();
    let mut collected_chunks = Vec::new();
    while let Some(chunk) = stream.next().await {
        assert!(chunk.is_ok());
        collected_chunks.push(chunk.unwrap());
    }
    assert!(!collected_chunks.is_empty());
    assert!(collected_chunks.first().unwrap().content.is_some());
    assert!(collected_chunks.last().unwrap().usage.is_some());
}
