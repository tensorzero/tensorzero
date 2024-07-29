use std::env;

use crate::integration::providers::common::{
    create_json_inference_request, create_simple_inference_request,
    create_streaming_inference_request, create_tool_inference_request,
};
use api::{
    inference::providers::openai::FireworksProvider,
    inference::providers::provider_trait::InferenceProvider, model::ProviderConfig,
};
use futures::StreamExt;
use secrecy::SecretString;

#[tokio::test]
async fn test_infer() {
    // Load API key from environment variable
    let api_key = env::var("FIREWORKS_API_KEY").expect("FIREWORKS_API_KEY must be set");
    let api_key = SecretString::new(api_key);
    let model_name = "accounts/fireworks/models/llama-v3-8b-instruct";
    let client = reqwest::Client::new();
    let inference_request = create_simple_inference_request();

    let provider_config = ProviderConfig::Fireworks {
        model_name: model_name.to_string(),
        api_key: Some(api_key),
    };
    let result = FireworksProvider::infer(&inference_request, &provider_config, &client).await;
    assert!(result.is_ok());
    assert!(result.unwrap().content.is_some());
}

#[tokio::test]
async fn test_infer_with_tool_calls() {
    // Load API key from environment variable
    let api_key = env::var("FIREWORKS_API_KEY").expect("FIREWORKS_API_KEY must be set");
    let api_key = SecretString::new(api_key);
    let model_name = "accounts/fireworks/models/firefunction-v2";
    let client = reqwest::Client::new();

    let inference_request = create_tool_inference_request();

    let provider_config = ProviderConfig::Fireworks {
        model_name: model_name.to_string(),
        api_key: Some(api_key),
    };
    let result = FireworksProvider::infer(&inference_request, &provider_config, &client).await;

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
    let api_key = env::var("FIREWORKS_API_KEY").expect("FIREWORKS_API_KEY must be set");
    let api_key = SecretString::new(api_key);
    let model_name = "accounts/fireworks/models/llama-v3-8b-instruct";
    let client = reqwest::Client::new();
    let inference_request = create_streaming_inference_request();

    let provider_config = ProviderConfig::Fireworks {
        model_name: model_name.to_string(),
        api_key: Some(api_key),
    };
    let result =
        FireworksProvider::infer_stream(&inference_request, &provider_config, &client).await;
    assert!(result.is_ok());
    let (chunk, mut stream) = result.unwrap();
    let mut collected_chunks = vec![chunk];
    while let Some(chunk) = stream.next().await {
        assert!(chunk.is_ok());
        collected_chunks.push(chunk.unwrap());
    }
    assert!(!collected_chunks.is_empty());
    // Here, we check the second chunk since fireworks sends an empty chunk at the beginning
    assert!(collected_chunks.get(1).unwrap().content.is_some());
    assert!(collected_chunks.last().unwrap().usage.is_some());
}

#[tokio::test]
async fn test_json_request() {
    // Load API key from environment variable
    let api_key = env::var("FIREWORKS_API_KEY").expect("FIREWORKS_API_KEY must be set");
    let api_key = SecretString::new(api_key);
    let model_name = "accounts/fireworks/models/llama-v3-8b-instruct";
    let client = reqwest::Client::new();
    let inference_request = create_json_inference_request();

    let provider_config = ProviderConfig::Fireworks {
        model_name: model_name.to_string(),
        api_key: Some(api_key),
    };
    let result = FireworksProvider::infer(&inference_request, &provider_config, &client).await;
    assert!(result.is_ok());
    let result = result.unwrap();
    assert!(result.content.is_some());
    // parse the result text and see if it matches the output schema
    let result_text = result.content.unwrap();
    let result_json: serde_json::Value = serde_json::from_str(&result_text).unwrap();
    assert!(result_json.get("thinking").is_some());
    assert!(result_json.get("answer").is_some());
}
