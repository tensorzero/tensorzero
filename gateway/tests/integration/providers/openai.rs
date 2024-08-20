use futures::StreamExt;
use secrecy::SecretString;
use std::env;

use gateway::inference::providers::{openai::OpenAIProvider, provider_trait::InferenceProvider};
use gateway::inference::types::{ContentBlock, JSONMode, Text};
use gateway::model::ProviderConfig;

use crate::providers::common::{
    create_json_inference_request, create_streaming_json_inference_request,
    create_tool_inference_request, test_simple_inference_request_with_provider,
    test_streaming_inference_request_with_provider,
};

#[tokio::test]
async fn test_simple_inference_request() {
    // Load API key from environment variable
    let api_key = env::var("OPENAI_API_KEY").expect("OPENAI_API_KEY must be set");
    let api_key = SecretString::new(api_key);
    let model_name = "gpt-4o-mini";
    let base_url = None;
    let provider = ProviderConfig::OpenAI(OpenAIProvider {
        model_name: model_name.to_string(),
        api_base: base_url,
        api_key: Some(api_key),
    });

    test_simple_inference_request_with_provider(provider).await;
}

#[tokio::test]
async fn test_streaming_inference_request() {
    // Load API key from environment variable
    let api_key = env::var("OPENAI_API_KEY").expect("OPENAI_API_KEY must be set");
    let api_key = SecretString::new(api_key);
    let model_name = "gpt-4o-mini";
    let provider = ProviderConfig::OpenAI(OpenAIProvider {
        model_name: model_name.to_string(),
        api_base: None,
        api_key: Some(api_key),
    });

    test_streaming_inference_request_with_provider(provider).await;
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
    let provider = ProviderConfig::OpenAI(OpenAIProvider {
        model_name: model_name.to_string(),
        api_base: base_url,
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
        _ => unreachable!(),
    }
}

#[tokio::test]
async fn test_json_request() {
    // Load API key from environment variable
    let api_key = env::var("OPENAI_API_KEY").expect("OPENAI_API_KEY must be set");
    let api_key = SecretString::new(api_key);
    let model_name = "gpt-4o-mini";
    let client = reqwest::Client::new();
    let inference_request = create_json_inference_request();

    let provider = ProviderConfig::OpenAI(OpenAIProvider {
        model_name: model_name.to_string(),
        api_base: None,
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
        _ => unreachable!(),
    }
}

#[tokio::test]
async fn test_json_request_strict() {
    // Load API key from environment variable
    let api_key = env::var("OPENAI_API_KEY").expect("OPENAI_API_KEY must be set");
    let api_key = SecretString::new(api_key);
    let model_name = "gpt-4o-mini";
    let client = reqwest::Client::new();
    let mut inference_request = create_json_inference_request();
    inference_request.json_mode = JSONMode::Strict;

    let provider = ProviderConfig::OpenAI(OpenAIProvider {
        model_name: model_name.to_string(),
        api_base: None,
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
        _ => unreachable!(),
    }
}

#[tokio::test]
async fn test_streaming_json_request() {
    // Load API key from environment variable
    let api_key = env::var("OPENAI_API_KEY").expect("OPENAI_API_KEY must be set");
    let api_key = SecretString::new(api_key);
    let model_name = "gpt-4o-mini";
    let client = reqwest::Client::new();
    let inference_request = create_streaming_json_inference_request();

    let provider = ProviderConfig::OpenAI(OpenAIProvider {
        model_name: model_name.to_string(),
        api_base: None,
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
    assert!(collected_chunks.last().unwrap().usage.is_some());
}
