use futures::StreamExt;
use gateway::inference::types::{ContentBlock, Text};
use secrecy::SecretString;
use std::env;

use gateway::inference::providers::{azure::AzureProvider, provider_trait::InferenceProvider};
use gateway::model::ProviderConfig;

use crate::providers::common::{
    create_json_inference_request, create_simple_inference_request,
    create_streaming_inference_request, create_streaming_json_inference_request,
    create_tool_inference_request,
};

#[tokio::test]
async fn test_infer() {
    // Load API key from environment variable
    let api_key = env::var("AZURE_OPENAI_API_KEY")
        .expect("Environment variable AZURE_OPENAI_API_KEY must be set");
    let api_base = env::var("AZURE_OPENAI_API_BASE")
        .expect("Environment variable AZURE_OPENAI_API_BASE must be set");
    let deployment_id = env::var("AZURE_OPENAI_DEPLOYMENT_ID")
        .expect("Environment variable AZURE_OPENAI_DEPLOYMENT_ID must be set");
    let api_key = SecretString::new(api_key);
    let model_name = "gpt-4o-mini";
    let client = reqwest::Client::new();
    let inference_request = create_simple_inference_request();
    let provider = ProviderConfig::Azure(AzureProvider {
        model_name: model_name.to_string(),
        api_base,
        api_key: Some(api_key),
        deployment_id,
    });
    let result = provider.infer(&inference_request, &client).await;
    assert!(result.is_ok());
    let result = result.unwrap();
    assert!(result.content.len() == 1);
    let content = result.content.first().unwrap();
    match content {
        ContentBlock::Text(Text { text }) => {
            assert!(!text.is_empty());
        }
        _ => panic!("Expected a text content block"),
    }
}

#[tokio::test]
async fn test_infer_with_tool_calls() {
    // Load API key from environment variable
    let api_key = env::var("AZURE_OPENAI_API_KEY")
        .expect("Environment variable AZURE_OPENAI_API_KEY must be set");
    let api_base = env::var("AZURE_OPENAI_API_BASE")
        .expect("Environment variable AZURE_OPENAI_API_BASE must be set");
    let deployment_id = env::var("AZURE_OPENAI_DEPLOYMENT_ID")
        .expect("Environment variable AZURE_OPENAI_DEPLOYMENT_ID must be set");
    let api_key = SecretString::new(api_key);
    let model_name = "gpt-4o-mini";
    let client = reqwest::Client::new();

    let inference_request = create_tool_inference_request();
    let provider = ProviderConfig::Azure(AzureProvider {
        model_name: model_name.to_string(),
        api_base,
        api_key: Some(api_key),
        deployment_id,
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
async fn test_infer_stream() {
    // Load API key from environment variable
    let api_key = env::var("AZURE_OPENAI_API_KEY")
        .expect("Environment variable AZURE_OPENAI_API_KEY must be set");
    let api_base = env::var("AZURE_OPENAI_API_BASE")
        .expect("Environment variable AZURE_OPENAI_API_BASE must be set");
    let deployment_id = env::var("AZURE_OPENAI_DEPLOYMENT_ID")
        .expect("Environment variable AZURE_OPENAI_DEPLOYMENT_ID must be set");
    let api_key = SecretString::new(api_key);
    let model_name = "gpt-4o-mini";
    let client = reqwest::Client::new();
    let inference_request = create_streaming_inference_request();

    let provider = ProviderConfig::Azure(AzureProvider {
        model_name: model_name.to_string(),
        api_base,
        api_key: Some(api_key),
        deployment_id,
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
    // NOTE: Azure OpenAI service does not return streaming usage and to the best of our knowledge
    // there's no way to get it to do so.
    // assert!(collected_chunks.last().unwrap().usage.is_some());
}

#[tokio::test]
async fn test_json_request() {
    // Load API key from environment variable
    let api_key = env::var("AZURE_OPENAI_API_KEY")
        .expect("Environment variable AZURE_OPENAI_API_KEY must be set");
    let api_base = env::var("AZURE_OPENAI_API_BASE")
        .expect("Environment variable AZURE_OPENAI_API_BASE must be set");
    let deployment_id = env::var("AZURE_OPENAI_DEPLOYMENT_ID")
        .expect("Environment variable AZURE_OPENAI_DEPLOYMENT_ID must be set");
    let api_key = SecretString::new(api_key);
    let model_name = "gpt-4o-mini";
    let client = reqwest::Client::new();
    let inference_request = create_json_inference_request();

    let provider = ProviderConfig::Azure(AzureProvider {
        model_name: model_name.to_string(),
        api_base,
        api_key: Some(api_key),
        deployment_id,
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
    let api_key = env::var("AZURE_OPENAI_API_KEY")
        .expect("Environment variable AZURE_OPENAI_API_KEY must be set");
    let api_base = env::var("AZURE_OPENAI_API_BASE")
        .expect("Environment variable AZURE_OPENAI_API_BASE must be set");
    let deployment_id = env::var("AZURE_OPENAI_DEPLOYMENT_ID")
        .expect("Environment variable AZURE_OPENAI_DEPLOYMENT_ID must be set");
    let api_key = SecretString::new(api_key);
    let model_name = "gpt-4o-mini";
    let client = reqwest::Client::new();
    let inference_request = create_streaming_json_inference_request();

    let provider = ProviderConfig::Azure(AzureProvider {
        model_name: model_name.to_string(),
        api_base,
        api_key: Some(api_key),
        deployment_id,
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
}
