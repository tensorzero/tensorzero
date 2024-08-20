use secrecy::SecretString;
use std::env;

use gateway::inference::providers::{openai::OpenAIProvider, provider_trait::InferenceProvider};
use gateway::inference::types::{ContentBlock, JSONMode};
use gateway::model::ProviderConfig;

use crate::providers::common::{create_json_mode_inference_request, TestableProviderConfig};

crate::generate_provider_tests!(OpenAIProvider);

impl TestableProviderConfig for OpenAIProvider {
    async fn get_simple_inference_request_providers() -> Vec<ProviderConfig> {
        vec![get_provider()]
    }

    async fn get_streaming_inference_request_providers() -> Vec<ProviderConfig> {
        vec![get_provider()]
    }

    async fn get_tool_use_inference_request_providers() -> Vec<ProviderConfig> {
        vec![get_provider()]
    }

    async fn get_tool_use_streaming_inference_request_providers() -> Vec<ProviderConfig> {
        vec![get_provider()]
    }

    async fn get_json_mode_inference_request_providers() -> Vec<ProviderConfig> {
        vec![get_provider()]
    }

    async fn get_json_mode_streaming_inference_request_providers() -> Vec<ProviderConfig> {
        vec![get_provider()]
    }
}

/// Get a generic provider for testing
fn get_provider() -> ProviderConfig {
    let api_key = env::var("OPENAI_API_KEY").expect("OPENAI_API_KEY must be set");
    let api_key = Some(SecretString::new(api_key));
    let model_name = "gpt-4o-mini".to_string();
    let api_base = None;

    ProviderConfig::OpenAI(OpenAIProvider {
        model_name,
        api_base,
        api_key,
    })
}

#[tokio::test]
async fn test_json_mode_inference_request_with_provider_strict() {
    // Set up and make the inference request
    let provider = get_provider();
    let mut inference_request = create_json_mode_inference_request();
    inference_request.json_mode = JSONMode::Strict;
    let client = reqwest::Client::new();
    let result = provider.infer(&inference_request, &client).await.unwrap();

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
